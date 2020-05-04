# coding: utf-8
# Copyright 2020 Masamichi J. Miyama
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================

from __future__ import division

import collections.abc as abc
import itertools
from warnings import warn

import numpy as np
import dimod

from six import iteritems, itervalues

import dimod
import minorminer

from dwave.embedding import (target_to_source, unembed_sampleset, embed_bqm,
                             chain_to_quadratic, MinimizeEnergy)
from dwave.embedding.chain_breaks import majority_vote, broken_chains
from dwave.system.warnings import WarningHandler, WarningAction
from dwave.system import DWaveSampler, EmbeddingComposite

class MinimizeEnergyComposite(dimod.ComposedSampler):
    """Maps problems to a structured sampler.
    Automatically minor-embeds a problem into a structured sampler such as a
    D-Wave system. A new minor-embedding is calculated each time one of its
    sampling methods is called. 
    This is a modified version based on the EmbeddingComposite class 
    implementation code included in D-Wave Ocean SDK so that MinimizeEnergy 
    can be used easily as a method to recover from chain-breaks.
    Args:
        child_sampler (:class:`dimod.Sampler`):
            A dimod sampler, such as a :obj:`.DWaveSampler`, that accepts
            only binary quadratic models of a particular structure.
        find_embedding (function, optional):
            A function `find_embedding(S, T, **kwargs)` where `S` and `T`
            are edgelists. The function can accept additional keyword arguments.
            Defaults to :func:`minorminer.find_embedding`.
        embedding_parameters (dict, optional):
            If provided, parameters are passed to the embedding method as
            keyword arguments.
        scale_aware (bool, optional, default=False):
            Pass chain interactions to child samplers that accept an `ignored_interactions`
            parameter.
        child_structure_search (function, optional):
            A function `child_structure_search(sampler)` that accepts a sampler
            and returns the :attr:`dimod.Structured.structure`.
            Defaults to :func:`dimod.child_structure_dfs`.
    Examples:
       >>> from dwave.system import DWaveSampler
       ...
       >>> sampler = MinimizeEnergyComposite(DWaveSampler())
       >>> h = {'a': -1., 'b': 2}
       >>> J = {('a', 'b'): 1.5}
       >>> sampleset = sampler.sample_ising(h, J, num_reads=100)
       >>> sampleset.first.energy
       -4.5
    """
    def __init__(self, child_sampler,
                 find_embedding=minorminer.find_embedding,
                 embedding_parameters=None,
                 scale_aware=False,
                 child_structure_search=dimod.child_structure_dfs
                 ):

        self.children = [child_sampler]

        # keep any embedding parameters around until later, because we might
        # want to overwrite them
        self.embedding_parameters = embedding_parameters or {}
        self.find_embedding = find_embedding

        # set the parameters
        self.parameters = parameters = child_sampler.parameters.copy()
        parameters.update(chain_strength=[],
                          chain_break_method=[],
                          chain_break_fraction=[],
                          embedding_parameters=[],
                          return_embedding=[],
                          warnings=[],
                          )

        # set the properties
        self.properties = dict(child_properties=child_sampler.properties.copy())

        # track the child's structure. We use a dfs in case intermediate
        # composites are not structured. We could expose multiple different
        # searches but since (as of 14 june 2019) all composites have single
        # children, just doing dfs seems safe for now.
        self.target_structure = child_structure_search(child_sampler)

        self.scale_aware = bool(scale_aware)

    parameters = None  # overwritten by init
    """dict[str, list]: Parameters in the form of a dict.
    For an instantiated composed sampler, keys are the keyword parameters
    accepted by the child sampler and parameters added by the composite.
    """

    children = None  # overwritten by init
    """list [child_sampler]: List containing the structured sampler."""

    properties = None  # overwritten by init
    """dict: Properties in the form of a dict.
    Contains the properties of the child sampler.
    """

    return_embedding_default = False
    """Defines the default behaviour for :meth:`.sample`'s `return_embedding`
    kwarg.
    """

    warnings_default = WarningAction.IGNORE
    """Defines the default behabior for :meth:`.sample`'s `warnings` kwarg.
    """

    @staticmethod
    def _unembed_sampleset(target_sampleset, embedding, source_bqm,
                           chain_break_method=None, chain_break_fraction=False,
                           return_embedding=False):
        """Unembed a samples set.

        Given samples from a target binary quadratic model (BQM), construct a sample
        set for a source BQM by unembedding.
        This is a modified version based on the method `dwave.embedding.unembed_sampleset`
        so that MinimizeEnergy can be chosen as the `chain_break_method`.

        Args:
            target_sampleset (:obj:`dimod.SampleSet`):
                Sample set from the target BQM.

            embedding (dict):
                Mapping from source graph to target graph as a dict of form
                {s: {t, ...}, ...}, where s is a source variable and t is a target
                variable.

            source_bqm (:obj:`dimod.BinaryQuadraticModel`):
                Source BQM.

            chain_break_method (function/list, optional):
                Method or methods used to resolve chain breaks. If multiple methods
                are given, the results are concatenated and a new field called
                "chain_break_method" specifying the index of the method is appended
                to the sample set.
                See :mod:`dwave.embedding.chain_breaks`.

            chain_break_fraction (bool, optional, default=False):
                Add a `chain_break_fraction` field to the unembedded :obj:`dimod.SampleSet`
                with the fraction of chains broken before unembedding.

            return_embedding (bool, optional, default=False):
                If True, the embedding is added to :attr:`dimod.SampleSet.info`
                of the returned sample set. Note that if an `embedding` key
                already exists in the sample set then it is overwritten.

        Returns:
            :obj:`.SampleSet`: Sample set in the source BQM.

        Examples:
           This example unembeds from a square target graph samples of a triangular
           source BQM.

            >>> # Triangular binary quadratic model and an embedding
            >>> J = {('a', 'b'): -1, ('b', 'c'): -1, ('a', 'c'): -1}
            >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, J)
            >>> embedding = {'a': [0, 1], 'b': [2], 'c': [3]}
            >>> # Samples from the embedded binary quadratic model
            >>> samples = [{0: -1, 1: -1, 2: -1, 3: -1},  # [0, 1] is unbroken
            ...            {0: -1, 1: +1, 2: +1, 3: +1}]  # [0, 1] is broken
            >>> energies = [-3, 1]
            >>> embedded = dimod.SampleSet.from_samples(samples, dimod.SPIN, energies)
            >>> # Unembed
            >>> samples = dwave.embedding.unembed_sampleset(embedded, embedding, bqm)
            >>> samples.record.sample   # doctest: +SKIP
            array([[-1, -1, -1],
                   [ 1,  1,  1]], dtype=int8)

        """

        if chain_break_method is None:
            chain_break_method = majority_vote
        elif isinstance(chain_break_method, abc.Sequence):
            # we want to apply multiple CBM and then combine
            samplesets = [unembed_sampleset(target_sampleset, embedding,
                                            source_bqm, chain_break_method=cbm,
                                            chain_break_fraction=chain_break_fraction)
                        for cbm in chain_break_method]
            sampleset = dimod.sampleset.concatenate(samplesets)

            # Add a new data field tracking which came from
            # todo: add this functionality to dimod
            cbm_idxs = np.empty(len(sampleset), dtype=np.int)

            start = 0
            for i, ss in enumerate(samplesets):
                cbm_idxs[start:start+len(ss)] = i
                start += len(ss)

            new = np.lib.recfunctions.append_fields(sampleset.record,
                                                    'chain_break_method', cbm_idxs,
                                                    asrecarray=True, usemask=False)

            return type(sampleset)(new, sampleset.variables, sampleset.info,
                                   sampleset.vartype)

        variables = list(source_bqm.variables)  # need this ordered
        try:
            chains = [embedding[v] for v in variables]
        except KeyError:
            raise ValueError("given bqm does not match the embedding")

        record = target_sampleset.record

        unembedded = None
        idxs = None
        if chain_break_method is MinimizeEnergy:
            # Use dwave.embedding.MinimizeEnergy to recover from chain-brakes 
            cbm = MinimizeEnergy(source_bqm, embedding)
            unembedded, idxs = cbm(target_sampleset, chains)
        else:
            unembedded, idxs = chain_break_method(target_sampleset, chains)

        reserved = {'sample', 'energy'}
        vectors = {name: record[name][idxs]
                   for name in record.dtype.names if name not in reserved}

        if chain_break_fraction:
            vectors['chain_break_fraction'] = broken_chains(target_sampleset, chains).mean(axis=1)[idxs]

        info = target_sampleset.info.copy()

        if return_embedding:
            embedding_context = dict(embedding=embedding,
                                     chain_break_method=chain_break_method.__name__)
            info.update(embedding_context=embedding_context)

        return dimod.SampleSet.from_samples_bqm((unembedded, variables),
                                                source_bqm,
                                                info=info,
                                                **vectors)

    def sample(self, bqm, chain_strength=1.0,
               chain_break_method=MinimizeEnergy,
               chain_break_fraction=True,
               embedding_parameters=None,
               return_embedding=None,
               warnings=None,
               **parameters):
        """Sample from the provided binary quadratic model.
        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.
            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between
                variables to create chains. The energy penalty of chain breaks
                is 2 * `chain_strength`.
            chain_break_method (function, optional):
                Method used to resolve chain breaks during sample unembedding.
                See :func:`~dwave.embedding.unembed_sampleset`.
            chain_break_fraction (bool, optional, default=True):
                Add a `chain_break_fraction` field to the unembedded response with
                the fraction of chains broken before unembedding.
            embedding_parameters (dict, optional):
                If provided, parameters are passed to the embedding method as
                keyword arguments. Overrides any `embedding_parameters` passed
                to the constructor.
            return_embedding (bool, optional):
                If True, the embedding, chain strength, chain break method and
                embedding parameters are added to :attr:`dimod.SampleSet.info`
                of the returned sample set. The default behaviour is defined
                by :attr:`return_embedding_default`, which itself defaults to
                False.
            warnings (:class:`~dwave.system.warnings.WarningAction`, optional):
                Defines what warning action to take, if any. See
                :mod:`~dwave.system.warnings`. The default behaviour is defined
                by :attr:`warnings_default`, which itself defaults to
                :class:`~dwave.system.warnings.IGNORE`
            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.
        Returns:
            :obj:`dimod.SampleSet`
        Examples:
            See the example in :class:`EmbeddingComposite`.
        """
        if return_embedding is None:
            return_embedding = self.return_embedding_default

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = self.target_structure

        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        # get the embedding
        if embedding_parameters is None:
            embedding_parameters = self.embedding_parameters
        else:
            # we want the parameters provided to the constructor, updated with
            # the ones provided to the sample method. To avoid the extra copy
            # we do an update, avoiding the keys that would overwrite the
            # sample-level embedding parameters
            embedding_parameters.update((key, val)
                                        for key, val in self.embedding_parameters
                                        if key not in embedding_parameters)

        embedding = self.find_embedding(source_edgelist, target_edgelist,
                                        **embedding_parameters)

        if warnings is None:
            warnings = self.warnings_default
        elif 'warnings' in child.parameters:
            parameters.update(warnings=warnings)

        warninghandler = WarningHandler(warnings)

        warninghandler.chain_strength(bqm, chain_strength, embedding)
        warninghandler.chain_length(embedding)

        if bqm and not embedding:
            raise ValueError("no embedding found")

        bqm_embedded = embed_bqm(bqm, embedding, target_adjacency,
                                 chain_strength=chain_strength,
                                 smear_vartype=dimod.SPIN)

        if 'initial_state' in parameters:
            # if initial_state was provided in terms of the source BQM, we want
            # to modify it to now provide the initial state for the target BQM.
            # we do this by spreading the initial state values over the
            # chains
            state = parameters['initial_state']
            parameters['initial_state'] = {u: state[v]
                                           for v, chain in embedding.items()
                                           for u in chain}

        if self.scale_aware and 'ignored_interactions' in child.parameters:

            ignored = []
            for chain in embedding.values():
                # just use 0 as a null value because we don't actually need
                # the biases, just the interactions
                ignored.extend(chain_to_quadratic(chain, target_adjacency, 0))

            parameters['ignored_interactions'] = ignored

        response = child.sample(bqm_embedded, **parameters)

        warninghandler.chain_break(response, embedding)

        sampleset = MinimizeEnergyComposite._unembed_sampleset(response, embedding, source_bqm=bqm,
                                                               chain_break_method=chain_break_method,
                                                               chain_break_fraction=chain_break_fraction,
                                                               return_embedding=return_embedding)


        if return_embedding:
            embedding_context = dict(embedding=embedding,
                                     chain_break_method=chain_break_method.__name__,
                                     embedding_parameters=embedding_parameters,
                                     chain_strength=chain_strength)
            sampleset.info.update(embedding_context=embedding_context)

        if chain_break_fraction and len(sampleset):
            warninghandler.issue("All samples have broken chains",
                                 func=lambda: (sampleset.record.chain_break_fraction.all(), None))

        if warninghandler.action is WarningAction.SAVE:
            # we're done with the warning handler so we can just pass the list
            # off, if later we want to pass in a handler or similar we should
            # do a copy
            sampleset.info.setdefault('warnings', []).extend(warninghandler.saved)

        return sampleset

class LazyFixedMinimizeEnergyComposite(MinimizeEnergyComposite, dimod.Structured):
    """Maps problems to the structure of its first given problem.
    This composite reuses the minor-embedding found for its first given problem
    without recalculating a new minor-embedding for subsequent calls of its
    sampling methods.
    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler.
        find_embedding (function, default=:func:`minorminer.find_embedding`):
            A function `find_embedding(S, T, **kwargs)` where `S` and `T`
            are edgelists. The function can accept additional keyword arguments.
            The function is used to find the embedding for the first problem
            solved.
        embedding_parameters (dict, optional):
            If provided, parameters are passed to the embedding method as keyword
            arguments.
    Examples:
        >>> from dwave.system import DWaveSampler
        ...
        >>> sampler = LazyFixedMinimizeEnergyComposite(DWaveSampler())
        >>> sampler.nodelist is None  # no structure prior to first sampling
        True
        >>> __ = sampler.sample_ising({}, {('a', 'b'): -1})
        >>> sampler.nodelist  # has structure based on given problem
        ['a', 'b']
    """

    @property
    def nodelist(self):
        """list: Nodes available to the composed sampler."""
        try:
            return self._nodelist
        except AttributeError:
            pass

        if self.adjacency is None:
            return None

        self._nodelist = nodelist = list(self.adjacency)

        # makes it a lot easier for the user if the list can be sorted, so we
        # try
        try:
            nodelist.sort()
        except TypeError:
            # python3 cannot sort unlike types
            pass

        return nodelist

    @property
    def edgelist(self):
        """list: Edges available to the composed sampler."""
        try:
            return self._edgelist
        except AttributeError:
            pass

        adj = self.adjacency

        if adj is None:
            return None

        # remove duplicates by putting into a set
        edges = set()
        for u in adj:
            for v in adj[u]:
                try:
                    edge = (u, v) if u <= v else (v, u)
                except TypeError:
                    # Py3 does not allow sorting of unlike types
                    if (v, u) in edges:
                        continue
                    edge = (u, v)

                edges.add(edge)

        self._edgelist = edgelist = list(edges)

        # makes it a lot easier for the user if the list can be sorted, so we
        # try
        try:
            edgelist.sort()
        except TypeError:
            # python3 cannot sort unlike types
            pass

        return edgelist

    @property
    def adjacency(self):
        """dict[variable, set]: Adjacency structure for the composed sampler."""
        try:
            return self._adjacency
        except AttributeError:
            pass

        if self.embedding is None:
            return None

        self._adjacency = adj = target_to_source(self.target_structure.adjacency,
                                                 self.embedding)

        return adj

    embedding = None
    """Embedding used to map binary quadratic models to the child sampler."""

    def _fix_embedding(self, embedding):
        # save the embedding and overwrite the find_embedding function
        self.embedding = embedding
        self.properties.update(embedding=embedding)

        def find_embedding(S, T):
            return embedding

        self.find_embedding = find_embedding

    def sample(self, bqm, **parameters):
        """Sample the binary quadratic model.
        On the first call of a sampling method, finds a :term:`minor-embedding`
        for the given binary quadratic model (BQM). All subsequent calls to its
        sampling methods reuse this embedding.
        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.
            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between
                variables to create chains. The energy penalty of chain breaks
                is 2 * `chain_strength`.
            chain_break_method (function, optional):
                Method used to resolve chain breaks during sample unembedding.
                See :func:`~dwave.embedding.unembed_sampleset`.
            chain_break_fraction (bool, optional, default=True):
                Add a ‘chain_break_fraction’ field to the unembedded response with
                the fraction of chains broken before unembedding.
            embedding_parameters (dict, optional):
                If provided, parameters are passed to the embedding method as
                keyword arguments. Overrides any `embedding_parameters` passed
                to the constructor. Only used on the first call.
            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.
        Returns:
            :obj:`dimod.SampleSet`
        """
        if self.embedding is None:
            # get an embedding using the current find_embedding function
            embedding_parameters = parameters.pop('embedding_parameters', None)

            if embedding_parameters is None:
                embedding_parameters = self.embedding_parameters
            else:
                # update the base parameters with the new ones provided
                embedding_parameters.update((key, val)
                                            for key, val in self.embedding_parameters
                                            if key not in embedding_parameters)

            source_edgelist = list(itertools.chain(bqm.quadratic,
                                                   ((v, v) for v in bqm.linear)))

            target_edgelist = self.target_structure.edgelist

            embedding = self.find_embedding(source_edgelist, target_edgelist,
                                            **embedding_parameters)

            self._fix_embedding(embedding)

        return super(LazyFixedMinimizeEnergyComposite, self).sample(bqm, **parameters)


class FixedMinimizeEnergyComposite(LazyFixedMinimizeEnergyComposite):
    """Maps problems to a structured sampler with the specified minor-embedding.
    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler such as a D-Wave system.
        embedding (dict[hashable, iterable], optional):
            Mapping from a source graph to the specified sampler’s graph (the
            target graph).
        source_adjacency (dict[hashable, iterable]):
            Deprecated. Dictionary to describe source graph. Ex. `{node:
            {node neighbours}}`.
        kwargs:
            See the :class:`EmbeddingComposite` class for additional keyword
            arguments. Note that `find_embedding` and `embedding_parameters`
            keyword arguments are ignored.
    Examples:
        >>> from dwave.system import DWaveSampler, FixedEmbeddingComposite
        ...
        >>> embedding = {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]}
        >>> sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
        >>> sampler.nodelist
        ['a', 'b', 'c']
        >>> sampler.edgelist
        [('a', 'b'), ('a', 'c'), ('b', 'c')]
        >>> sampleset = sampler.sample_ising({'a': .5, 'c': 0}, {('a', 'c'): -1}, num_reads=500)
        >>> sampleset.first.energy
        -1.5
    """
    def __init__(self, child_sampler, embedding=None, source_adjacency=None,
                 **kwargs):
        super(FixedMinimizeEnergyComposite, self).__init__(child_sampler, **kwargs)

        # dev note: this entire block is to support a deprecated feature and can
        # be removed in the next major release
        if embedding is None:

            warn(("The source_adjacency parameter is deprecated"),
                 DeprecationWarning)

            if source_adjacency is None:
                raise TypeError("either embedding or source_adjacency must be "
                                "provided")

            source_edgelist = [(u, v) for u in source_adjacency for v in source_adjacency[u]]

            embedding = self.find_embedding(source_edgelist,
                                            self.target_structure.edgelist)

        self._fix_embedding(embedding)


class LazyMinimizeEnergyComposite(LazyFixedMinimizeEnergyComposite):
    """Deprecated. Maps problems to the structure of its first given problem.
    This class is deprecated; use the :class:`LazyFixedEmbeddingComposite` class instead.
    Args:
        sampler (dimod.Sampler):
            Structured dimod sampler.
    """
    def __init__(self, child_sampler):
        super(LazyMinimizeEnergyComposite, self).__init__(child_sampler)
        warn("'LazyMinimizeEnergyComposite' has been renamed to 'LazyFixedMinimizeEnergyComposite'.", DeprecationWarning)


class AutoMinimizeEnergyComposite(MinimizeEnergyComposite):
    """Maps problems to a structured sampler, embedding if needed.
    This composite first tries to submit the binary quadratic model directly
    to the child sampler and only embeds if a
    :exc:`dimod.exceptions.BinaryQuadraticModelStructureError` is raised.
    Args:
        sampler (:class:`dimod.Sampler`):
            Structured dimod sampler, such as a
            :obj:`~dwave.system.samplers.DWaveSampler()`.
        find_embedding (function, optional):
            A function `find_embedding(S, T, **kwargs)` where `S` and `T`
            are edgelists. The function can accept additional keyword arguments.
            Defaults to :func:`minorminer.find_embedding`.
        kwargs:
            See the :class:`EmbeddingComposite` class for additional keyword
            arguments.
    """
    def __init__(self, child_sampler, **kwargs):

        child_search = kwargs.get('child_structure_search',
                                  dimod.child_structure_dfs)

        def permissive_child_structure(sampler):
            try:
                return child_search(sampler)
            except ValueError:
                return None
            except (AttributeError, TypeError):  # support legacy dimod
                return None

        super(AutoMinimizeEnergyComposite, self).__init__(child_sampler,
                                                          child_structure_search=permissive_child_structure,
                                                          **kwargs)

    def sample(self, bqm, **parameters):
        child = self.child

        # we want to pass only the parameters relevent to the child sampler
        subparameters = {key: val for key, val in parameters.items()
                         if key in child.parameters}
        try:
            return child.sample(bqm, **subparameters)
        except dimod.exceptions.BinaryQuadraticModelStructureError:
            # does not match the structure so try embedding
            pass

        return super(AutoMinimizeEnergyComposite, self).sample(bqm, **parameters)

def main():
    sampler = MinimizeEnergyComposite(DWaveSampler())
    h = {'a': -1., 'b': 2, 'c': 1, 'd': -3}
    J = {('a', 'b'): -1.5, ('b', 'c'): -1.0, ('b', 'd'): -0.5, ('c', 'd'): -1.0, ('d', 'a'): -3.0}
    sampleset = sampler.sample_ising(h, J, return_embedding=True, chain_break_method=MinimizeEnergy, chain_break_fraction=True, num_reads=100)
    print(sampleset.first.energy)
    print(sampleset)
    print(sampleset.info)

if __name__ == '__main__':
    main()