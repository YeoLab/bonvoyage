# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from modish import ModalityEstimator

def bin_range_strings(bins):
    """Given a list of bins, make a list of strings of those bin ranges

    Parameters
    ----------
    bins : list_like
        List of anything, usually values of bin edges

    Returns
    -------
    bin_ranges : list
        List of bin ranges

    >>> bin_range_strings((0, 0.5, 1))
    ['0-0.5', '0.5-1']
    """
    return ['{}-{}'.format(i, j) for i, j in zip(bins, bins[1:])]


def binify(data, bins):
    """Makes a histogram of each column the provided binsize

    Parameters
    ----------
    data : pandas.DataFrame
        A samples x features dataframe. Each feature (column) will be binned
        into the provided bins
    bins : iterable
        Bins you would like to use for this data. Must include the final bin
        value, e.g. (0, 0.5, 1) for the two bins (0, 0.5) and (0.5, 1).
        nbins = len(bins) - 1

    Returns
    -------
    binned : pandas.DataFrame
        An nbins x features DataFrame of each column binned across rows
    """
    if bins is None:
        raise ValueError('"bins" cannot be None')
    binned = data.apply(lambda x: pd.Series(np.histogram(x, bins=bins)[0]))
    binned.index = bin_range_strings(bins)

    # Normalize so each column sums to 1
    binned = binned / binned.sum().astype(float)
    return binned


class VoyageSpace(object):

    n_components = 2
    binsize = 0.1
    bins = np.arange(0, 1+binsize, binsize)

    near1_label = '~1'
    near0_label = '~0'

    xlabel = near0_label
    ylabel = near1_label

    near0_binned = [1] + [0] * 9
    near1_binned = near0_binned[::-1]
    seed_data = pd.DataFrame(
        # Use two near0_binned to ensure the x-axis is near-0, i.e. that 0th
        # axis is near-0
        [near0_binned, near0_binned, near1_binned])
    
    def __init__(self):
        self.nmf = NMF(n_components=self.n_components, init='nndsvdar')
        self.nmf.fit(self.seed_data)

    def transform(self, data):
        """Project the fraction-based data into voyage space using NMF

        Use non-negative matrix factorization (NMF) to transform fraction-based
        data (ranging from 0 to 1) into a two-dimensional space for
        visualization and interpretation.

        Parameters
        ----------
        data : pandas.DataFrame
            A (samples, features) array of data which are composed of
            fraction-based units (or scaled-down percent based units) which
            range from 0 to 1

        Returns
        -------
        transformed : pandas.DataFrame
            A (features, 2) array of the data transformed by NMF, and scaled
            such that the maximum x- and y-values are 1.

        Raises
        ------
        ValueError
            If the data contains any values that are greater than 1 or less
            than 0.
        """
        if (data > 1).any().any():
            raise ValueError("Some of the data is greater than 1 - only values"
                             " between 0 and 1 are accepted")
        if (data < 0).any().any():
            raise ValueError("Some of the data is less than 0 - only values"
                             " between 0 and 1 are accepted")

        binned = self.binify(data).T
        transformed = self.nmf.transform(binned)
        transformed = pd.DataFrame(transformed, index=data.columns)

        # Normalize data so maximum for x and y axis is always 1. Since
        # transformed data is non-negative, don't need to subtract the minimum,
        # since the minimum >= 0.
        transformed = transformed/transformed.max()

        return transformed

    def binify(self, data):
        return binify(data, self.bins)


DISTANCE_COLUMNS = ['group1', 'group2', 'voyage_distance', 'delta_x',
                    'delta_y', 'direction']


def single_voyage_distance(positions, transition):
    """Get NMF distance of a single feature between phenotype transitions

    Parameters
    ----------
    positions : pandas.DataFrame
        A ((n_features, phenotypes), 2) MultiIndex dataframe of the NMF
        positions of splicing events for different phenotypes
    transitions : list of 2-string tuples
        List of (group1, group2) transitions

    Returns
    -------
    transitions : pandas.DataFrame
        A (n_features, n_transitions) DataFrame of the NMF distances
        of features between different phenotypes
    """
    # positions_phenotype = positions.copy()
    # positions_phenotype.index = positions_phenotype.index.droplevel(1)
    # distances = pd.Series(index=transitions)
    lines = []
    # for transition in transitions:
    try:
        group1, group2 = transition
        delta = positions.loc[group2] - positions.loc[group1]
        delta_x, delta_y = delta.values[0]
        norm = np.linalg.norm([delta_x, delta_y])
        line = [group1, group2, norm, delta_x, delta_y]
        lines.append(line)
    except KeyError:
        pass
    distances = pd.Series(lines, index=DISTANCE_COLUMNS)
    return distances


def voyage_distances(voyage_positions, transitions):
    """Get distance in NMF space of different splicing events

    Parameters
    ----------
    voyage_positions : pandas.DataFrame
        A ((group, features), 2) multiindexed dataframe with the groups labeled
        in the transitions as the first level on the rows, and the feature ids
        as the second level. Exactly the output from a
    transitions : list of str pairs
        Which phenotype follows from one to the next, for calculating
        distances between features
    n : int or float
        If int, then this is the absolute number of cells that are minimum
        required to calculate. If a float, then require this
        fraction of samples to calculate, e.g. if 0.6, then at
        least 60% of samples must have an event detected for voyage space
    Returns
    -------
    nmf_space_transitions : pandas.DataFrame
        A (n_events, n_phenotype_transitions) sized DataFrame of the
        distances of these events in NMF space
    """

    # distances = voyage_positions.groupby(
    #     level=1, axis=0, as_index=True, group_keys=False).transform(
    #     single_voyage_distance, transitions=transitions)

    grouped = voyage_positions.groupby(level=0, axis=0)
    groups = {}
    for group in grouped.groups:
        df = grouped.get_group(group)
        df.index = df.index.droplevel(0)
        groups[group] = df

    deltas = []
    for group1, group2 in transitions:
        df1 = groups[group1]
        df2 = groups[group2]
        delta = df2 - df1
        # print delta.head()
        delta['voyage_distance'] = np.linalg.norm(delta, axis=1)
        delta = delta.reset_index()
        delta['group1'] = group1
        delta['group2'] = group2
        deltas.append(delta)

    distances = pd.concat(deltas, ignore_index=True)
    distances = distances.rename(columns={0: 'delta_x', 1: 'delta_y',
                                          'index': 'event_id'})
    distances['direction'] = distances.apply(voyage_direction, axis=1)

    return distances


def voyage_direction(row):
    dx = row['delta_x']
    dy = row['delta_y']
    if dx > 0 and dy > 0:
        # Towards upper right --> bimodal
        return r'$\nearrow$'
    elif dx > 0 and dy <= 0:
        # Towards lower right --> ~0
        return r'$\searrow$'
    elif dx <= 0 and dy > 0:
        # Towards upper left --> ~1
        return r'$\nwarrow$'
    elif dx < 0 and dy < 0:
        # Towards origin/lower left --> middle
        return r'$\swarrow$'
    else:
        return np.nan
