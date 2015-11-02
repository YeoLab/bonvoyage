# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from .infotheory import binify

class WaypointSpace(object):

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
        """Project the fraction-based data into waypoints space using NMF

        Use non-negative matrix factorization (NMF) to transform fraction-based
        data (ranging from 0 to 1) into a two-dimensional space for
        visualization and interpretation.

        Parameters
        ----------
        data : pandas.DataFrame
            A (samples, features) array of data which are composed of
            fraction-based units (or scaled-down percent based units) which
            range from 0 to 1. Columns whose only value is NA wil be removed.

        Returns
        -------
        waypoints : pandas.DataFrame
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
        data = data.dropna(how='all', axis=1)
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


def voyages(positions, transitions):
    """Find magnitude and direction of waypoints between transitions

    Parameters
    ----------
    positions : pandas.DataFrame
        A ((group, features), 2) multiindexed dataframe with the groups labeled
        in the transitions as the first level on the rows, and the feature ids
        as the second level. Exactly the output from WaypointSpace.transform().
    transitions : list of str pairs
        Which phenotype follows from one to the next, for calculating
        voyages between features
    n : int or float
        If int, then this is the absolute number of cells that are minimum
        required to calculate. If a float, then require this
        fraction of samples to calculate, e.g. if 0.6, then at
        least 60% of samples must have an event detected for voyage space
    Returns
    -------
    voyages : pandas.DataFrame
        A (n_events, n_phenotype_transitions) sized DataFrame of the
        voyages of these events in NMF space
    """
    grouped = positions.groupby(level=0, axis=0)
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
        delta = delta.dropna()
        delta['magnitude'] = np.linalg.norm(delta, axis=1)
        delta = delta.reset_index()
        delta['group1'] = group1
        delta['group2'] = group2
        deltas.append(delta)

    distances = pd.concat(deltas, ignore_index=True)
    distances = distances.rename(columns={0: '$\Delta x$',
                                          1: '$\Delta y$',
                                          'index': 'event_id'})
    distances['direction'] = distances.apply(direction, axis=1)
    distances['transition'] = distances['group1'] + '-' + distances['group2']

    return distances


def direction(row):
    """Assign orientation of change based on delta x and delta y

    Parameters
    ----------
    row : pandas.Series
        A row with items "$\Delta x$" and "$\Delta y$"

    Returns
    -------
    orientation : str
        The direction of change, as a LaTeX arrow, e.g. r'$\nearrow$' (north
        east arrow) for bimodal

    >>> # Towards upper right --> bimodal
    >>> direction(pd.Series({'$\Delta x$': 0.4, '$\Delta y$': 0.1}))
    r'$\nearrow$'
    >>> # Towards lower right --> ~0
    >>> direction(pd.Series({'$\Delta x$': 0.4, '$\Delta y$': -0.1}))
    r'$\searrow'
    >>> # Towards upper left --> ~1
    >>> direction(pd.Series({'$\Delta x$': -0.4, '$\Delta y$': 0.1}))
    r'$\nwarrow$'
    >>> # Towards origin/lower left --> middle
    >>> direction(pd.Series({'$\Delta x$': -0.4, '$\Delta y$': -0.1}))
    r'$\swarrow$'
    >>> # No movement --> Not a number
    >>> direction(pd.Series({'$\Delta x$': 0, '$\Delta y$': 0}))
    np.nan

    """
    dx = row['$\Delta x$']
    dy = row['$\Delta y$']

    if dx == 0 and dy == 0:
        # No movement --> Not a number
        return np.nan
    elif dx > 0 and dy > 0:
        # Towards upper right --> bimodal
        return r'$\nearrow$'
    elif dx > 0 and dy <= 0:
        # Towards lower right --> ~0
        return r'$\searrow$'
    elif dx <= 0 and dy > 0:
        # Towards upper left --> ~1
        return r'$\nwarrow$'
    elif dx <= 0 and dy <= 0:
        # Towards origin/lower left --> middle
        return r'$\swarrow$'
    else:
        return np.nan
