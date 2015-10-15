# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

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
    binned = df.apply(lambda x: pd.Series(np.histogram(x, bins=bins)[0]))
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

    seed_data = None

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

    def binned_nmf_reduced(self, sample_ids=None, feature_ids=None,
                           data=None):
        if data is None:
            data = self._subset(self.data, sample_ids, feature_ids,
                                require_min_samples=False)
        binned = self.binify(data)
        reduced = self.nmf.transform(binned.T)
        return reduced

    def _is_nmf_space_x_axis_excluded(self, groupby):
        nmf_space_positions = self.nmf_space_positions(groupby)
    
        # Get the correct included/excluded labeling for the x and y axes
        event, phenotype = nmf_space_positions.pc_1.argmax()
        top_pc1_samples = self.data.groupby(groupby).groups[
            phenotype]
    
        data = self._subset(self.data, sample_ids=top_pc1_samples)
        binned = self.binify(data)
        return bool(binned[event][0])
    
    
    def _nmf_space_xlabel(self, groupby):
        if self._is_nmf_space_x_axis_excluded(groupby):
            return self.near0_label
        else:
            return self.near1_label
    
    
    def _nmf_space_ylabel(self, groupby):
        if self._is_nmf_space_x_axis_excluded(groupby):
            return self.near1_label
        else:
            return self.near0_label