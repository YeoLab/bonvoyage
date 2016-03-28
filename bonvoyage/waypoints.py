# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from anchor.infotheory import binify

class Waypoints(object):

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
        self.seed_data_transformed = pd.DataFrame(
            self.nmf.fit_transform(self.seed_data))

    def fit(self, data):
        """Discretize the data in preparation for transformation to waypoints

        Parameters
        ----------
        data : pandas.DataFrame
            A (samples, features) array of data which are composed of
            fraction-based units (or scaled-down percent based units) which
            range from 0 to 1. Columns whose only value is NA wil be removed.

        Returns
        -------
        binned : pandas.DataFrame
            A (features, 10) array of the discretized data

        Raises
        ------
        ValueError
            If the data contains any values that are greater than 1 or less
            than 0.
        """
        if isinstance(data, pd.DataFrame):
            if (data > 1).any().any():
                raise ValueError(
                    "Some of the data is greater than 1 - only values"
                    " between 0 and 1 are accepted")
            if (data < 0).any().any():
                raise ValueError(
                    "Some of the data is less than 0 - only values"
                    " between 0 and 1 are accepted")

            data = data.dropna(how='all', axis=1)
            binned = self.binify(data).T
        elif isinstance(data, pd.Series):
            if (data > 1).any():
                raise ValueError(
                    "Some of the data is greater than 1 - only values"
                    " between 0 and 1 are accepted")
            if (data < 0).any():
                raise ValueError(
                    "Some of the data is less than 0 - only values"
                    " between 0 and 1 are accepted")

            binned = self.binify(data)
        else:
            raise ValueError('Only pandas DataFrames and Series are accepted')

        return binned

    def transform(self, binned):
        """Project the fraction-based data into waypoints space using NMF

        Use non-negative matrix factorization (NMF) to transform fraction-based
        data (ranging from 0 to 1) into a two-dimensional space for
        visualization and interpretation.

        Parameters
        ----------
        binned : pandas.DataFrame | pandas.Series
            A (features, 10) array of data, or (10,) series binned into bins
            of size 0.1: (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

        Returns
        -------
        waypoints : pandas.DataFrame
            A (features, 2) array of the binned data transformed by NMF, and
            scaled such that the maximum x- and y-values are 1.

        """
        if isinstance(binned, pd.DataFrame):
            transformed = self.nmf.transform(binned)
            transformed = pd.DataFrame(transformed, index=binned.index)

        elif isinstance(binned, pd.Series):
            transformed = self.nmf.transform(binned.to_frame().T)[0]
            transformed = pd.Series(transformed, name=binned.name)
        else:
            raise ValueError('Only pandas DataFrames and Series are accepted')

        # Normalize data so maximum for x and y axis is always 1. Since
        # transformed data is non-negative, don't need to subtract the minimum,
        # since the minimum >= 0.
        transformed = transformed / self.seed_data_transformed.max()

        return transformed

    def fit_transform(self, data):
        """A one-step to discretize data and transform to waypoints

        Parameters
        ----------
        data : pandas.DataFrame
            A (samples, features) array of data which are composed of
            fraction-based units (or scaled-down percent based units) which
            range from 0 to 1. Columns whose only value is NA wil be removed.

        Returns
        -------
        waypoints : pandas.DataFrame
            A (features, 2) array of the binned data transformed by NMF, and
            scaled such that the maximum x- and y-values are 1.

        Raises
        ------
        ValueError
            If the data contains any values that are greater than 1 or less
            than 0.
        """
        return self.transform(self.fit(data))

    def binify(self, data):
        return binify(data, self.bins)


