import numpy as np
import pandas as pd


VOYAGE_COLUMNS = ['group1', 'group2', 'magnitude', '$\Delta x$',
                  '$\Delta y$ ', 'direction']


class Voyages(object):

    def voyages(self, waypoints, transitions):
        """Find magnitude and direction of waypoints between transitions

        Parameters
        ----------
        waypoints : pandas.DataFrame
            A ((group, features), 2) multiindexed dataframe with the groups labeled
            in the transitions as the first level on the rows, and the feature ids
            as the second level. Exactly the output from Waypoints.transform().
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
        grouped = waypoints.groupby(level=0, axis=0)
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
        distances['direction'] = distances.apply(self.direction, axis=1)
        distances['transition'] = distances['group1'] + '-' + distances['group2']

        return distances

    @staticmethod
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
