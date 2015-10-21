import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def switchy_score(array):
    """Transform a 1D array of data scores to a vector of "switchy scores"

    Calculates std deviation and mean of sine- and cosine-transformed
    versions of the array. Better than sorting by just the mean which doesn't
    push the really lowly variant events to the ends.

    Parameters
    ----------
    array : numpy.array
        A 1-D numpy array or something that could be cast as such (like a list)

    Returns
    -------
    switchy_score : float
        The "switchy score" of the study_data which can then be compared to
        other splicing event study_data

    """
    array = np.array(array)
    variance = 1 - np.std(np.sin(array[~np.isnan(array)] * np.pi))
    mean_value = -np.mean(np.cos(array[~np.isnan(array)] * np.pi))
    return variance * mean_value


def get_switchy_score_order(x):
    """Apply switchy scores to a 2D array of data scores

    Parameters
    ----------
    x : numpy.array
        A 2-D numpy array in the shape [n_events, n_samples]

    Returns
    -------
    score_order : numpy.array
        A 1-D array of the ordered indices, in switchy score order
    """
    switchy_scores = np.apply_along_axis(switchy_score, axis=0, arr=x)
    return np.argsort(switchy_scores)


def arrowplot(*args, **kwargs):
    data = kwargs.pop('data')
    voyage_space_positions = kwargs.pop('voyage_space_positions')
    ax = plt.gca()
    phenotype1, phenotype2 = data.transition.values[0].split('-')
    print phenotype1, phenotype2

    # PLot a phantom line for the legend to work
    ax.plot(0, 0, **kwargs)
    for event in data.event_name:
        df = voyage_space_positions.ix[event].ix[[phenotype1, phenotype2]].dropna()
        if df.shape[0] != 2:
            continue
        x1, x2 = df.pc_1.values
        y1, y2 = df.pc_2.values
        dx = x2 - x1
        dy = y2 - y1
        ax.arrow(x1, y1, dx, dy, head_width=0.005, head_length=0.005, #fc='k', ec='k',
                 alpha=0.25, **kwargs)

def hexbin(x, y, *args, **kwargs):
    """Wrapper around hexbin to create a colormap from provided color

    Created for compatibility with seaborn FacetGrid
    """
    ax = kwargs['ax'] if 'ax' in kwargs else plt.gca()
    color = kwargs.pop('color', 'k')
    cmap = sns.light_palette(color, as_cmap=True)

    ax.hexbin(x, y, cmap=cmap, *args, **kwargs)

def voyageplot(nmf_space_positions, feature_id, phenotype_to_color,
               phenotype_to_marker, order, ax=None, xlabel=None, ylabel=None):
    """Plot 2d space traveled by individual splicing events

    Parameters
    ----------
    nmf_space_positions : pandas.DataFrame
        A dataframe with a multiindex of (event, phenotype) and columns of
        x- and y- position, respectively
    feature_id : str
        Unique identifier of the feature to plot
    phenotype_to_color : dict
        Mapping of the phenotype name to a color
    phenotype_to_marker : dict
        Mapping of the phenotype name to a plotting symbol
    order : tuple
        Order in which to plot the phenotypes (e.g. if there is a biological
        ordering)
    ax : matplotlib.Axes object, optional
        An axes to plot these onto. If not provided, grabs current axes
    xlabel : str, optional
        How to label the x-axis
    ylabel : str, optional
        How to label the y-axis
    """
    df = nmf_space_positions.ix[feature_id]

    if ax is None:
        ax = plt.gca()

    for color, s in df.groupby(phenotype_to_color, axis=0):
        phenotype = s.index[0]
        marker = phenotype_to_marker[phenotype]
        ax.plot(s.pc_1, s.pc_2, color=color, marker=marker, markersize=14,
                alpha=0.75, label=phenotype, linestyle='none')

    # ax.scatter(df.ix[:, 0], df.ix[:, 1], color=color, s=100, alpha=0.75)
    # ax.legend(points, df.index.tolist())
    ax.set_xlim(0, nmf_space_positions.ix[:, 0].max() * 1.05)
    ax.set_ylim(0, nmf_space_positions.ix[:, 1].max() * 1.05)

    x = [df.ix[pheno, 0] for pheno in order if pheno in df.index]
    y = [df.ix[pheno, 1] for pheno in order if pheno in df.index]

    ax.plot(x, y, zorder=-1, color='#262626', alpha=0.5, linewidth=1)
    ax.legend()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        ax.set_yticks([])