import numpy as np
import matplotlib.pyplot as plt

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