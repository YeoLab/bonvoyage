import matplotlib.pyplot as plt

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