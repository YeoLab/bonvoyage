import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(params=['hexbin', 'scatter'])
def kind(request):
    return request.param


def test_waypointplot(waypoints, kind):
    from bonvoyage import waypointplot

    fig, ax = plt.subplots()
    waypointplot(waypoints, kind, ax=ax)

    if kind == 'hexbin':
        assert isinstance(ax.collections[0], mpl.collections.PolyCollection)
    if kind == 'scatter':
        assert isinstance(ax.collections[0], mpl.collections.PathCollection)
