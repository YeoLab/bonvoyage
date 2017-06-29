import os
import pandas as pd
import pandas.util.testing as pdt
import pytest
from sklearn.decomposition import NMF


class TestWaypoints(object):

    @pytest.fixture
    def waypoints(self):
        from bonvoyage import Waypoints
        wp = Waypoints()
        return wp

    def test___init__(self, waypoints, data_folder):
        assert isinstance(waypoints.nmf, NMF)

        test = waypoints.seed_data_transformed
        test.index = test.index.to_native_types()

        csv = os.path.join(data_folder, 'seed_data_transformed.csv')
        true = pd.read_csv(csv)
        pdt.assert_almost_equal(test.values, true.values)

    def test_fit(self, waypoints, maybe_everything, data_folder):
        test = waypoints.fit(maybe_everything)

        csv = os.path.join(data_folder, 'waypoints_fit.csv')
        true = pd.read_csv(csv, index_col=0)
        pdt.assert_frame_equal(test, true)

    def test_fit_transform(self, waypoints, maybe_everything, data_folder):
        test = waypoints.fit_transform(maybe_everything)

        csv = os.path.join(data_folder, 'waypoints_fit_transform.csv')
        true = pd.read_csv(csv, index_col=0)
        pdt.assert_almost_equal(test.values, true.values)

    def test_binify(self, waypoints, maybe_everything, data_folder):
        test = waypoints.binify(maybe_everything)

        csv = os.path.join(data_folder, 'waypoints_binify.csv')
        true = pd.read_csv(csv, index_col=0)
        true.columns = pd.RangeIndex(start=0, stop=20, step=1)
        pdt.assert_frame_equal(test, true)
