import numpy as np
import os
import pandas as pd
import pytest


@pytest.fixture()
def data_folder():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture()
def waypoints(data_folder):
    csv = os.path.join(data_folder, 'waypoints_fit_transform.csv')
    return pd.read_csv(csv, index_col=0)


@pytest.fixture()
def n_samples():
    return 20


@pytest.fixture()
def n_features(n_samples):
    return n_samples


@pytest.fixture
def maybe_bimodal(n_samples, n_features):
    data = np.zeros((n_samples, n_features))
    data[np.tril_indices(n_samples)] = 1
    data = pd.DataFrame(data)
    return data


@pytest.fixture
def maybe_excluded_middle(n_samples, n_features):
    data = np.zeros((n_samples, n_features))
    data[np.tril_indices(n_samples)] = 0.5
    data = pd.DataFrame(data)
    return data


@pytest.fixture
def maybe_included_middle(n_samples, n_features):
    data = np.ones((n_samples, n_features))
    data[np.tril_indices(n_samples)] = 0.5
    data = pd.DataFrame(data)
    return data


@pytest.fixture
def maybe_everything(maybe_bimodal, maybe_excluded_middle,
                     maybe_included_middle):
    return pd.concat([maybe_bimodal, maybe_excluded_middle,
                      maybe_included_middle])
