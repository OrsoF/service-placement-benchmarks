"""Archived experiments share one pickle implementation after the cleanup."""

import pytest

from service_placement.compat.pickle_io import pickle_load, pickle_save


def test_pickle_helpers_round_trip(tmp_path):
    path = tmp_path / "instance.pkl"
    payload = {"name": "legacy", "quantities": [1, 2]}
    pickle_save(payload, path)
    assert pickle_load(path) == payload


@pytest.mark.requires("pandas")
def test_experiment_pickle_exports_use_the_shared_implementation():
    from service_placement.compat import experiment_io

    assert experiment_io.pickle_save is pickle_save
    assert experiment_io.pickle_load is pickle_load
