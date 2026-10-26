import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from make_fake_cifti import make  # noqa: E402
from src.cifti_utils import load_labels, structure_counts  # noqa: E402
from src.extract_parcels import parcellate  # noqa: E402
import nibabel as nib  # noqa: E402


@pytest.fixture
def fake(tmp_path):
    return make(str(tmp_path / "f"))


def test_structure_counts(fake):
    dt, _, _, _ = fake
    counts = dict(structure_counts(nib.load(dt)))
    assert counts["CIFTI_STRUCTURE_CORTEX_LEFT"] == 20
    assert counts["CIFTI_STRUCTURE_THALAMUS_LEFT"] == 8
    assert sum(counts.values()) == 56


def test_parcellate_is_mean(fake):
    dt, dl, data, labels = fake
    ids, ts = parcellate(dt, dl)
    assert list(ids) == [1, 2, 3, 4, 5, 6]
    assert ts.shape == (30, 6)
    np.testing.assert_allclose(ts[:, 4], data[:, labels == 5].mean(axis=1), rtol=1e-5)


def test_size_mismatch_raises(fake, tmp_path):
    dt, _, _, _ = fake
    _, dl2, _, _ = make(str(tmp_path / "g"), n_vert=25)
    with pytest.raises(ValueError):
        parcellate(dt, dl2)


def test_compat_match_and_mismatch(fake, tmp_path):
    from scripts.check_compat import compare
    dt, dl, _, _ = fake
    assert compare(nib.load(dt), nib.load(dl))[1] == []
    _, dl2, _, _ = make(str(tmp_path / "h"), n_vert=25)
    assert compare(nib.load(dt), nib.load(dl2))[1] != []
