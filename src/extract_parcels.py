"""Average grayordinate time series into parcel time series using a dlabel atlas."""
import nibabel as nib
import numpy as np

from .cifti_utils import load_labels


def parcellate_arrays(data, labels):
    """Core mean-per-parcel step. data: (time, grayordinates), labels: (grayordinates,).
    Returns (parcel_ids, ts) with ts shape (time, n_parcels). Label 0 is skipped."""
    if data.shape[1] != labels.size:
        raise ValueError(
            f"scan has {data.shape[1]} grayordinates, atlas has {labels.size}")
    ids = np.array(sorted(set(labels.tolist()) - {0}))
    ts = np.empty((data.shape[0], ids.size))
    for j, pid in enumerate(ids):
        ts[:, j] = data[:, labels == pid].mean(axis=1)
    return ids, ts


def parcellate(dtseries_path, dlabel_path):
    """Return (parcel_ids, ts) where ts has shape (n_timepoints, n_parcels).

    Each parcel's signal is the plain MEAN over its grayordinates (same as
    wb_command -cifti-parcellate default). Label 0 (unlabeled) is skipped.
    Raises if the atlas and scan don't have the same number of grayordinates:
    that means the column order can't be trusted.
    """
    img = nib.load(dtseries_path)
    data = np.asarray(img.get_fdata())          # (time, grayordinates)
    labels, _ = load_labels(dlabel_path)
    return parcellate_arrays(data, labels)
