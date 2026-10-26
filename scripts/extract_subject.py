"""Extract parcel-level BOLD time series for ONE scan using the 374-parcel atlas.

  python scripts/extract_subject.py --dtseries scan.dtseries.nii \
      --dlabel Glasser360_Subcortex14.dlabel.nii --lookup parcel_lookup.csv \
      --out outputs/100206_MOTOR_LR [--glasser-cortex Glasser360_working.dlabel.nii]

Steps: (1) refuse to run if scan and atlas brain models differ, (2) mean of each parcel's
grayordinates at every time point, (3) sanity checks, (4) save <out>.npz and <out>.csv.
Optional baseline check: the 360 cortical parcels must equal a cortex-only Glasser
parcellation of the same scan.
"""
import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.check_compat import compare  # noqa: E402
from src.cifti_utils import load_labels  # noqa: E402
from src.extract_parcels import parcellate_arrays  # noqa: E402


def read_lookup(path):
    with open(path, newline="") as f:
        return {int(r["ParcelKey"] if "ParcelKey" in r else r["parcel_id"]):
                (r.get("ParcelName") or r.get("name")) for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtseries", required=True)
    p.add_argument("--dlabel", required=True)
    p.add_argument("--lookup", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--glasser-cortex", help="cortex-only Glasser dlabel for the baseline check")
    p.add_argument("--n-cortical", type=int, default=360)
    a = p.parse_args()

    scan, atlas = nib.load(a.dtseries), nib.load(a.dlabel)
    _, problems = compare(scan, atlas)
    if problems:
        sys.exit("Atlas and scan do not match, not extracting:\n - " + "\n - ".join(problems))

    data = np.asarray(scan.get_fdata())                    # (time, grayordinates)
    labels, _ = load_labels(a.dlabel)
    ids, ts = parcellate_arrays(data, labels)
    names = read_lookup(a.lookup)
    print(f"output shape (time, parcels): {ts.shape}")

    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + msg); ok = ok and cond
    check(not np.isnan(ts).any(), "no NaNs")
    flat = [names.get(int(i), "?") for i in ids[(ts == 0).all(axis=0)]]
    check(not flat, f"no all-zero parcels {flat}")
    missing = [i for i in names if i not in set(ids.tolist())]
    check(not missing, f"every lookup parcel got a signal (missing: {missing})")

    if a.glasser_cortex:
        gl, _ = load_labels(a.glasser_cortex)
        bm = scan.header.get_axis(1)
        n_ctx = int(bm.surface_mask.sum())
        check(gl.size == n_ctx and bm.surface_mask[:n_ctx].all(),
              "cortex columns come first and match the Glasser-only file size")
        gid, gts = parcellate_arrays(data[:, :n_ctx], gl)
        keep = np.isin(ids, gid)
        same = np.array_equal(ids[keep], gid) and np.allclose(ts[:, keep], gts, atol=1e-6)
        check(same, f"the {a.n_cortical} cortical parcels equal the Glasser-only baseline")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out + ".npz", ts=ts.astype(np.float32), parcel_ids=ids,
                        parcel_names=np.array([names.get(int(i), "?") for i in ids]))
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([names.get(int(i), str(i)) for i in ids])
        w.writerows(ts)
    print("saved", a.out + ".npz /.csv")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
