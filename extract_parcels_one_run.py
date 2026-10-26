"""Extract 374 parcel BOLD time series for ONE HCP task run, using Aya's atlas.

Usage (PowerShell, one line):
  python extract_parcels_one_run.py --zip "R:\\path\\to\\100206_..._MOTOR_preproc.zip" --task MOTOR --phase LR ^
      --atlas-dir "C:\\Users\\yhaddad\\Glasser374_Atlas" --out "C:\\Users\\yhaddad\\outputs\\100206_MOTOR_LR"
Or, if the dtseries is already extracted:  --dtseries "path\\to\\file.dtseries.nii"  (instead of --zip/--task/--phase)

What it does: (1) pulls the dtseries out of the zip into a temp folder, (2) checks the scan and
the atlas describe the same grayordinates (stops if not), (3) averages each parcel's grayordinates
at every time point, (4) runs sanity checks incl. the Glasser-only cortex baseline, (5) saves
<out>.npz and <out>.csv, (6) deletes the temp copy.
"""
import argparse, csv, os, shutil, sys, tempfile, zipfile
import nibabel as nib
import numpy as np


def load_labels(path):
    img = nib.load(path)
    return np.asarray(img.get_fdata()).reshape(-1).astype(int)


def parcel_means(data, labels):
    ids = np.array(sorted(set(labels.tolist()) - {0}))
    ts = np.empty((data.shape[0], ids.size))
    for j, pid in enumerate(ids):
        ts[:, j] = data[:, labels == pid].mean(axis=1)
    return ids, ts


def structures(axis):
    out = {}
    for n, s, _ in axis.iter_structures():
        out[n] = (s.start, s.stop if s.stop is not None else len(axis))
    return out


def compat_problems(scan, atlas):
    a, b = scan.header.get_axis(1), atlas.header.get_axis(1)
    p = []
    if len(a) != len(b):
        p.append(f"grayordinate count: scan={len(a)} atlas={len(b)}")
    sa, sb = structures(a), structures(b)
    if list(sa) != list(sb):
        p.append(f"structure list/order differs\n scan : {list(sa)}\n atlas: {list(sb)}")
    for n in sa:
        if n in sb and sa[n] != sb[n]:
            p.append(f"{n}: scan columns {sa[n]} vs atlas columns {sb[n]}")
    if not p:
        if not np.array_equal(a.vertex[a.surface_mask], b.vertex[b.surface_mask]):
            p.append("cortical vertex indices differ")
        if not np.array_equal(a.voxel[a.volume_mask], b.voxel[b.volume_mask]):
            p.append("subcortical voxel indices differ")
        if not np.allclose(a.affine, b.affine):
            p.append("volume affine differs")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip"); ap.add_argument("--task"); ap.add_argument("--phase", choices=["LR", "RL"])
    ap.add_argument("--dtseries")
    ap.add_argument("--atlas-dir", required=True, help="folder with Glasser360_Subcortex14.dlabel.nii etc.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tmp = None
    if a.dtseries:
        scan_path = a.dtseries
    else:
        if not (a.zip and a.task and a.phase):
            sys.exit("Give --dtseries, or --zip with --task and --phase")
        run = f"tfMRI_{a.task}_{a.phase}"
        with zipfile.ZipFile(a.zip) as z:
            names = [n for n in z.namelist() if n.endswith(".dtseries.nii") and f"/{run}/" in n]
            if not names:
                print("No match. dtseries files in zip:")
                print("\n".join(n for n in z.namelist() if n.endswith(".dtseries.nii")))
                sys.exit(1)
            if len(names) > 1:
                print("Several matches, using the first:", *names, sep="\n  ")
            tmp = tempfile.mkdtemp(prefix="hcp_extract_")
            scan_path = z.extract(names[0], tmp)
        print("extracted:", names[0])

    try:
        d = a.atlas_dir
        dlabel = os.path.join(d, "Glasser360_Subcortex14.dlabel.nii")
        lookup = os.path.join(d, "parcel_lookup.csv")
        working = os.path.join(d, "Glasser360_working.dlabel.nii")
        scan, atlas = nib.load(scan_path), nib.load(dlabel)
        print("scan shape (time, grayordinates):", scan.shape)

        problems = compat_problems(scan, atlas)
        if problems:
            sys.exit("STOP: scan and atlas do not match, so nothing was extracted:\n - " + "\n - ".join(problems))
        print("PASS  scan and atlas have identical brain models")

        data = np.asarray(scan.get_fdata())
        labels = load_labels(dlabel)
        ids, ts = parcel_means(data, labels)
        with open(lookup, newline="") as f:
            names_by_id = {int(r["ParcelKey"]): r["ParcelName"] for r in csv.DictReader(f)}
        print("output shape (time, parcels):", ts.shape)

        ok = True
        def check(cond, msg):
            nonlocal ok
            print(("PASS  " if cond else "FAIL  ") + msg); ok = ok and cond
        check(ts.shape[1] == 374, "374 parcels")
        check(not np.isnan(ts).any(), "no NaNs")
        zero = [names_by_id.get(int(i), "?") for i in ids[(ts == 0).all(axis=0)]]
        check(not zero, f"no all-zero parcels {zero}")
        check(set(names_by_id) == set(ids.tolist()), "parcel ids match parcel_lookup.csv")
        if os.path.exists(working):
            gl = load_labels(working)
            n_ctx = int(scan.header.get_axis(1).surface_mask.sum())
            gid, gts = parcel_means(data[:, :n_ctx], gl)
            keep = np.isin(ids, gid)
            check(gl.size == n_ctx and np.array_equal(ids[keep], gid) and np.allclose(ts[:, keep], gts, atol=1e-6),
                  "the 360 cortical parcels equal the Glasser-only baseline")

        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez_compressed(a.out + ".npz", ts=ts.astype(np.float32), parcel_ids=ids,
                            parcel_names=np.array([names_by_id.get(int(i), "?") for i in ids]))
        with open(a.out + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([names_by_id.get(int(i), str(i)) for i in ids])
            w.writerows(ts)
        print("saved", a.out + ".npz and .csv")
        sys.exit(0 if ok else 1)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
