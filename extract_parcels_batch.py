"""Batch version: extract 374 parcel BOLD time series for many subjects / tasks / runs.

Reads each subject's task zip (read-only), pulls out one dtseries at a time to a temp folder,
checks it against the atlas, averages the parcels, saves a small .npz, deletes the temp copy.
Safe to stop and re-run: runs that already have an output file are skipped.

Example (PowerShell, from task-fMRI):
  python extract_parcels_batch.py --zip-dir "R:\\Yazdan Lab Members\\Grad\\fschwock\\HCP - task fMRI" ^
      --atlas-dir ".\\Glasser374_Atlas" --out-dir ".\\outputs" --limit 2

Outputs:  <out-dir>\\<subject>\\<subject>_<TASK>_<PHASE>.npz   (ts, parcel_ids, parcel_names)
Log:      <out-dir>\\batch_log.csv  (one row per run: ok / skipped / failed + reason)
"""
import argparse, csv, glob, os, shutil, sys, tempfile, time, zipfile
import nibabel as nib
import numpy as np

TASKS = ["EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]
PHASES = ["LR", "RL"]


def structures(axis):
    return {n: (s.start, s.stop if s.stop is not None else len(axis)) for n, s, _ in axis.iter_structures()}


def compat_problems(a, b):
    """a, b = brain model axes (scan, atlas). Empty list = identical."""
    p = []
    if len(a) != len(b):
        return [f"grayordinate count: scan={len(a)} atlas={len(b)}"]
    sa, sb = structures(a), structures(b)
    if list(sa) != list(sb) or any(sa[n] != sb[n] for n in sa):
        p.append("structure list/order/counts differ")
    if not p:
        if not np.array_equal(a.vertex[a.surface_mask], b.vertex[b.surface_mask]):
            p.append("cortical vertex indices differ")
        if not np.array_equal(a.voxel[a.volume_mask], b.voxel[b.volume_mask]):
            p.append("subcortical voxel indices differ")
        if not np.allclose(a.affine, b.affine):
            p.append("volume affine differs")
    return p


class Atlas:
    def __init__(self, d):
        dl = nib.load(os.path.join(d, "Glasser360_Subcortex14.dlabel.nii"))
        self.axis = dl.header.get_axis(1)
        labels = np.asarray(dl.get_fdata()).reshape(-1).astype(int)
        keep = np.where(labels > 0)[0]
        order = keep[np.argsort(labels[keep], kind="stable")]
        sl = labels[order]
        self.ids, starts, counts = np.unique(sl, return_index=True, return_counts=True)
        self.order, self.starts, self.counts = order, starts, counts
        with open(os.path.join(d, "parcel_lookup.csv"), newline="") as f:
            lk = {int(r["ParcelKey"]): r["ParcelName"] for r in csv.DictReader(f)}
        self.names = np.array([lk.get(int(i), "?") for i in self.ids])
        self.n_expected = len(lk)

    def parcellate(self, data):
        """data (time, grayordinates) -> (time, parcels): plain mean over each parcel's grayordinates."""
        sums = np.add.reduceat(data[:, self.order], self.starts, axis=1, dtype=np.float64)
        return sums / self.counts


def process(scan_path, atlas, out_path):
    scan = nib.load(scan_path)
    problems = compat_problems(scan.header.get_axis(1), atlas.axis)
    if problems:
        raise RuntimeError("atlas/scan mismatch: " + "; ".join(problems))
    ts = atlas.parcellate(scan.get_fdata(dtype=np.float32))
    if ts.shape[1] != atlas.n_expected:
        raise RuntimeError(f"got {ts.shape[1]} parcels, expected {atlas.n_expected}")
    if np.isnan(ts).any():
        raise RuntimeError("NaNs in output")
    zero = atlas.names[(ts == 0).all(axis=0)]
    if len(zero):
        raise RuntimeError(f"all-zero parcels: {list(zero)}")
    np.savez_compressed(out_path, ts=ts.astype(np.float32), parcel_ids=atlas.ids, parcel_names=atlas.names)
    return ts.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", required=True, help="folder holding the per-subject zips")
    ap.add_argument("--zip-pattern", default="*_Task3TRecommended.zip")
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--phases", nargs="+", default=PHASES)
    ap.add_argument("--subjects", nargs="+", help="only these subject IDs")
    ap.add_argument("--limit", type=int, help="only the first N subjects (for testing)")
    a = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(a.zip_dir, a.zip_pattern)))
    if a.subjects:
        zips = [z for z in zips if os.path.basename(z).split("_")[0] in set(a.subjects)]
    if a.limit:
        zips = zips[: a.limit]
    if not zips:
        sys.exit(f"No zips matching {a.zip_pattern} in {a.zip_dir}")
    print(f"{len(zips)} subject zip(s), tasks={a.tasks}, phases={a.phases}")

    atlas = Atlas(a.atlas_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    log_path = os.path.join(a.out_dir, "batch_log.csv")
    new_log = not os.path.exists(log_path)
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    with open(log_path, "a", newline="") as lf:
        log = csv.writer(lf)
        if new_log:
            log.writerow(["time", "subject", "task", "phase", "status", "detail"])
        for zi, zpath in enumerate(zips, 1):
            subj = os.path.basename(zpath).split("_")[0]
            sdir = os.path.join(a.out_dir, subj)
            os.makedirs(sdir, exist_ok=True)
            try:
                z = zipfile.ZipFile(zpath)
                members = [n for n in z.namelist() if n.endswith(".dtseries.nii")]
            except Exception as e:
                print(f"[{zi}/{len(zips)}] {subj}: cannot open zip: {e}")
                log.writerow([time.strftime("%F %T"), subj, "", "", "failed", f"zip: {e}"]); lf.flush()
                counts["failed"] += 1
                continue
            with z:
                for task in a.tasks:
                    for ph in a.phases:
                        out = os.path.join(sdir, f"{subj}_{task}_{ph}.npz")
                        tag = f"[{zi}/{len(zips)}] {subj} {task}_{ph}"
                        if os.path.exists(out):
                            counts["skipped"] += 1
                            continue
                        hit = [n for n in members if f"/tfMRI_{task}_{ph}/" in n]
                        if not hit:
                            msg = "no dtseries in zip"
                            print(tag, "FAILED:", msg); counts["failed"] += 1
                            log.writerow([time.strftime("%F %T"), subj, task, ph, "failed", msg]); lf.flush()
                            continue
                        tmp = tempfile.mkdtemp(prefix="hcp_extract_")
                        try:
                            t0 = time.time()
                            path = z.extract(hit[0], tmp)
                            shape = process(path, atlas, out)
                            print(f"{tag} ok {shape} {time.time() - t0:.1f}s")
                            counts["ok"] += 1
                            log.writerow([time.strftime("%F %T"), subj, task, ph, "ok", str(shape)])
                        except Exception as e:
                            print(tag, "FAILED:", e); counts["failed"] += 1
                            log.writerow([time.strftime("%F %T"), subj, task, ph, "failed", str(e)])
                            if os.path.exists(out):
                                os.remove(out)
                        finally:
                            shutil.rmtree(tmp, ignore_errors=True)
                            lf.flush()
    print(f"\nDone. ok={counts['ok']} skipped(already done)={counts['skipped']} failed={counts['failed']}")
    print("Log:", log_path)


if __name__ == "__main__":
    main()
