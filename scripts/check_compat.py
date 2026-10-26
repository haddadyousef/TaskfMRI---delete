"""Check that an atlas (.dlabel.nii) and a scan (.dtseries.nii) describe the SAME grayordinates.

If they differ, averaging by column position would silently use the wrong voxels/vertices.

  python scripts/check_compat.py --dtseries scan.dtseries.nii --dlabel atlas.dlabel.nii
Exit code 0 = identical brain models, 1 = mismatch.
"""
import argparse
import sys

import nibabel as nib
import numpy as np


def compare(scan_img, atlas_img):
    a = scan_img.header.get_axis(scan_img.ndim - 1)
    b = atlas_img.header.get_axis(atlas_img.ndim - 1)
    problems = []
    if len(a) != len(b):
        problems.append(f"grayordinate count: scan={len(a)} atlas={len(b)}")
    sa = {n: (s.start, s.stop if s.stop is not None else len(a)) for n, s, _ in a.iter_structures()}
    sb = {n: (s.start, s.stop if s.stop is not None else len(b)) for n, s, _ in b.iter_structures()}
    if list(sa) != list(sb):
        problems.append(f"structure list/order differs:\n  scan : {list(sa)}\n  atlas: {list(sb)}")
    for n in sa:
        if n in sb and sa[n] != sb[n]:
            problems.append(f"{n}: scan columns {sa[n]} vs atlas columns {sb[n]}")
    if not problems:
        # same layout; now compare the actual vertex / voxel identities
        if not np.array_equal(a.vertex[a.surface_mask], b.vertex[b.surface_mask]):
            problems.append("cortical vertex indices differ")
        if not np.array_equal(a.voxel[a.volume_mask], b.voxel[b.volume_mask]):
            problems.append("subcortical voxel indices differ")
        if not np.allclose(a.affine, b.affine):
            problems.append("volume affine differs")
        if tuple(a.volume_shape) != tuple(b.volume_shape):
            problems.append(f"volume shape: scan={a.volume_shape} atlas={b.volume_shape}")
    return sa, problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtseries", required=True)
    p.add_argument("--dlabel", required=True)
    args = p.parse_args()
    scan, atlas = nib.load(args.dtseries), nib.load(args.dlabel)
    structs, problems = compare(scan, atlas)
    for n, (s, e) in structs.items():
        print(f"{n:45s} {e - s}")
    if problems:
        print("\nMISMATCH:")
        print("\n".join(" - " + x for x in problems))
        sys.exit(1)
    print("\nOK: scan and atlas have identical brain models (structures, vertices, voxels, affine).")


if __name__ == "__main__":
    main()
