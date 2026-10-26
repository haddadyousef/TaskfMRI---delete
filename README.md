# TaskfMRI

Applying the **GDAR** (Graph Diffusion Autoregressive) model to human **HCP task-fMRI**
(UW NERD Lab). Pipeline: dataset overview -> loader -> structural connectome/graph ->
fit GDAR -> relate flow to task.

## Rules for this repo
- **No HCP data in git.** No `.nii`, `.zip`, `.gii`, or extracted time series. HCP data use
  terms don't allow redistribution, and GitHub rejects files over 100 MB.
  Data stays on the lab remote desktop; code reads the path from `config.local.yaml`.
- **No passwords or keys**, ever. A pushed secret stays in git history.
- The atlas (`atlas/`) is the single source of truth for parcel definitions.
  Don't silently change the 360 Glasser cortical parcels or their ordering.

## Setup
```
pip install -r requirements.txt
cp config.example.yaml config.local.yaml   # then edit paths
```

## Layout
```
atlas/      combined cortical + subcortical atlas, parcel lookup, build script
docs/       decisions.md (atlas choice, open questions), data_notes.md (facts about the data)
src/        cifti_utils.py, extract_parcels.py (+ loader, graph, GDAR as they get added)
scripts/    list_structures.py (CIFTI structure counts), check_atlas.py (atlas QC)
tests/      tests run on a tiny fake CIFTI (make_fake_cifti.py), no HCP data needed
```

## Quick checks
```
python scripts/list_structures.py --zip <subject.zip> --subject 100206 --task MOTOR --phase LR
python tests/make_fake_cifti.py out/fake      # makes fake_dtseries / fake_atlas
pytest
```

## Your extraction job (Yousef)
```
# 1. do the real scan and the atlas describe the same grayordinates? (must say OK)
python scripts/check_compat.py --dtseries scan.dtseries.nii --dlabel Glasser360_Subcortex14.dlabel.nii
# 2. 374 parcel time series for one scan, with sanity checks + cortex baseline check
python scripts/extract_subject.py --dtseries scan.dtseries.nii \
    --dlabel Glasser360_Subcortex14.dlabel.nii --lookup parcel_lookup.csv \
    --glasser-cortex Glasser360_working.dlabel.nii --out outputs/100206_MOTOR_LR
```
Outputs go to `outputs/` (gitignored). Atlas files come from Aya (see `atlas/README.md`).

## Status
See `docs/decisions.md` for the atlas decision and open questions.
