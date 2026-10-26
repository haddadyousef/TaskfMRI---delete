# TaskfMRI

Extract parcel-level BOLD time series from HCP task-fMRI CIFTI files using the combined
**Glasser 360 cortical + 14 subcortical (374 parcel)** atlas.

For each run, every parcel's signal is the plain mean of its grayordinates at each time point.
Nothing is averaged across runs or subjects. Each run gets its own output (time x 374).

## Needs
- Python with `pip install -r requirements.txt`
- HCP task zips (one per subject, e.g. `100206_Task3TRecommended.zip`), read-only
- The atlas folder from Aya, placed next to the scripts as `Glasser374_Atlas/`
  (needs `Glasser360_Subcortex14.dlabel.nii`, `parcel_lookup.csv`; `Glasser360_working.dlabel.nii`
  is used for an optional cortex check). Not stored in this repo.

## One run
```
python extract_parcels_one_run.py --zip "<subject zip>" --task MOTOR --phase LR ^
    --atlas-dir ".\Glasser374_Atlas" --out ".\outputs\100206_MOTOR_LR"
```
Stops if the scan and atlas don't describe the same grayordinates, then checks for NaNs,
all-zero parcels and the Glasser-only cortex baseline. Saves `.npz` and `.csv`.

## Many subjects / tasks
```
python extract_parcels_batch.py --zip-dir "<folder of zips>" --atlas-dir ".\Glasser374_Atlas" ^
    --out-dir ".\outputs" --tasks MOTOR EMOTION
```
Saves `outputs/<subject>/<subject>_<TASK>_<PHASE>.npz` (arrays `ts` = time x 374, `parcel_ids`,
`parcel_names`) and a log, `outputs/batch_log.csv`. Safe to re-run: finished runs are skipped.
Use `--limit 2` to test on two subjects first. Assumes zips are named `*_Task3TRecommended.zip`.

## Status
- Checked on real HCP data: subject 100206 MOTOR LR (single-run script), and 2 subjects x all
  14 runs (batch script): all checks passed.
- Not yet checked: that the signals line up with the task events (EV files).

## Rules
- No HCP data, atlas source files, or outputs in git (see `.gitignore`).
- No passwords or keys.
