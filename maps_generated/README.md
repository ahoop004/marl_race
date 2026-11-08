# phase5_v2 — frozen 60-track research dataset

60 geometrically validated, independently screened layouts: **40 train / 8
validation / 12 test**, with **18 short / 22 medium / 20 long** tracks. Training
contains 20 specialists and 20 composites. All 30 specialists and 30 composites
retain their preassigned slots and lineages.

[Complete gallery](plots/overview.png) · [Gallery index](index.html) ·
[Master manifest](manifest.json) · [Train](train_manifest.json) ·
[Validation](validation_manifest.json) · [Test](test_manifest.json) ·
[Statistics](track_statistics.csv) · [Integrity report](integrity_report.json)

The Phase 5.2E candidate pool was verified, then its exact selected candidates were
frozen and published. Publication performs no alternative search or substitution.
Every native recipe, track ID, centerline identity, coordinate hash and geometric
statistic matches the frozen preflight. All 1,770 selected pairs, including 896
cross-split pairs, pass the unchanged `development_screen.v1` policy.

## Frozen selection

- Plan identity: `448fb2b45d961d105c7c0cac0dc134833ff8e0d0e597507896ca0c7a31b95642`
- Canonical selected-list SHA-256: `bf78bd2f3277c78db7231736762aca161b380d0e822f8be759d6667f12babcd8`
- [selection_plan.json](selection_plan.json) records all 60 slot/attempt/seed/native
  recipe identities, full resolved requests, expected statistics and selector evidence.
- `selection_plan.sha256` checks the actual plan file bytes. `selected_candidates.sha256`
  records the canonical JSON selected-list identity, not a separate file's hash.
- `development/preflight_pool.json` and `development/preflight_protocol.json` preserve
  the input pool and bounded selection protocol. Original calibration and the vehicle
  snapshot are copied alongside them; V1 development evidence is unchanged.

Candidate outcomes are separate: 60 published candidates in `evidence/`, 17 rejected
preparations in `candidate_evidence/geometry_rejected/`, and 74 valid-but-unselected
candidates in `candidate_evidence/valid_unselected/`. Native recipes and failure
journals remain available for every pool candidate.

## Replay, checksums and recovery

Each standalone ROS-compatible bundle contains PNG/PGM occupancy, map YAML, native
recipe, centerline/wall/profile CSVs, metadata, evaluation CSV and validation report.
Every bundle was reloaded from native YAML, regenerated and compared across ten
artifacts byte-for-byte: **600 artifact replay comparisons**. The manifests verify
**660 per-track file checksums**, including the 60 validation reports. `SHA256SUMS`
covers the complete finalized release, including plots and selection evidence.

From this directory, run `sha256sum -c SHA256SUMS`. From the repository root:

```sh
python3 scripts/generate_research_dataset.py audit datasets/phase5_v2 --require-complete
```

The selected-publication coordinator stages bundles atomically and checkpoints
manifests. Explicit resume verifies inputs and adopts a fully validated orphan
bundle only when it matches the frozen selected recipe. A specialist/composite
subset test exercised interruption after bundle publication and recovered without
rewriting the bundle. Completed resume is read-only and can use archived inputs:

```sh
python3 scripts/generate_research_dataset.py publish-selection \
  datasets/phase5_v2/selection_plan.json --output datasets/phase5_v2 --resume \
  --snapshot datasets/phase5_v2/development/vehicle_snapshot.json \
  --calibration datasets/phase5_v2/development/calibration.json
```

The offline regression result is recorded in [release_tests.json](release_tests.json).

## Geometry review and limits

**28/30 composites** have repetition below .65. High-repetition exceptions are
`train_composite_03` and `train_composite_06`; both remain as frozen preflight choices.
Nine composites have active-curvature exposure below the review threshold of 25%;
[review_flags.json](review_flags.json) lists their IDs and measured fractions.
These values agree with preflight; nothing was silently replaced or rejected.
Primary annotation coverage includes genuine fixed primary straights and is
reported separately from active curvature.

Physical drivability remains **unassessed**, difficulty **unrated**, and simulator
integration **not run**. Clearance is checked at sampled poses. Independence means
passing the frozen geometric policy, not distinct topology or held-out topology
families. Low repetition does not establish driving difficulty. Replay is exact
in the recorded [software environment](environment.json). No simulator or training
was used; V1 and existing pilots are preserved.
