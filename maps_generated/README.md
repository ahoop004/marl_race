# phase5_v2 runtime maps

This distribution contains the 60 frozen Phase 5 V2 tracks used by `marl_race`
(40 train, 8 validation, 12 test). The manifest records the original track
IDs, split membership, and family classifications.
The `configs/maps/generated*.yaml` presets define the map order and subsets
used by experiments. [Runtime manifest](manifest.json)

Each track retains its map YAML, PNG occupancy image, centerline and wall
CSVs, and metadata JSON. The map YAML contains the grid origin, resolution,
spawn poses, and finish line. Track metadata includes the selected generation
parameters and geometry provenance. Its file references cover the retained bundle.

Verify the runtime distribution from this directory with `sha256sum -c SHA256SUMS`.
The checksum list covers every shipped file except itself. The retained map YAML,
PNG, centerline, and wall files use their original hashes. Per-track metadata
hashes reflect the removal of references to archived companion files.

The frozen research release contains the original metadata and checksums, native
replay recipes, alternate PGM exports, reports, and publication evidence. The
dataset specification, selection plan, and selected-candidate hashes in the
runtime manifest identify that release.
[Full research release in map-edit](https://github.com/ahoop004/map-edit/tree/50dac7614d01c969bcac0ab2406272bbe0facc26/datasets/phase5_v2)
