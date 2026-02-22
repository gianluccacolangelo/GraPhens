# Non-Pipeline Inventory (Phase 1)

## Scope
This inventory marks scripts/modules that are not reachable from the Phase 1 supported entrypoints:

- `src/graphens.py`
- `src/simulation/phenotype_simulation/create_hpo_dataset.py`
- `training/datasets/convert_to_lmdb.py`
- `training/training.py`

No deletions are performed in Phase 1. This file provides suggested next-step disposition only.

## Disposition Labels
- `keep`: useful auxiliary tooling or documentation example.
- `deprecate`: likely still useful, but should be migrated, archived, or moved out of `src`.
- `candidate delete`: appears legacy/experimental and should be reviewed first for external usage.

## Inventory
| Path | Why non-pipeline in Phase 1 | Suggested disposition |
|---|---|---|
| `src/examples/quick_start.py` | Demo/example script, not called by supported entrypoints | keep |
| `src/examples/create_dummy_embeddings.py` | Example utility, not used in main flow | keep |
| `src/examples/lookup_embeddings.py` | Example utility, not used in main flow | keep |
| `src/graph/demo_adjacency.py` | Demo only | deprecate |
| `src/graph/demo_assembler.py` | Demo only | deprecate |
| `src/graph/demo_validation.py` | Demo only | deprecate |
| `src/augmentation_demo.py` | Demo script, not imported by supported entrypoints | deprecate |
| `src/augmentation/test.py` | Ad-hoc script-style test module | deprecate |
| `src/ontology/example_check_deprecated.py` | Example script, not part of runtime pipeline | deprecate |
| `src/embedding/evaluation.py` | Model-evaluation utility outside main pipeline | deprecate |
| `src/embedding/vector_db/scripts/build_vector_db.py` | Offline embedding utility | keep |
| `src/embedding/vector_db/scripts/build_all_vector_dbs.py` | Offline embedding utility | keep |
| `src/embedding/vector_db/scripts/convert_to_memmap.py` | Offline embedding utility | keep |
| `src/embedding/vector_db/scripts/find_similar_phenotypes.py` | Offline similarity utility | keep |
| `src/embedding/vector_db/scripts/demo_similarity.py` | Demo-only script | deprecate |
| `src/simulation/phenotype_simulation/demo.py` | Demo-only workflow | deprecate |
| `src/simulation/phenotype_simulation/integration_demo.py` | Demo model pipeline, not runtime entrypoint | deprecate |
| `src/simulation/phenotype_simulation/ablation_simulation.py` | Experiment script | candidate delete |
| `src/simulation/analysis.py` | Research analysis script, not main pipeline | candidate delete |
| `src/simulation/analyze_phenotypes.py` | Analysis script, not runtime pipeline | candidate delete |
| `src/simulation/fenotipos_processing.py` | Data-cleaning script, not runtime pipeline | candidate delete |
| `src/simulation/fenotipos_cleanup.py` | Data-cleaning script, not runtime pipeline | candidate delete |
| `src/simulation/keep_leaves_of_goldstandard_dataset.py` | Dataset maintenance script, not runtime pipeline | deprecate |
| `src/simulation/update_goldstandard_dataset.py` | Dataset update script, not runtime pipeline | deprecate |
| `src/simulation/update_deprecated_hpo.py` | Maintenance script, not runtime pipeline | deprecate |
| `src/simulation/example_update_deprecated.py` | Example script | deprecate |
| `src/simulation/gene_phenotype/demo.py` | Demo script | deprecate |
| `src/ontology/orphanet_mapper.py` | Specialized mapper used by external analysis scripts only | candidate delete |
| `src/simulation/translation.py` | Utility used by non-pipeline scripts only in current scope | candidate delete |
| `training/datasets/test.py` | Dataset test utility, not in main training execution path | deprecate |

## Notes
- These labels are recommendations for the next wave only.
- Before any removal, verify external references in notebooks, ad-hoc scripts, and automation jobs.
