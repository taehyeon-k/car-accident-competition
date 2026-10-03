# Repository cleanup record

The cleanup retains the three best recorded methods, their executable inference dependencies, required vendor licenses, checkpoint identities/restoration, training sources and portable training instructions. Unrelated campaign outputs, feature caches, dataset manifests, prediction arrays, repeated vendored trees, obsolete submissions and superseded reports are removed from the active tree.

The cleanup preserves the original main history. The archive tag `archive/pre-cleanup-2026-10-03` points to the latest WIP tip, whose ancestors include both other experiment tips. It preserves all three histories before their obsolete branch names are removed. A separately prepared Git bundle also contains the original histories and the tested cleanup tree. No history is rewritten or old Git objects purged. Clone history can therefore still contain old artifacts; use a shallow clone for the compact current tree.

| Prior branch | Preserved tip |
|---|---|
| `main` | `c437a5b5baae5227e101eefd00b44372a1dfdb07` |
| `handoff-2026-09-24` | `2f065e3dd6e44d4f8d85c785c3d56a30c28f29c9` |
| `stage3-v2-experiments` | `18ec7cb5431dd80495474faa800b773a48204f9e` |
| `wip-stage3-snapshot-2026-09-29` | `012883a417710f60db9963373148e34b592a75de` |

To restore an old branch locally, for example:

```bash
git fetch origin
git switch -c recovered-handoff 2f065e3dd6e44d4f8d85c785c3d56a30c28f29c9
```

Stage 2 training source is from the reviewed handoff snapshot. Stage 3 inference/training source is recovered from the recorded v8 submission, so unreviewed changes in the later WIP branch do not silently alter the retained method. Stage 1 preserves the submitted model and inference functions; unrelated Stage 2/3 placeholder functions were removed. Its new trainer is explicitly labeled as a reconstruction.

Publishing and branch cleanup use the GitHub browser interface because the connector denied write access. The archive tag records the prior experiment tips without keeping active experiment branches. R2 was inspected read-only. Original archives, data and caches in the bucket remain intact because the requested storage cleanup concerns GitHub's active repository.
