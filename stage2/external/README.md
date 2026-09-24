# Stage-2 external data pipeline

This directory implements metadata-first collection for MM-AU and CausalCrash.
It preserves the repository's native Stage-2 labels: original numeric frame IDs,
`entry_side` as image-coordinate `LEFT`/`RIGHT`, and binary `evasion_space`.

The safe default downloads metadata and only strict CausalCrash candidates:

```bash
bash stage2/external/run_pipeline.sh
```

To additionally stream the 42.42 GiB MM-AU CAP types 1–10 archive and extract
only strict candidates:

```bash
bash stage2/external/run_pipeline.sh --with-mmau-cap-1-10
```

The MM-AU option never stores the compressed archive. Other MM-AU archives are
intentionally not implicit because each requires a separate 28–115 GiB transfer.
MM-AU publishes JPEG sequences rather than source video files. Selected sequences
are encoded frame-for-frame as H.264 MP4s with `.frames.json` sidecars mapping
video positions to release frame IDs; the temporary JPEG trees are then removed.

Launch the persistent labeling service:

```bash
bash stage2/external/start_stage2_external_labeling.sh
```

Back up curated artifacts after validation:

```bash
bash stage2/external/backup_r2.sh
```

Runtime Python dependencies beyond the repository environment are `openpyxl`,
`xlrd`, `yt-dlp`, and `flask`. FFmpeg, rclone, and supervisor are instance tools.
