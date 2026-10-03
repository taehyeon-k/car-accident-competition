# Cleanup verification — 2026-10-03

Verified before publication:

* The R2 v8 archive size and SHA-256 were recorded; all sixteen retained weight files were hashed individually.
* Stage 1 and Stage 3 checkpoints and the Stage 2 backbone are byte-identical in the downloaded v8/v10 archives.
* Stage 1's submitted model loads its checkpoint strictly and passes a CPU forward check.
* Stage 2's backbone and twelve inference members load. All twelve member checkpoints also load strictly into the retained original training architecture and pass CPU forward checks.
* Stage 3's V3 EMA model and SEA-RAFT-S load and pass a model forward check; original loss/trainer imports succeed.
* Real sample integration: Stage 1 returned an ID/class row for one baseline video; Stage 2 processed eight decoded frames and returned all five required columns; Stage 3 decoded four frames, ran SEA-RAFT, geometry, V3 and decoding, and returned four acceleration/steering labels.
* The reconstructed Stage 1 preprocessing generated correctly shaped tensor inputs. One CPU training epoch, gradient accumulation, validation and checkpoint writing completed on a tiny smoke dataset.
* All three training entry points expose their CLI; Python source compilation succeeds.
* The offline ZIP builder verified checkpoint hashes and created the required root layout (`inference.py`, `requirements.txt`, `model/`). Its requirements file uses DACON's preinstalled dependencies, avoiding replacement of the evaluation server's PyTorch.

The smoke tests establish runtime and packaging compatibility, not model accuracy. No GPU was available here, so a full L40S timing run and full-dataset training were not performed. Stage 2 and Stage 3 historical training performance is supported by their preserved source configurations and project reports, not by retraining during this cleanup.

CPU checks used PyTorch 2.8.0+cpu / torchvision .23.0+cpu; auxiliary installed libraries may differ from DACON's pinned server versions. The underlying model and flow checkpoints remain unchanged.

To repeat the checkpoint/model checks after restoration:

```bash
pip install -r requirements-training.txt
python scripts/fetch_checkpoints.py
python scripts/verify_models.py
```
