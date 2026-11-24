> **Note:** this is the original README of the [KA-Edge-AI-Node](https://github.com/codeNick7/KA-Edge-AI-Node) contest repository, moved here from the repo root on 2026-11-24. Paths below are relative to the repository root. The current README is [../README.md](../README.md).

# Karbon-Assure Edge AI Node

An edge-AI field device for soil-health monitoring and digital MRV (Measurement,
Reporting, and Verification). It senses, reasons, and runs inference entirely
on-device on an Arduino UNO Q, then syncs compact, self-explaining evidence
records to the cloud instead of streaming raw telemetry.

Built for the **Arduino Physical AI Challenge India 2026** (Robu.in x Arduino).
See `demo/Karbon-Assure_Arduino_Physical_AI_Challenge_Report.docx` for the full
contest report.

## How it works

Two potted plants (Pot A control, Pot B drought-stressed) are each instrumented
with a capacitive soil-moisture probe, a BME688 environmental/gas sensor, and a
BH1750 ambient-light sensor. The pipeline runs in four on-device stages:

1. **Sense** — the UNO Q's STM32U585 real-time domain polls all sensors at 1 Hz
   and streams readings to the Linux domain over Bridge RPC, while also logging
   continuously to a microSD card.
2. **Reason** — the QRB2210 Linux domain converts raw readings into soil-type-
   agnostic relative features (RSMI, light delta, gas-index trend).
3. **Detect** — a compact trained transformer classifier (1 self-attention block,
   1,155 parameters, exported to dependency-free pure Python) classifies soil-moisture
   state with a confidence score. Trained on 1.27M real readings from the controlled
   Pot A/B trial — **98.9% train / 88.9% test accuracy** (chronological split: train
   on the earlier 80%, test on the later 20%) — see `docs/model-choice-jury-justification.md`
   for the full training methodology, accuracy numbers, and why this architecture was
   chosen over the earlier decision tree. A standalone background routine
   (`scripts/inference_service.py`) computes this continuously; the dashboard only
   reads the results, it never runs the model itself.
4. **Synchronize** — HMAC-signed evidence records are queued in a durable local
   store and pushed to the Karbon-Assure ingestion API, with automatic retry on
   failure.

See `docs/karbon-assure-edge-node-system-architecture.md` and
`docs/karbon-assure-edge-node-PRD.md` for the full design.

## Repository layout

- `KA-Edge-source/` — board code that runs on the Arduino UNO Q
  - `sketch/sketch.ino` — STM32 real-time firmware (sensor polling, SD logging)
  - `python/main.py` — Linux-side app (feature extraction, evidence sync to the cloud)
  - `python/edge_node/` — feature computation, validity checks, the trained model
    (`trained_model.py`, dependency-free — see below)
  - `python/push_to_karbon_assure.py`, `sync_queue.py` — HMAC-signed sync with a durable local queue
- `source/dashboard/` — Flask dashboard for live/raw sensor views and inference results;
  a pure reader of the inference sidecar, computes nothing itself
- `source/edge_node/` — model training/export code (`tiny_transformer.py`,
  `export_transformer.py`, plus the legacy decision-tree tooling `train.py`/`export.py`/
  `decision_tree.py`) and offline validation — dev-machine only, not deployed to the board
  (`trained_model.py` is the one generated file both sides share)
- `source/cloud_mock/` — mock cloud-side demo UI
- `source/tests/` — test suite (pytest)
- `scripts/` — training (`train_transformer.py`, `build_training_set.py`), the standalone
  background inference routine (`inference_service.py`), data-migration, and deploy scripts
  (e.g. `deploy_edge_node_to_board.sh`, `deploy_dashboard_to_board.sh`)
- `docs/` — architecture, PRD, BOM, citations, execution plan
- `demo/` — contest report, abstract, and presentation decks

## Running the dashboard

The dashboard only reads inference results — start the background inference service
first (or at least once, to populate the sidecar) so there's something to show:

```bash
.venv/bin/python scripts/inference_service.py --data-dir <dir-containing-sensor_data.db> --watch
```

```bash
.venv/bin/python -m flask --app source.dashboard.app run --host 0.0.0.0 --port 8080
```

or directly:

```bash
.venv/bin/python source/dashboard/app.py
```

## Training the model

Retrains `edge_node/tiny_transformer.py`'s classifier on a labeled training set and
exports it to the dependency-free on-device module both the board and the dashboard
share (`source/edge_node/trained_model.py`):

```bash
.venv/bin/python scripts/build_training_set.py /path/to/sensor_data.db \
    --out training_set.jsonl
.venv/bin/python scripts/train_transformer.py training_set.jsonl \
    --export source/edge_node/trained_model.py
```

Trains only on the three flags with real, sustained support in the current trial data
(`stable`, `waterlogged_risk`, `soil_health_degrading`) — see
`docs/model-choice-jury-justification.md` for why, and for the full accuracy breakdown
including the leave-one-pot-out generalization check.

## Running tests

```bash
.venv/bin/python -m pytest source/tests
.venv/bin/python -m pytest KA-Edge-source/python
```

## License

MIT — see `LICENSE`.
