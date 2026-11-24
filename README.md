# Edge Transformer: NumPy → PyTorch → TensorFlow → a microcontroller

A 1,155-parameter transformer that classifies soil health from sensor features. It is written three times (hand-written
NumPy, PyTorch, TensorFlow/Keras) and runs on the **Arduino UNO Q's STM32U585 microcontroller** with **TensorFlow
Lite Micro**, giving the same answers as on a laptop.

It extends the [Karbon-Assure Edge AI Node](https://github.com/codeNick7/KA-Edge-AI-Node), an evidence-grade soil-health
monitor for digital MRV (Measurement, Reporting and Verification) of carbon projects
([project article on Electronics For You](https://www.electronicsforu.com/electronics-projects/karbon-assure-edge-ai-node-an-evidence-grade-soil-health-monitor-for-digital-mrv)).

## Outcome at a glance

| | Result |
|---|---|
| Same model in 3 implementations | Hand-written NumPy, PyTorch, TensorFlow/Keras: **1,155 parameters each** |
| Accuracy (identical in all three) | **98.86% train / 88.87% test** on 1.58M real sensor rows (chronological 80/20 split) |
| Proven equivalent | With the same weights, PyTorch and Keras outputs match the hand-written model (PyTorch gradients too); 11 tests |
| Training time (Mac CPU) | NumPy 226 s · **PyTorch 71 s** · TensorFlow 87 s |
| Exports | ONNX 36 KB (0 mismatches on 20,000 rows) · **TFLite float32 17 KB** (0 mismatches on all 316,471 test rows) |
| **On the microcontroller** | **8/8 test rows identical to the Mac** (incl. two rows the model gets wrong), probabilities within 1e-7 |
| **On-device latency** | **~1.96 ms per inference** on the STM32U585 (Cortex-M33 at 160 MHz); target was < 50 ms |
| Tried and dropped | int8 quantisation: 39.6% accuracy, not smaller. Float32 is what runs on the board |

## Why this is interesting

- **The same network three times.** The original model has hand-written forward and backward passes (no autodiff), checked
  against numerical gradients. The PyTorch and Keras ports are tested against it: load the hand-written weights and the
  logits (and, for PyTorch, the gradients) must match. So the accuracy numbers compare like with like.
- **A transformer on a microcontroller without BATCH_MATMUL.** The TFLite Micro build available for the board has no batched
  matrix multiply. Attention is written as a broadcast multiply + sum instead (same maths), and the weights are frozen to
  constants before conversion. The converted model uses 15 op types, all available on the board, and a test fails if any
  other op appears.
- **Getting TFLite Micro to build on the UNO Q.** Two library problems were found and fixed on the way: a missing `ulong`
  type on the board's Zephyr core, and archive linking that silently dropped TFLite objects whose files share a name. See
  [docs/tflite-on-unoq-findings.md](docs/tflite-on-unoq-findings.md).

## The model

```
4 features ──► per-feature quantile bins (16 each) ──► embedding + position ──► 5 tokens ([CLS] + 4)
   rsmi            (bin edges fitted on the
   light_delta      training part only)
   gas_index_trend
   rsmi_trend
                                 ┌──────────────────────────────────────────────┐
   tokens ──► LayerNorm ──► single-head self-attention ──► + residual          │  d_model = 8
          ──► LayerNorm ──► feed-forward (8 → 16 → 8, ReLU) ──► + residual     │  d_ff = 16
          ──► [CLS] vector ──► linear ──► softmax over 3 classes               │  1,155 params
                                 └──────────────────────────────────────────────┘
classes: soil_health_degrading · stable · waterlogged_risk
```

Features come from capacitive soil-moisture, BME688 gas and BH1750 light sensors on two pots (one watered normally, one
drought-stressed), sampled at 1 Hz. Labels are seed labels from a literature-based rule tree. See
[docs/contest-README.md](docs/contest-README.md) for the original device.

## Results in detail

### Data
| | |
|---|---|
| Source | Board export `sept15`: 2026-08-14 → 2026-09-15, both pots |
| Rows used | 1,582,352 (3 in-scope classes; Pot A rows before 2026-09-10 08:23 UTC removed: corroded sensor) |
| Split | Chronological: train 1,265,881 rows (up to 2026-09-11 16:50 UTC), test 316,471 rows (after) |

### Training (same data, same settings: Adam lr 0.05, batch 4,096, 60 epochs, keep the lowest-loss epoch)
| Implementation | Params | Train acc | Test acc | Fit time (Mac CPU) |
|---|---|---|---|---|
| Hand-written NumPy | 1,155 | 98.86% | 88.87% | 226.1 s |
| PyTorch 2.15 | 1,155 | 98.86% | 88.87% | 70.6 s |
| TensorFlow 2.21 / Keras | 1,155 | 98.86% | 88.87% | 87.1 s |

PyTorch on the Mac GPU (MPS) took 129 s with the same result: the model is too small to benefit.

Test confusion matrix (PyTorch and TensorFlow are identical; rows = true, columns = predicted):

| | soil_health_degrading | stable | waterlogged_risk |
|---|---|---|---|
| **soil_health_degrading** | 25,551 | 287 | 6,982 |
| **stable** | 172 | 97,426 | 27,797 |
| **waterlogged_risk** | 0 | 0 | 158,256 |

Most errors are `stable` readings called `waterlogged_risk`.

### Exports
| Format | Size | Check |
|---|---|---|
| ONNX (from PyTorch) | 36 KB | onnxruntime: 0 class mismatches on 20,000 test rows, max prob diff 1.2e-7; 0.02 ms/row on Mac CPU |
| TFLite float32 (from Keras) | 17,256 bytes | full test set: 88.87%, 0 mismatches vs Keras, max prob diff 3.0e-7 |
| TFLite int8 (experiment) | 17,376 bytes | 39.6% test accuracy: dropped |

### On the UNO Q microcontroller
| | |
|---|---|
| Board / chip | Arduino UNO Q, STM32U585 (Cortex-M33, FPU), Arduino Zephyr core 1.1.0, built in Arduino App Lab |
| Runtime | EloquentTinyML 3.0.1 + tflm_cortexm 1.0.0 (TensorFlow Lite Micro), with two local patches |
| Check | 8 real test rows: 2 correct per class + 2 the model gets wrong. Board class and probabilities compared with the Mac |
| Result | **8/8 PASS on every cycle**, max prob diff < 1e-7, both misclassifications reproduced |
| Latency | **~1.96 ms per inference** (1,942–1,969 µs) |

### Honest caveats
- **88.87% is on a chronological split** (train on earlier readings, test on later ones from the same two pots). It is not
  a cross-site result. A leave-one-pot-out check scores poorly (2.7% and 58.1%), because the `clay_loam` moisture
  calibration clips Pot A's moisture feature to its maximum, so Pot A never varies.
- **Every in-scope Pot A row is labelled `waterlogged_risk`** for the same calibration reason. All three frameworks learn
  this quirk equally, so the comparison is fair, but the labels are rule-based seed labels, not field observations.
- **The combined `sept15 + Oct05` run reaches 99.86% test accuracy, which is not a real gain**: half of that later test
  period is the clipped Pot A rows (always `waterlogged_risk`) and Pot B has no `waterlogged_risk` rows then.
- The learning rate (0.05, inherited from the original) is high: the loss spikes during training in all frameworks. Keeping
  the lowest-loss epoch handles it for now.
- The on-board test uses pre-computed bin ids. Binning raw sensor values on the microcontroller is not done yet.

## Repository layout

```
ml/                         ← this project's work
  common/data.py            shared loading + binning (reuses the original binning code)
  data/build_datasets.py    reproducible dataset build (Pot A cut-off, chronological split, .meta.json)
  pytorch/                  model.py · train.py · export_onnx.py
  keras_tf/                 tf_model.py · train.py · convert_tflite.py
  board_test/
    make_smoke_model.py     tiny MLP smoke test for the MCU
    make_transformer_header.py   writes model.h for the transformer sketch
    tflm_smoke/sketch/      first MCU test (FULLY_CONNECTED + SOFTMAX)
    tflm_transformer/sketch/ the transformer on the MCU
    tflm_cortexm_ops.txt    ops available on the board's TFLite Micro build
  artifacts/                trained models, metrics, logs (sept15, combined)
  tests/                    equivalence + conversion tests
source/edge_node/           original hand-written transformer, features, validity gate
KA-Edge-source/             original board firmware (sketch) and Linux-side Python
scripts/                    original training / data / deploy scripts
docs/                       findings, plans, original contest README and design docs
```

## Setup and reproduction

### 1. Mac / Linux environment
```bash
uv venv --python 3.12 .venv          # TensorFlow needs Python ≤ 3.12 here
uv pip install --python .venv/bin/python -r requirements.txt -r ml/requirements.txt
.venv/bin/python -m pytest ml/tests -q          # 11 tests: equivalence + board-op check
```

### 2. Data
The raw board exports (several GB of SQLite / JSONL) are **not in this repository**. The trained models and metrics are in
`ml/artifacts/`. With a board export:
```bash
.venv/bin/python scripts/migrate_jsonl_to_sqlite.py --data-dir <copy-of-export>      # if the export is JSONL
.venv/bin/python scripts/build_training_set.py <export>/sensor_data.db --out data-local/datasets/sept15_raw.jsonl
.venv/bin/python ml/data/build_datasets.py --sept15 data-local/datasets/sept15_raw.jsonl --out-dir data-local/datasets
```
`migrate_jsonl_to_sqlite.py` renames its input files, so run it on a copy.

### 3. Train and export
```bash
.venv/bin/python scripts/train_transformer.py data-local/datasets/sept15.jsonl      # hand-written NumPy
.venv/bin/python ml/pytorch/train.py         data-local/datasets/sept15.jsonl
.venv/bin/python ml/pytorch/export_onnx.py   data-local/datasets/sept15.jsonl
.venv/bin/python ml/keras_tf/train.py        data-local/datasets/sept15.jsonl
.venv/bin/python ml/keras_tf/convert_tflite.py data-local/datasets/sept15.jsonl
.venv/bin/python ml/board_test/make_transformer_header.py data-local/datasets/sept15.jsonl
```

### 4. Run on the UNO Q microcontroller
1. **Arduino App Lab**, board core **`arduino:zephyr` 1.1.0** (App Lab's software update).
2. Create an app; copy `ml/board_test/tflm_transformer/sketch/sketch.ino` and `model.h` into its sketch folder.
3. Add libraries **EloquentTinyML 3.0.1** and **tflm_cortexm 1.0.0**. Do not add `tensorflow-runtime-universal`
   (a second copy of TFLite Micro). Arduino_RouterBridge comes with the platform.
4. Patch tflm_cortexm on the board (once; App Lab keeps it in `~/.arduino15/internal/tflm_cortexm_1.0.0_*/tflm_cortexm`):
   ```bash
   L=$(ls -d ~/.arduino15/internal/tflm_cortexm_*/tflm_cortexm)
   F=$L/src/tensorflow/lite/micro/system_setup.cpp
   cp "$F" "$F.orig";  sed -i '1i typedef unsigned long ulong;' "$F"
   cp $L/library.properties $L/library.properties.orig
   sed -i 's/^dot_a_linkage=true/dot_a_linkage=false/' $L/library.properties
   ```
5. Build and run. The first build compiles TensorFlow Lite Micro on the board and takes ~15–20 minutes; later builds of the
   same app reuse the cache. The App Lab console should show:
   ```
   sample 0: true 0, mac 0, board 0, conf 1.0000, max|dp| 0.0000000  PASS
   ...
   sample 6: true 1, mac 2, board 2, conf 0.9764, max|dp| 0.0000000  PASS
   RESULT: 8/8 PASS, 1966 us per prediction
   ```
Details, error messages and fixes: [docs/tflite-on-unoq-findings.md](docs/tflite-on-unoq-findings.md).

## Roadmap
- [ ] Vendor a patched copy of tflm_cortexm into the repo (instead of patching App Lab's cache)
- [ ] Bin raw sensor values on the microcontroller and feed the result into the evidence records
- [ ] Tune the learning rate for all three implementations
- [ ] CI: tests plus a short training run for both frameworks

## License
MIT (see [LICENSE](LICENSE)).
