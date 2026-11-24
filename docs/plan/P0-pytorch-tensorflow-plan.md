# P0 Plan: Edge AI Node v2 (PyTorch + TensorFlow + TFLite)
_Started 2026-11-23. PRD: `docs/plan/PRD.md`. Original repo `~/workspace/Contest` stays untouched._

## Goal
Re-implement the hand-written 1,155-parameter transformer in **PyTorch** and **TensorFlow/Keras**, export to
**TFLite (int8)** and **ONNX**, run it on the UNO Q (MCU if possible, Linux side as fallback), and publish an honest
comparison: hand-written vs PyTorch vs TensorFlow vs TFLite.

## Ground rules
- No changes to the board's working contest setup without the user's go-ahead. The user does every flash and board command.
- Keep the hand-written model and its tests as the baseline; new code goes in a new `ml/` folder.
- Python 3.12 venv (`uv venv --python 3.12`); keep the existing `.venv` setup intact.
- The user creates the new GitHub showcase repo and adds it as `origin`; no pushes from Claude.

## Findings so far (2026-11-23)
- **October:** TFLite on the UNO Q MCU was **never actually tried**. It was judged too risky for the contest timeline
  (user, 2026-11-23). Forum reports at the time described linker errors (undefined math functions) for TFLite C++ libraries on the UNO Q.
- **Arduino Zephyr core releases since then** (github.com/arduino/ArduinoCore-zephyr/releases):
  - 0.54.1 (Mar): "Fix linker errors when using CPP features"; missing time and math function exports
  - 0.55.0 (Apr): `fmod` exported
  - 1.0.0 (2 Sep): missing 32-bit float AEABI helpers exported
  - 1.1.0 (29 Oct, latest): "fix C library linkage"; updated RouterBridge / RPClite
  - None mention TensorFlow by name. Several fix exactly the error class seen before, so a retest on **1.1.0** is worthwhile.
- **Vanilla Zephyr route** (Edge Impulse guide; Derek Molloy, Jul 2026): `west build -b arduino_uno_q`, flash/debug through
  `adb forward tcp:3333` + `arduino-debug` + `west debug -r openocd`. This **replaces the Arduino sketch loader and Bridge RPC**
  until restored with `arduino-cli burn-bootloader -b arduino:zephyr:unoq -P jlink`. Use only if route A fails, and only with the user's OK.
- **Training data (checked 2026-11-23):**
  - Committed `data/training_set.jsonl.gz` (171,010 rows; stable / moisture_low) is the older decision-tree set.
  - The shipped transformer was trained on **`data-from-board/sept15/`** (pot_a/pot_b readings JSONL), Pot A rows before
    2026-09-10T08:23:00Z excluded (corroded sensor) → 1,582,346 in-scope rows; 1,265,876 = the 80% chronological train part.
    That Pot A cut-off was applied by hand; it is not in any script.
  - `Oct05/data/sensor_data.db` (copied to `data-local/oct05/`, gitignored) holds **raw 1 Hz readings only for
    2026-09-18 → 2026-10-05** (~265k rows per pot); older days were pruned by the board to hourly rows (680 per pot).
    So Oct05 alone is NOT the full history.
  - Leave-one-pot-out in the original work was **near-zero accuracy** both ways (Pot A's RSMI clipped to 1.0 by the
    clay_loam texture bounds). The user decided to ship on the chronological split. Don't present LOPO as a success.
- **Fallback D (user, 2026-11-23):** if the board can't run TensorFlow/TFLite at all, run inference on the Mac with sensor data
  imported from the board.
- **Pre-existing test failure:** `source/tests/test_board_db.py::test_prune_old_protects_unprocessed_day_and_does_not_archive_it`
  fails because the test pins `_NOW` to 2026-09-25 while `prune_old()` uses the real clock (hard cap 21 days has passed).
  Not caused by the new venv; unrelated to P0. Left as is pending the user's call.

## Phases and TODO

### Phase 0 — MCU feasibility spike (board; user runs the commands) — ROUTE A WORKS (2026-11-24)
- [ ] Check the board's Arduino Zephyr core version (was 0.90.0; App Lab offered arduino:zephyr 1.1.0 and the user is upgrading, 2026-11-23). App Lab is used (not the IDE); adb not installed on the Mac
- [x] Board core upgraded 0.90.0 → **1.1.0** via App Lab; contest app verified working afterwards
- [x] Library choice: **EloquentTinyML 3.0.1 + tflm_cortexm** (its examples are ESP32/Iris only, so we wrote our own)
- [x] Own smoke test: `ml/board_test/make_smoke_model.py` trains a 4→16→3 Keras MLP on real sept15 features
      (FULLY_CONNECTED + SOFTMAX only, 2,280 bytes) and writes `ml/board_test/tflm_smoke/sketch/model.h` with 6 samples
      and the Mac's expected class/probabilities (TFLite reference kernels, no XNNPACK). `sketch.ino` prints PASS/FAIL per sample
- [x] User compiled + ran in App Lab as a new app (TensorFlow-Test): **6/6 PASS, 34 µs/prediction** after 2 local library patches
- [x] If route A compiles: flash, run inference, read the output — DONE (output via Monitor)
- [x] Outcome written to `docs/tflite-on-unoq-findings.md`
- [ ] Vendor a patched tflm_cortexm into the repo so the build doesn't depend on App Lab's cache
- [ ] Check which TFLM ops tflm_cortexm 1.0.0 provides vs the transformer's op list (after Phase 3 conversion)
- [ ] If route A fails: decide with the user between route B (vanilla Zephyr + tflite-micro module; loses Bridge until restored)
      and route C (LiteRT on the QRB2210 Linux side)

### Phase 1 — Data (Mac)
- [x] Python 3.12 venv: torch 2.15.0 (MPS ok), tensorflow 2.21.0 (CPU), onnx 1.23.2, onnxruntime 1.32.0; existing tests 136 pass / 1 pre-existing failure
- [x] Datasets (user: both): `ml/data/build_datasets.py` applies the in-scope filter and the **Pot A cut-off 2026-09-10T08:23:00 UTC**
      and writes `<name>.jsonl` + `<name>.meta.json` (split index/timestamp, counts) to `data-local/datasets/` (gitignored)
  - **sept15:** 1,582,352 rows (original 1,582,346; 6-row difference, likely pre- vs post-filter at the cut-off).
    Train 1,265,881 / test 316,471, split at 2026-09-11T16:50:08Z. Built from copies (migrate_jsonl_to_sqlite run on copies only)
  - **combined (sept15 + Oct05):** 2,100,958 rows; train 1,680,766 / test 420,192, split at 2026-09-30T05:54:17Z.
    No raw readings 2026-09-15 → 09-18. Oct05 adds 11,221 sensor_fault rows (excluded, as by design)
  - **Oct05 export is corrupt at source** (`database disk image is malformed`; same checksum as the Contest original).
    Recovered with `sqlite3 .recover` → `data-local/oct05/sensor_data.recovered.db` (integrity ok). Lost 2 Pot A and 36 Pot B readings
  - Oct05 raw density is ~0.18 readings/s (265k per pot over 17 days), not 1 Hz; there are gaps or a lower sample rate in that period
  - Caveat: every in-scope Pot A row is `waterlogged_risk` (RSMI clipped to 1.0, same clay_loam calibration issue)
- [ ] Freeze the split: chronological 80/20 + leave-one-pot-out; save split indices so all frameworks use identical data

### Phase 2 — PyTorch model (Mac) — DONE 2026-11-23 (sept15)
- [x] `ml/common/data.py`: shared loader; reuses the original `compute_bin_edges`/`quantize` (16 quantile bins fitted on the train part), cached as .npz
- [x] `ml/pytorch/model.py`: same architecture, **1,155 params**; `load_handwritten_params` maps the numpy weights across
- [x] `ml/pytorch/train.py`: Adam lr 0.05, batch 4096, 60 epochs, keep the lowest-loss epoch (as the original)
- [x] sept15 result: **98.86% train / 88.87% test** (published hand-written: 98.9% / 88.9%). Fit 70.6 s on CPU; MPS 129.2 s with identical accuracy (GPU overhead > gain for a 1,155-param model)
- [x] Loss plateaued at ~0.25 for ~20 epochs, spiked to 0.35, then fell to ~0.047: lr 0.05 is on the high side; tune later (both frameworks, same settings)
- [x] ONNX (`ml/pytorch/export_onnx.py`, dynamo exporter, needs onnxscript): 0 class mismatches on 20,000 test rows, max prob diff 1.2e-7; 36 KB; 0.02 ms/row onnxruntime on Mac CPU
- [x] Tests `ml/tests/test_pytorch_model.py` (6 pass): param count; forward equals hand-written with the same weights (rtol 1e-9); gradients equal the hand-written backward; ONNX round-trip
- [x] `ml/requirements.txt` pins the ML stack
- [x] Combined dataset run: 97.36% train / **99.86% test — NOT a real gain; do not headline it.** The test period (2026-09-30 → 10-05)
      is easy: 210,096 of 420,192 test rows are Pot A with rsmi == 1.0 and rsmi_trend == 0.0 (always waterlogged_risk), and Pot B has
      no waterlogged_risk rows in that period. sept15's test errors were mostly Pot B stable → waterlogged_risk (27,797).
      **Headline comparison = sept15.** For combined, report per-pot results and this caveat
- [ ] Leave-one-pot-out (optional; known near-zero for the calibration reason; report only if run for all frameworks)

### Phase 3 — TensorFlow model (Mac)
- [ ] `ml/tensorflow/model.py`: Keras functional model, same architecture and parameter count
- [ ] Train with the same split; compare accuracy
- [ ] Export TFLite float32 and int8 (post-training quantisation with a representative dataset); measure the accuracy drop
- [ ] Generate the C array (`xxd -i`) for the MCU build
- [ ] Tests (param count, TFLite round-trip within tolerance)

### Phase 4 — On-device deployment (board; user flashes)
- [ ] Deploy via the route chosen in Phase 0; measure latency, RAM, flash
- [ ] Linux side: LiteRT (`ai-edge-litert`) inference service as an alternative to the pure-Python model; compare latency
- [ ] Keep the evidence-record format; stamp the framework + model version

### Phase 5 — Write-up and repo hygiene
- [ ] README section + `docs/framework-comparison.md`: accuracy, parameters, file size, latency, effort for hand-written / PyTorch / TF / TFLite int8
- [ ] GitHub Actions: pytest + a short training smoke test for both frameworks
- [ ] User creates the showcase repo; add as `origin`
- [ ] Update `Resume-Service/docs/career-master.md` with verified facts only

## Progress log
| Date | Update |
|---|---|
| 2026-11-23 | Repo cloned; plan written; Zephyr core release notes reviewed (1.1.0 has C-library/linker fixes worth retesting) |
| 2026-11-23 | Board updating. Linux-side packages before update: arduino-app-cli 0.14.1, arduino-app-lab 0.12.0, arduino-router 0.11.0, arduino-unoq 0.2.1, arduino-unoq-config 0.2.1, arduino-linux-config 0.3.0, arduino-cloud-connector 0.1.2. MCU core (arduino:zephyr) version still to check |
| 2026-11-23 | Venv set up; data sources traced (sept15 = shipped model; Oct05 = Sep 18–Oct 5 raw only); fallback D (Mac inference) added |
| 2026-11-23 | Board core 0.90.0 → 1.1.0 via App Lab update (in progress). Both datasets built; Oct05 DB recovered from corruption |
| 2026-11-23 | Phase 2 PyTorch done on sept15: 98.86% / 88.87%, ONNX verified, 6 tests pass. Existing suites unchanged (103+33 pass, 1 pre-existing failure) |
| 2026-11-23 | Combined PyTorch run: 99.86% test explained as an easy test period (clipped Pot A + no Pot B waterlogged rows); sept15 stays the headline |
| 2026-11-23 | Hand-written baseline retrained on rebuilt sept15: **98.86% / 88.87%** (identical to PyTorch), fit 226.1 s (PyTorch CPU 70.6 s). LOPO on rebuilt data: train A→test B 2.70%, train B→test A 58.14% (poor, not near-zero both ways) |
| 2026-11-23 | Board core now arduino:zephyr 1.1.0 (via App Lab); contest app verified building and running after the upgrade. TFLM library choice for route A: tensorflow-runtime-universal (Simone Salerno, Cortex-M + ESP32); fallbacks tflm_cortexm, Chirale_TensorFlowLite; Adafruit TensorFlow Lite rejected (tied to Adafruit/Arcada boards) |
| 2026-11-23 | Smoke-test model + sketch written (not compiled locally: no arduino-cli/zephyr core on the Mac); handed to user for App Lab |
| 2026-11-23 | Route A first build on board (core 1.1.0, EloquentTinyML 3.0.1, tensorflow-runtime-universal 1.0.0 + tflm_cortexm 1.0.0): all TFLM kernels compiled for cortex-m33; failed in tensorflow-runtime-universal src/tensorflow/lite/micro/system_setup.cpp: `ulong` not defined on the Zephyr core (lines 68, 72, 88, 122, 146). Not the earlier linker/libm issue. Next: check whether tflm_cortexm duplicates TFLM sources; fix = drop the duplicate lib or add `typedef unsigned long ulong;` |
| 2026-11-23 | tflm_cortexm 1.0.0 ships its own full TFLM copy (tensorflow/, third_party/, cortex-m4/) with the same `ulong` use in system_setup.cpp. Fix applied by user: remove tensorflow-runtime-universal (duplicate TFLM) + local patch `typedef unsigned long ulong;` at line 1 of tflm_cortexm system_setup.cpp (backup .orig). Patch lives in App Lab internal cache, may be overwritten; if route A works, vendor a patched copy into the sketch |
| 2026-11-23 | Rebuild with tflm_cortexm only + ulong patch: full compile OK; link failed with undefined refs to TFLM helpers (UnpackDenseInt4IntoInt8, CalculateActivationRangeQuantized, GetQuantizedConvolutionMultipler), not libm. Hypothesis: library uses dot_a_linkage and duplicate basenames (kernel_util.cpp, portable_tensor_utils.cpp in two dirs) overwrite each other in tflm_cortexm.a. Verifying on board before any change |
| 2026-11-24 | Hypothesis CONFIRMED on board: tflm_cortexm library.properties has dot_a_linkage=true; duplicate basenames kernel_util.cpp, portable_tensor_utils.cpp, tensor_utils.cpp; tflm_cortexm.a holds each once (overwritten). Fix applied by user: dot_a_linkage=false (backup library.properties.orig). Second local patch in App Lab cache |
| 2026-11-24 | **Route A works**: TFLM smoke MLP on the UNO Q MCU, 6/6 PASS (probabilities match the Mac to <1e-6), 34 µs/prediction. Findings doc written; duplicate op registration removed from sketch |
| 2026-11-24 | Phase 3 started (NOT in the first commit): Keras port `ml/keras_tf/tf_model.py` matches the hand-written model (equivalence tests pass), but the TFLite converter still emits BATCH_MATMUL (not on the board). Board op list saved to ml/board_test/tflm_cortexm_ops.txt |
