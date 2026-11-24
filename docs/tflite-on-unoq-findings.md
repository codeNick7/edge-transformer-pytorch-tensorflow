# TensorFlow Lite Micro on the Arduino UNO Q MCU — findings
_Verified 2026-11-24 on the user's UNO Q. Route A of `docs/plan/P0-pytorch-tensorflow-plan.md`._

## Result
**Works.** A TFLite model runs on the STM32U585 (Cortex-M33) through Arduino App Lab, with results identical to
TensorFlow Lite on the Mac.

| Item | Value |
|---|---|
| Test model | Keras MLP 4 → 16 (ReLU) → 3 (softmax), float32, trained on real sept15 soil features; ops FULLY_CONNECTED + SOFTMAX; 2,280 bytes |
| Samples | 6 (2 per class) with expected class + probabilities from the Mac's TFLite interpreter (reference kernels) |
| Board result | **6/6 PASS every cycle**; max abs probability difference < 1e-6 (prints as 0.000000) |
| Latency | **34 µs per prediction** steady state (39 µs first cycle) |
| Arena | 8,192 bytes configured (not yet minimised) |
| Output | App Lab console via `Monitor` (Arduino_RouterBridge); Bridge to the Linux side unaffected |

History: in October 2026 (contest) this route was judged too risky and **never tried**. Forum reports then described
linker errors (undefined math functions) for TFLite C++ libraries on the UNO Q; core 0.54.1–1.1.0 fixed that class.

## Result 2: the real transformer (2026-11-24)
| Item | Value |
|---|---|
| Model | The edge transformer (1,155 params), trained in TensorFlow/Keras, float32 TFLite, **17,256 bytes**, 15 op types |
| Ops | ADD, CAST, CONCATENATION, FULLY_CONNECTED, GATHER, MEAN, MUL, PACK, RESHAPE, RSQRT, SOFTMAX, SQUARED_DIFFERENCE, STRIDED_SLICE, SUB, SUM |
| Samples | 8 real sept15 test rows: 2 correct per class + 2 the model gets wrong |
| Board result | **8/8 PASS every cycle**, incl. both misclassifications reproduced with the same confidence; max abs probability difference < 1e-7 |
| Latency | **~1.96 ms per prediction** (1,942–1,969 µs) on the STM32U585; PRD target < 50 ms |
| Arena | 32,768 bytes configured (not yet minimised) |
| Build | Reused the TensorFlow-Test app so the compiled library cache was reused (a new app recompiles TFLM, ~15–20 min) |

How the model was made board-compatible (tflm_cortexm 1.0.0 has no BATCH_MATMUL):
- attention written as broadcast MUL + SUM instead of matmul (identical math)
- weights frozen to constants before conversion (`convert_variables_to_constants_v2`); left as variables, every dense layer
  became BATCH_MATMUL
- float32 input (bin ids) cast to int32 inside the model, so EloquentTinyML's float `predict()` works
- ops registered explicitly from the generated `model.h` (`REGISTER_MODEL_OPS`), each once, no `TF_OP_*` macros
- int8 post-training quantisation was tried and dropped: 39.6% test accuracy (see the P0 plan)

## Working setup
| Component | Version |
|---|---|
| Board core | `arduino:zephyr` **1.1.0** (updated through App Lab's software update; was 0.90.0) |
| App Lab | 0.12.0 → updated along with the core (user, 2026-11-23) |
| Toolchain | arm-zephyr-eabi 1.0.1 (GCC 14.3.0), `-mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard` |
| Wrapper | **EloquentTinyML 3.0.1** (Simone Salerno) |
| Runtime | **tflm_cortexm 1.0.0** (same author; ships its own full TFLM source tree) |
| NOT used | `tensorflow-runtime-universal` (a second, duplicate TFLM copy; remove it), `Adafruit TensorFlow Lite` (Adafruit/Arcada boards only), `EloquentTensorFlowCortexM` (older all-in-one) |

Sketch: `ml/board_test/tflm_smoke/sketch/` (`sketch.ino` + generated `model.h`).
Model generator: `ml/board_test/make_smoke_model.py`. Include order: `model.h` → `<tflm_cortexm.h>` → `<eloquent_tinyml.h>`.

## Problems hit and fixes (in order)
1. **Build looked stuck** after "library added from sketch project". Not stuck: App Lab compiles the whole TFLM
   source tree on the board's QRB2210 (first build ~15–20 min, no progress shown). Check with
   `ps aux | grep -o "micro/[a-z_/0-9]*\.cpp" | sort -u`. Later builds use `.cache/`.
2. **`'ulong' does not name a type`** in `tensorflow/lite/micro/system_setup.cpp` (lines 68, 72, 88, 122, 146).
   The Zephyr core has no `ulong` typedef (Mbed cores do). Present in both tensorflow-runtime-universal and tflm_cortexm.
   **Fix:** add `typedef unsigned long ulong;` as line 1 of tflm_cortexm's `system_setup.cpp` (backup `.orig`).
3. **Two TFLM copies** (tensorflow-runtime-universal + tflm_cortexm) were both in the app. **Fix:** remove tensorflow-runtime-universal.
4. **Link errors** `undefined reference to tflite::tensor_utils::UnpackDenseInt4IntoInt8`, `tflite::CalculateActivationRangeQuantized`,
   `tflite::GetQuantizedConvolutionMultipler`. Cause (verified): tflm_cortexm's `library.properties` has `dot_a_linkage=true`, and the
   source has duplicate basenames (`kernel_util.cpp`, `portable_tensor_utils.cpp`, `tensor_utils.cpp` in different folders); the
   `.a` archive keeps one object per name, so the other copy's functions vanish. **Fix:** `dot_a_linkage=false` (backup `.orig`).
5. **Warning** `Calling AddBuiltin with the same op more than once (Op: #9 / #25)`: harmless; `begin()` auto-registers ops from the
   `TF_OP_*` macros, so the sketch must not also call `tf.resolver.Add...()`. Removed from the sketch.

## Caveats / to do
- Patches 2 and 4 live in App Lab's library cache (`~/.arduino15/internal/tflm_cortexm_1.0.0_*/`) and can be overwritten by a
  reinstall or App Lab update. Before the real model ships, vendor a patched copy of the library into the repo/sketch.
- The transformer's 15 ops are verified on the board (Result 2).
- Arena size to be minimised with the real model.

## Reproduce
1. Mac: `.venv/bin/python ml/board_test/make_smoke_model.py data-local/datasets/sept15.jsonl` → writes `model.h`.
2. App Lab: new app; copy `sketch.ino` + `model.h` into its sketch folder; libraries EloquentTinyML 3.0.1 + tflm_cortexm 1.0.0
   (Arduino_RouterBridge comes with the platform on core ≥ 0.54.1).
3. Apply patches 2 and 4 on the board, build (first build ~15–20 min), run, read the console.
