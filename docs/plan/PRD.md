# PRD — P0: Edge AI Node v2
_Initial draft PRD, copied from `Resume-Service/docs/plan/16-portfolio-projects-roadmap.md` on 2026-11-23.
Refine it here as the project evolves._

## Context
Part of a 3-project portfolio (execution order: P0 Edge AI Node v2 → P1 dMRV Verification Copilot → P2 SkyTriage)
that closes AI/ML resume gaps with public, working code on github.com/codeNick7.

## Environment (checked 2026-11-23)
- Apple M4 Pro, 48 GB unified memory, 20-core GPU; ~490 GB free. Arduino UNO Q available.
- Docker, Ollama, uv, Python 3.12 installed. Use a **Python 3.12** venv (`uv venv --python 3.12`) because TensorFlow support for 3.13+ lags.
- PyTorch on the Mac GPU via MPS; TensorFlow on CPU or via `tensorflow-metal`. No CUDA: rent a GPU (RunPod) for heavier jobs if needed.
- AWS account available; no Azure account yet (explore the free tier).

## Scope and TODO (from roadmap)
**Repo:** extend `KA-Edge-AI-Node` (new `ml/` folder) or a new `KA-Edge-AI-Node-v2`.
**Goal:** re-implement the hand-written transformer in PyTorch and TensorFlow, and run it on the STM32U585 MCU.

- [ ] Re-implement the 1,155-param transformer in PyTorch; train on the same 1.27M readings and chronological 80/20 split
- [ ] Re-implement in TensorFlow/Keras; same data and split
- [ ] Match or beat the baseline (98.9% train / 88.9% test); repeat the leave-one-pot-out check
- [ ] Export to TFLite; int8 post-training quantisation; check the accuracy drop
- [ ] Run on the STM32U585 with TFLite Micro (Arduino sketch); measure latency, RAM and flash
- [ ] Export PyTorch → ONNX; compare with TensorFlow
- [ ] Write-up: hand-written vs PyTorch vs TensorFlow vs TFLite (accuracy, size, latency, effort)
- [ ] GitHub Actions: tests + training smoke test
- [ ] Update `career-master.md`

**Done when:** both framework versions are in the repo, the model runs on the MCU, and the comparison table is in the README.

## Notes
- Repo: cloned from `~/workspace/Contest` (KA-Edge-AI-Node). Remotes: `upstream` = the original GitHub repo (push disabled),
  `local-contest` = the local Contest folder (push disabled). The user will create a new GitHub showcase repo and add it as `origin`.
- The original repo stays untouched. Uncommitted work in `Contest` (an edited architecture doc, demo media) and the 9.7 GB
  `data-from-board/` folder were NOT cloned; read them from `~/workspace/Contest` when needed.
- TFLite Micro on the UNO Q STM32U585 failed in the October 2026 contest work (no written record). Re-test it first:
  libraries or the board's OS/core may have been updated since. Fallback: LiteRT on the QRB2210 Linux side.
