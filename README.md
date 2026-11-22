# adaptive-baseline-blindness

When does recalibration blind a streaming anomaly detector to persistent shifts, and how do you avoid it?

A persistent shift is both an anomaly and the new normal. The right behaviour is **alert, hold, then re-baseline**. An adaptive baseline can instead absorb the shift before the detector has finished reacting. This repository contains a pre-registered simulation study of that effect, and of a drift-gated update policy that keeps adaptation without the blindness.

**Status** test run complete, preprint in draft

## Claims under test

* **C1 (characterisation).** Recalibration can make a detector miss, or quickly stop flagging, a persistent shift, at matched false-alarm rate.
* **C2 (mechanism).** A simple model predicts when a rolling or periodic baseline blinds the detector, and simulation agrees.
* **C3 (remedy).** A drift-gated update policy restores detection, keeps the benefit of adaptation, and stays within false-alarm tolerance.
* **C4 (real-data illustration).** BTCUSDT 2022 shows the same pattern, reported with its limits.

The hypotheses, metrics, seeds and analysis plan are in [`protocol.md`](protocol.md), which is the authoritative document. If this README and the protocol disagree, the protocol wins.

## Files

| File | Purpose |
|---|---|
| `protocol.md` | Pre-registered hypotheses, metrics, seeds, analysis plan, deviation log |
| `sim.py` | Stream generator, baseline policies, detectors, calibration, metrics, theory |
| `blind_cells.json` | Cells used by the remedy hypotheses, frozen before the test run (not yet created) |
| `FREEZE.md` | Commit hash of the `prereg-v1` tag (not yet created) |

## What `sim.py` contains today

* **Generator:** iid Gaussian and AR(1) noise; level shifts and variance x2; permanent or 200-bar shifts.
* **Policies:** static, rolling, periodic, triggered (ADWIN / Page-Hinkley), gated (one-level), DSPOT-style.
* **Detectors:** z-score, EWMA, CUSUM, Isolation Forest.
* **Calibration:** bisects the threshold to a target alarm-episode rate on a long in-control stream.
* **Theory:** absorption model, CUSUM detection time, periodic-reset prediction.

Still to be added, as listed in the protocol: the btclike (GARCH, Student-t) stream, the benign-drift arm, the two-level gate, the lagged and robust baselines, the scaled variance statistic and the model stream.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy scikit-learn river numba joblib matplotlib pandas
python sim.py --selftest           # should end with ALL PASS
```

`river` is needed for ADWIN in the `triggered` policy; without it a simple Page-Hinkley fallback is used.

## Usage

```bash
python sim.py --selftest
python sim.py --calibrate cusum rolling iid --W 250      # detector, policy, noise
```

## Procedure and freeze

1. Build the remaining simulator pieces and their unit tests.
2. Time one stream per policy and set the compute tiers.
3. Calibrate (300,000 in-control bars per cell) and check on an independent stream.
4. Development runs on seeds 1 to 30: Figure 1, the go/no-go for H1a, `k_var`, the blind-cell list.
5. Tag `prereg-v1` on the protocol and simulator together, and record the hash in `FREEZE.md`.
6. Run the test seeds (101 to 300) once, in tier order.
7. Real-data illustration (Table 1) after the simulation results are fixed.

Test seeds are not touched before the freeze. Any change after it goes in the protocol's deviation log.

## Provenance

The hypotheses were motivated by earlier exploratory results on BTC data. They were not tested confirmatively on those data.

## Data

Real BTC data is not included. The btclike simulation parameters are matched to BTCUSDT 1-minute returns from 2022-01-01 to 2022-06-30, and Table 1 events are taken from 2022-07-01 onward where possible.

## License and citation

To be added.