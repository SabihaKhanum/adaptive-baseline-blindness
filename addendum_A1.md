# Addendum A1 to the protocol: two more detectors

Status: written before the test seeds (101 to 300) are run for these detectors. Commit this file and tag it
`prereg-v1-addendum-a1` BEFORE running `extra_detectors.py run` on seeds 101-300. 
Date and commit hash: Date: 2026-11-24. Commit: see tag prereg-v1-addendum-a1.

## What is added

Two detectors besides CUSUM, run under the same baseline policies (static, rolling(W), periodic(W, R = W/4)) on the
same btclike stream (rev-6 generator), with the same shifts (permanent level shift, delta = 1 primary, 0.75 secondary), the same
horizon (H = 500), the same episode rule (gap 5), the same calibration rule (1 alarm episode per 1,000 in-control bars
per detector, policy and W, on calibration seeds 12345 and check seeds 54321), and the same metrics. Code: `extra_detectors.py`
(its CUSUM branch is checked against `sim.py` by `extra_detectors.py test`).

* **ewma**: z_t = 0.1 u_t + 0.9 z_{t-1}, alarm when |z_t| > h; z is reset at each periodic recompute.
* **ma20**: moving-average z-score, sqrt(20) x |mean of the last 20 values of u|, alarm when above h.

u = (x - m)/s with (m, s) the baseline of the policy in force. The gated policies are not run (the gate was designed and
calibrated for CUSUM). The parameters lambda = 0.1 and L = 20 are fixed here and are not tuned.

## Hypotheses

* **H4a (shortened alert, per detector).** For each of ewma and ma20, in the family of 12 cells (delta = 1; rolling and
  periodic; W in {50, 100, 250, 500, 1000, 4000}), at least one cell has alert coverage at most 0.5 times static's, with a
  one-sided paired Wilcoxon (policy < static, ties split) Holm-adjusted over the 12 cells of that detector at 0.05
  (`sim.shortened_table`). The delta = 0.75 family (12 cells, own Holm) is secondary.
* **H4b (practical windows).** Same, at W >= 250 (the same fallback ladder as H1a-practical).
* **H4c (detection).** P(detect) of each policy against static, McNemar, Holm over the 12 cells, reported; non-inferiority is
  not claimed.

Outcomes are reported whatever they are. If H4a fails for a detector, the paper says the effect was not found for that detector.

## Disclosure of what was seen before this addendum

The code was run on development seeds 1 to 30 to check that it works. That run also produced results (ewma and ma20 show
shortened cells at small W on development data). One detector was replaced after that run: a **single-point threshold
(Shewhart, alarm when |u| > h)** was first included, and was dropped because at 1 alarm per 1,000 bars its static coverage is
about 0.3% (it cannot see a one-sigma shift at all), so coverage ratios against static are not informative; ma20 replaced it.
The parameters of ewma and ma20 were chosen before any run and were not changed after.

## Not included

DSPOT / extreme-value thresholds, ADWIN and other drift detectors, the variance statistic, transient shifts, and gated
policies for these detectors. These are limitations, not results.

## Seeds, compute, commands

Dev seeds 1-30 (done as a code check); test seeds 101-300 (run once). About 10 seconds per detector for calibration and
under a minute for the test run on one core.

    python extra_detectors.py test
    python extra_detectors.py calibrate --out cal_extra.json
    python extra_detectors.py run --cal cal_extra.json --seeds 101-300 --out extra_test