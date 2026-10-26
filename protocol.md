# Protocol: when does recalibration blind a streaming anomaly detector to persistent shifts, and how do you avoid it?

Status: pre-registration draft, revision 3. Everything below is fixed before the test run. Anything changed after the freeze goes in the deviation log at the end.

## 0. Key idea and definitions

A persistent shift is both an anomaly and the new normal. The right behaviour is **alert, hold, then re-baseline**. An adaptive baseline does the last step without the first two, so it can absorb the shift before the detector has finished reacting. The remedy must keep the benefit of adaptation: it must not raise alarms when the data drifts benignly.

* **Alert state.** CUSUM and EWMA accumulators persist: they are not reset on an alarm. The detector is "in alert" at every bar where S > h (CUSUM) or |e| exceeds its limit (EWMA). Only periodic(W, R) resets accumulators, at each recompute. Flags separated by 5 bars or fewer merge into one **alert episode**. A persistent shift therefore gives one long episode, not repeated short ones.
* **First episode.** The first alert episode of an event is the first one that *starts* at or after onset. Flags belonging to an episode already in progress at onset are ignored. The rate of such contaminated events is reported, and a sensitivity analysis excludes them.
* **Blind** = no flag in the horizon, or the first episode lasts fewer than D_min bars. D_min = 10 bars (10 minutes at 1m) is the primary setting; 5 and 30 are sensitivity checks. Because accumulators persist, D_min will rarely bind for CUSUM, so **alert duration is reported directly** and not only through P(blind).
* **Horizon.** H = 500 bars after onset for permanent shifts, and 200 bars (the shift's lifetime) for 200-bar shifts.

## 1. Claims and hypotheses

### C1 (characterisation)

Recalibration can make a detector miss, or quickly stop flagging, a persistent shift, at matched false-alarm rate.

* **H1a (existence, the go/no-go).** For CUSUM on the btclike stream there is at least one cell in the primary family (section 6) where P(detect) under rolling(W) or periodic(W, R = W/4) is below half of static's, with the Holm-adjusted paired test significant at 0.05.
* **H1a-practical (what the paper may claim).** A pass of H1a only at W <= 100 supports only a claim about **toy windows** (50 to 100 minutes), and the paper says so. The practical claim, for windows in the range a pipeline would use (250 to 1,500), requires at least one Holm-significant blind cell at W >= 250 in the primary family. Fallback ladder, fixed now:
  1. blind cell at W >= 250 in the primary family: "practical" claim;
  2. none in the primary family, but a blind cell at W >= 250 at delta = 0.75 (secondary family, Holm over its own 12 comparisons): "practical for small shifts only";
  3. otherwise: "toy windows only".
* **H1b (dependence on W).** P(detect) rises with W. Tested as a trend with a logistic mixed model, logit P(detect) ~ log W + (1 | seed), one-sided test on the slope, per shift type. Jonckheere-Terpstra is a robustness check. (A per-seed Spearman correlation is undefined when all W give the same outcome, so it is not used.)
* **H1c (within the shift's lifetime).** A 200-bar shift and a permanent shift are identical for their first 200 bars, so P(detect within 200) cannot differ between them. The in-lifetime question is therefore tested once, on the permanent runs, truncated at 200 bars: P(blind within 200 bars) and the fraction of the 200 bars spent in alert fall as W falls (same trend test as H1b). The only difference a transient shift can add is what happens after it ends, reported as an exploratory **rebound** metric: alerts raised in the 100 bars after the shift ends, because the partly absorbed baseline now sees the return to normal as a shift.
* **H1d (periodic resets).** periodic(W, R) is worse than rolling(W) in cells where the model-stream detection time D exceeds R.

If H1a fails: no blindness on the corrected grid. Use Fallback A or B and do not stretch claims.

### C2 (mechanism)

A simple model predicts when and whether a rolling or periodic baseline blinds the detector, and the simulation agrees.

* **Model.** s bars after onset a fraction p = min(s/W, 1) of the window is post-shift.
  * *Level shift:* baseline mean p*delta, baseline variance 1 + p(1-p)delta^2. A shifted observation has standardised mean delta(1-p)/sd(p) and scale 1/sd(p), with sd(p) = sqrt(1 + p(1-p)delta^2).
  * *Variance x2:* baseline variance 1 + p, so a shifted observation is Gaussian with scale sqrt(2)/sqrt(1 + p) in baseline units.
* **Model stream.** Run the same detector, with the same k and calibrated h, on a Gaussian stream whose mean and scale follow that path, using the policy's update schedule (for periodic: baseline constant between recomputes, accumulators reset at each recompute, onset phase uniform over R). The Monte Carlo P(detect), P(blind) and median alert duration are the **predictions**. The deterministic detection time D from the noise-free recursion is kept only as an intuition and diagnostic, since it can only give 0 or 1.
* **What the model ignores.** It ignores the estimation noise of the baseline (the baseline mean has sd 1/sqrt(W), about 0.14 at W = 50), which matters at W = 50 to 100. The comparison is made against both arms so gaps can be attributed: model versus iid shows the effect of estimation noise (and discreteness); iid versus btclike shows the effect of non-Gaussian noise. A model stream with estimation noise added is an optional refinement and is exploratory.
* **H2b (agreement).** Predictions are computed and saved before the test run. Pass: across the primary cells, mean absolute error of P(blind) at most 0.10 and Spearman correlation between predicted and simulated P(blind) at least 0.8. The worst cell is reported. Cells with W <= 100 are reported separately because of estimation noise.
* Unit test, not a hypothesis (section 7): the expected window mean moves linearly and reaches 0.9*delta at 0.9W.

### C3 (remedy)

A drift-gated update policy (freeze the baseline for at most K bars once a change is suspected, then re-baseline) restores detection, keeps the benefit of adaptation, has false-alarm rate within tolerance, and costs fewer baseline updates than the comparator.

A one-level gate engages only after the first alarm, so with the same threshold it behaves exactly like rolling until then and cannot change P(detect) (unit-tested). It can only change what follows the alarm: alert duration, coverage and P(blind). So two variants are pre-registered:

* **gated-1L(W, K)**: freeze on an alarm episode start. An ablation, claimed on alert duration, coverage and P(blind), not on P(detect).
* **gated-2L(W, K)** (the primary gate): a more sensitive **watch** level, calibrated to 1 episode per 100 in-control bars, freezes the baseline at its pre-watch values for at most K bars. The alarm level stays at 1 per 1,000, calibrated with the gate active. Early release if the watch level has not flagged for 50 bars. At timeout, set (m, s) from the last K bars. The alarm is scored against the frozen baseline.

**Blind cells (frozen list).** The cells used by H3a and H3g are chosen *before* the test run, so they are not selected on test outcomes. A cell is a blind cell if either the model-stream prediction or the 30 dev seeds shows P(blind | rolling) - P(blind | static) >= 0.25. The list is written to `blind_cells.json` and committed before the test seeds are run. H3a and H3g are also reported separately on each source's list as a sensitivity check.

* **H3a (1L).** In the blind cells, gated-1L closes at least half of the gap in P(blind), with higher alert duration and coverage than rolling.
* **H3g (2L).** In the blind cells, gated-2L closes at least half of the gap in P(blind) and recovers at least 90% of static's P(detect). K = 250 is primary, K = 1000 secondary.
* **H3b (false alarms).** With the gate active, the alarm rate on an independent in-control stream is within +/-20% of 1 per 1,000.
* **H3c (cost of the watch level).** In control: fraction of bars frozen, baseline updates per 1,000 bars, and effect on the alarm rate. A rough estimate (watch episodes at 1 per 100 bars, 50 quiet bars to release) is that the baseline is frozen about half the time, so this cost is expected to be visible and is reported, not assumed away. A watch-rate sensitivity run at 1 per 300 is included.
* **H3d (updates).** Baseline updates per 1,000 bars are reported for every policy. gated recomputes only when not frozen, so it is at most rolling's by construction. periodic recomputes far less often than a per-bar gate, so "equal or lower updates" is claimed against rolling and DSPOT-style only. A gate wrapped around the R = W/4 schedule is reported as the budget-matched comparison.
* **H3e (transients).** After a 200-bar shift ends, the 50-quiet-bar early release returns the gate to rolling without a rebound problem (rebound metric of section 1, exploratory).
* **H3f (benefit of adaptation).** On the benign-drift arm (section 2), with thresholds calibrated on the stationary stream, the primary metric is the **fraction of ramp-window bars in alert**, A_ramp, compared with the same fraction on a stationary in-control stream, A_ctrl (same policy, same bar positions). With persistent accumulators a static detector enters alert and stays there as one merged episode, so episodes per 1,000 bars would look low, not flooded. Alarm episodes per 1,000 bars is kept only as a secondary number. Pass: static has A_ramp at least 5 times A_ctrl (if not, the arm is not a test and is reported as such), and gated has A_ramp at most 2 times max(A_ramp of rolling, A_ctrl). Expected side finding: large-W rolling lags a ramp by about slope*W/2, so its A_ramp may rise with W; this is reported either way.

If H3g fails while H1a holds: report characterisation and mechanism only and drop C3.

### C4 (real-data illustration)

BTCUSDT 2022 shows the same pattern, reported with its limits.

* One illustration: btc_events_check.py --onset for static, rolling, gated-2L, lagged and robust, reporting the first run only (Table 1).
* Real events have no ground-truth onset, so P(detect) and time-to-absorption cannot be measured as in simulation. Table 1 reports flags near each candidate event, alert duration and baseline movement.
* To limit circularity, btclike parameters are matched on 2022-01-01 to 2022-06-30, and Table 1 events are taken from 2022-07-01 onward where possible. If any event comes from the matching period, Table 1 says so.
* Limits stated up front: one asset, one year, onsets chosen from the data, no ground truth, microstructure noise at 1 minute.

## 2. Synthetic data

* **Bars.** 1-minute, n = 20,000 (about 13.9 days), warm-up 3,000, onset uniform in bars 6,000 to 12,000. W in wall-clock terms: 50 = 50 min, 100 = 1.7 h, 250 = 4.2 h, 500 = 8.3 h, 1000 = 16.7 h, 4000 = 2.8 days.
* **btclike (primary).** r_t = sigma_t z_t, sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2, z_t standardised Student-t with unit unconditional variance. Provisional: nu = 5, alpha = 0.05, beta = 0.93, omega = 0.02. The t(5) innovations alone have kurtosis 9 (Var(z^2 - 1) = 8); with these GARCH parameters the marginal kurtosis is about 18 (Var(u^2 - 1) about 17). This is why raw u^2 - 1 is not used (section 3).
* **iid (reference).** Gaussian, needed for the theory check. The ar1 arm is dropped to save time (a revision from the sprint doc).
* **Matching to BTCUSDT 1m.** On robust statistics: the quantile ratios (q99 - q1)/(q75 - q25) and (q99.9 - q0.1)/(q75 - q25) of returns, the autocorrelation of |r| at lags 1 and 60, and the lag-1 autocorrelation of r. Excess kurtosis is reported but is not a matching target (unstable for heavy tails). Parameters change only before the test run and are logged.
* **Shifts.** Level +delta in unconditional sigma, delta in {0.5, 0.75, 1, 2}; variance x2 (x *= sqrt(2)). Permanent, or 200 bars (tier 2 only, for the rebound metric). delta = 0.5 equals k, so CUSUM drifts at zero rate even before absorption and comparison against static there is uninformative.
* **Benign-drift arm.** A slow ramp that should not alert: level rising +2 sigma over 5,000 bars, or variance rising from 1 to 2 over 5,000 bars, with the same onset range, held afterwards. Measured over the 5,000-bar ramp window only. Thresholds stay calibrated on the stationary stream.

## 3. Policies, detectors and statistics

* **Policies.** static; rolling(W); periodic(W, R = W/4); triggered (ADWIN or Page-Hinkley); DSPOT-style; gated-1L and gated-2L(W, K in {250, 1000}); and two comparators:
  * **lagged(W, L)**: window ending L bars ago, L in {250, 1000} to match K. Expected to delay absorption by about L bars.
  * **robust(W)**: rolling median and 1.4826 x MAD. No a priori claim that it avoids blindness: for a level shift the mixture median is halfway at p = 0.5 but reaches 90% of the new level only at p of about 0.89 (delta = 0.75 to 1) and 0.85 (delta = 2), about the same as the mean (0.90). It is reported as a comparator, with its own model-stream prediction.
* **W grid.** {50, 100, 250, 500, 1000, 4000}. W = 50 and 100 are toy windows used to verify the mechanism where theory predicts an effect; the deployment-relevant range is 250 to 1,500. CUSUM blindness at delta = 1 is expected mainly at small W, so the pre-registered delta = 0.75 (long D) probes the larger windows.
* **Detectors.** Rolling z-score; EWMA (lambda = 0.1); CUSUM; Isolation Forest (30-bar features, last 1,000 rows, 1st percentile of out-of-sample scores). Primary: CUSUM.
* **Level shifts.** The statistic is u_t, with CUSUM k = 0.5.
* **Variance shifts.** The raw u^2 - 1 is not used. The statistic is y_t = (|u_t| - a)/b, where a and b are the mean and sd of |u| over the warm-up. A log(u^2 + eps) statistic is a sensitivity check. k and h are in y units. A x2 variance shifts y by only about 0.55 sd under Gaussian noise and less under btclike, so k = 0.5 would sit at or above the shift. k_var = 0.5 x (the measured shift in y), measured on dev seeds with a static baseline and frozen before the test run.
* **Calibration.** Per detector x policy x noise arm, bisect the threshold to 1 alarm episode per 1,000 in-control bars on 300,000 bars with the policy active (for gated-2L: watch level first, then alarm level with the gate active). Episodes count from bar 4,600. Pass: measured rate within 20% of target on the calibration stream and on an independent in-control stream. A block-bootstrap interval (5,000-bar blocks) is reported for each measured rate, since GARCH clustering makes episode counts overdispersed relative to the Poisson noise of about 6% that 300 episodes suggest. The 20% tolerance is expected to cover it.

## 4. Metrics (per event)

| Metric | Definition |
|---|---|
| P(detect) | any flag, from an episode that starts at or after onset, within the horizon |
| Restricted-mean delay | bars to first flag, misses censored at the horizon |
| Alert duration | length of the first alert episode, censored at the horizon (reported directly) |
| Alert coverage | fraction of horizon bars in alert |
| P(blind) | no flag, or first episode shorter than D_min |
| Time-to-absorption | first bar where the baseline mean (level) or scale (variance) has moved 90% of the way to the new level |
| Baseline updates per 1,000 bars | recomputations of (m, s); Isolation Forest refits reported separately |
| Fraction of bars frozen | gated policies |
| Fraction of ramp-window bars in alert | benign-drift arm (primary for H3f); episodes per 1,000 bars secondary |
| Rebound alerts | alerts in the 100 bars after a 200-bar shift ends |

## 5. Seeds, compute and run order

| Purpose | Seeds |
|---|---|
| Calibration stream | 12345 |
| Calibration check (independent in-control stream) | 54321 |
| Development (Figure 1, go/no-go, k_var, blind-cell list) | 1 to 30 |
| Test run (executed once) | 101 to 300 (200 seeds) |
| Declared extension, reported separately if used | 301 to 500 |
| Bootstrap resampling | 2026 |

The same seed gives the same stream for every policy and detector (each stream is generated once and reused), so comparisons are paired. Test seeds are not touched during development.

**Tiers and stop rule.** The full grid is about 290,000 streams per detector, and the gated, robust and DSPOT policies are stateful per-bar loops. So the test run is organised in tiers, run in this order, and if time runs out the lowest tier is dropped, and the paper says so:

* **Tier 1:** the primary family (CUSUM, btclike, permanent shifts, delta = 1 and variance x2, static, rolling and periodic, six W values); gated-2L (K = 250); the benign-drift arm for static, rolling and gated-2L; the iid arm for the theory check; and, for the vectorised policies only, the other shift sizes (0.5, 0.75, 2).
* **Tier 2:** lagged, robust, DSPOT-style, gated-1L, K = 1000, EWMA and z-score, and the 200-bar transient shifts (rebound metric).
* **Tier 3:** Isolation Forest (seeds 101 to 150 only) and triggered.

Before the test run, one stream per policy is timed and the timings are logged; the per-bar policies use numba or joblib. A tier is started only if its projected time fits the remaining compute budget (budget to be recorded at freeze: ________ hours). Claims that depend on a dropped tier are not made.

## 6. Analysis plan

1. **Calibration table** per detector x policy x noise arm. A cell outside +/-20% is flagged and excluded from claims.
2. **Figure 1** (dev seeds): fraction flagging at each bar after onset, static versus rolling(W), for permanent +1 sigma and +2 sigma.
3. **Go/no-go** (30 dev seeds): H1a on dev seeds. Proceed only if it holds. The blind-cell list and k_var are fixed here.
4. **Primary family.** CUSUM, btclike, permanent shifts, delta = 1 and variance x2, rolling and periodic versus static across the six W values: 2 x 6 x 2 = 24 paired comparisons of P(detect). Exact McNemar on the paired detect indicators; effect size with a paired bootstrap 95% interval (10,000 resamples); Holm correction over the 24. The delta = 0.75 secondary family (rolling and periodic versus static, six W: 12 comparisons) has its own Holm correction. All other comparisons are exploratory.
5. **Test run** in tier order. Detection-probability map over W and delta with Wilson 95% intervals.
6. **Theory check (Figure 3):** overlay the model-stream predicted blind region on the simulated map; evaluate H2b, with the gaps attributed as in section 1.
7. **Gating experiments (Figure 4):** Pareto plot of P(blind) against false-alarm rate and baseline updates per 1,000 bars for gated-1L, gated-2L, rolling, periodic, DSPOT-style, lagged and robust, at matched false-alarm rate; the benign-drift arm (H3f) and the in-control freeze cost (H3c).
8. **Equivalence cases.** Report where policies are indistinguishable (paired difference interval within +/-0.05), e.g. large W with large delta, and the earlier benchmark finding that graded, uniform and periodic were indistinguishable.
9. **BTC illustration (Table 1):** descriptive, run after the simulation results are fixed.

## 7. Unit tests (not hypotheses)

* In-control streams have the right mean, variance and autocorrelation; btclike has the target quantile ratios and |r| autocorrelation.
* Baselines are causal (changing future bars does not change earlier (m, s)).
* The expected window mean after onset is linear and reaches 0.9*delta at 0.9W.
* gated-1L and rolling give identical flags up to the first alarm, for every seed, **when both use the same threshold (the one calibrated for rolling)**. The gated-1L threshold calibrated separately may differ and is not used in this test.
* The vectorised and bar-by-bar paths agree.
* Model-stream predictions reproduce the known CUSUM visibility limits (p of about 0.44 at delta = 1 and 0.66 at delta = 2), and the mixture-median absorption points (p of about 0.89 at delta = 1 and 0.85 at delta = 2).
* The first-episode rule ignores an episode in progress at onset.

## 8. Outcomes and what is claimed

| Result | Claim |
|---|---|
| H1a at W >= 250, H2b, H3g and H3f hold | Full paper: C1 to C4, with the practical claim |
| H1a only at W <= 100 (or only delta = 0.75 at W >= 250) | The paper states "toy windows only" (or "small shifts only"); C3 still tested |
| H1a holds, H3g fails | Characterisation and mechanism only; drop C3 and say so |
| H1a holds, H2b fails | Characterisation only; the simple model does not explain it |
| H1a fails on the corrected grid | Fallback A or B; no blindness claim |
| A tier is dropped | Claims depending on it are not made; the paper says so |

## 9. Provenance and freeze

* These hypotheses were motivated by earlier exploratory results on BTC data: that periodic, graded and uniform schedules were indistinguishable, and that the Isolation Forest threshold, re-derived on the freshest quarter, adapted quickly. They were not tested confirmatively on those data.
* Freeze: tag both the protocol and the simulator at the same commit as `prereg-v1`, after the simulator is complete, k_var and the btclike parameters are set, and `blind_cells.json` exists, and before any test seed is run. The commit hash is recorded in `FREEZE.md`, committed after the tag, since a file cannot contain the hash of its own commit.

## 10. Known limits

* The synthetic stream is a model of BTCUSDT 1m, not the data. Shifts are injected, so onsets are known by construction.
* The theory is Gaussian and ignores baseline estimation noise; the iid and btclike arms attribute the gaps.
* delta is in units of unconditional sigma; under volatility clustering the local sigma varies with regime at onset.
* Isolation Forest adapts through its own threshold re-derivation, a different mechanism from the baseline policies, so it is reported but not fitted to the model.
* D_min = 10, the btclike parameters, and the choice of gated-2L as the primary gate are my choices, to be confirmed.

## Revision log

* Rev 3 (pre-freeze): H3f metric changed to fraction of ramp-window bars in alert; H1c replaced (the transient-versus-permanent comparison is vacuous within the lifetime) by an in-lifetime test plus an exploratory rebound metric; blind-cell list frozen from model-stream and dev seeds before the test run; H1a-practical added with a fallback ladder; gated-1L unit test pinned to the rolling threshold; robust-baseline claim replaced by the computed mixture-median result; model stream extended to variance shifts and its estimation-noise gap stated; compute tiers and a stop rule added; first-episode rule, alert duration reporting and calibration block-bootstrap added; H1b changed to a logistic mixed model.
* Rev 2 (pre-freeze): two-level gate, benign-drift arm, CUSUM alert state, scaled variance statistic, probabilistic model stream, 200 test seeds and 30 dev seeds, 24-comparison primary family, W = 500 and delta = 0.75, lagged and robust comparators, ar1 dropped, trend test for H1b, updates per 1,000 bars, robust matching.

## Deviation log

(empty)