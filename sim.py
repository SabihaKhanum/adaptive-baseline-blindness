"""
sim.py -- Tier 1 simulator for "when does recalibration blind a streaming detector?"
Implements protocol revision 4 (protocol.md): generators, baseline policies, CUSUM, matched-false-alarm
calibration, gated-1L / gated-2L, per-event metrics, the dev-seed go/no-go, the benign-drift (H3f) runner,
the model stream (H2b), blind-cell list, trend tests and gate evaluation.

    python sim.py test                                         # unit tests (protocol section 7)
    python sim.py kvar --arm btclike --seeds 1-30              # k for the variance statistic (frozen before test run)
    python sim.py calibrate --arm btclike --out cal_btclike.json
    python sim.py calibrate --arm iid --out cal_iid.json
    python sim.py dev --arm btclike --cal cal_btclike.json --seeds 1-30 --out dev_out
    python sim.py figure1 --arm btclike --cal cal_btclike.json --seeds 1-30 --out dev_out
    python sim.py ramp --arm btclike --cal cal_btclike.json --seeds 1-30 --out dev_out         # H3f
    python sim.py model --cal cal_iid.json --out model_pred_iid.json                            # H2b predictions
    python sim.py blindcells --events dev_out/dev_events.csv --model model_pred_iid.json --out blind_cells.json
    python sim.py compare-model --events dev_out/dev_events.csv --model model_pred_iid.json     # H2b check
    python sim.py trend --events dev_out/dev_events.csv                                         # H1b / H1c
    python sim.py gate-eval --events dev_out/dev_events.csv --cells blind_cells.json            # H3a / H3g

Tier 1 = static, rolling(W), periodic(W, R=W/4), gated-1L(W, K), gated-2L(W, K); CUSUM (level + variance
statistics); iid and btclike (GARCH(1,1)-t) noise; level / variance shifts; benign-drift ramps.
Tier 2 (lagged, robust, DSPOT-style, EWMA, z-score) and Tier 3 (Isolation Forest, triggered) are NOT here.

Implementation choices (any change goes in the protocol BEFORE the test run):
  * Baseline for bar t uses bars < t only (causal), via prefix sums; std is the biased window std.
    At t == WARM every policy starts from the whole warm-up window (irrelevant for onsets >= 6,000).
  * Static baseline and the |u| scaling (a, b) come from the warm-up bars [0, WARM).
  * Gate: on trigger the baseline is frozen from the NEXT bar; it is released early after QUIET quiet bars
    (statistic <= the gate's own level), or at timeout (K bars) when the baseline is re-anchored to the last K
    bars (window start = max(t - W, t_timeout - K)) and the accumulators reset.
  * Calibration uses M (default 120) independent 20,000-bar in-control streams (each with its own warm-up) instead of
    one 300,000-bar stream, so the static baseline's sampling variability is represented (protocol section 3).
    The in-control fraction of bars IN ALERT is reported next to the episode rate, because under volatility
    clustering the episodes can be very long.
  * Alarm episode: flags <= GAP bars apart merge. Episodes are counted from bar EP_START.
  * First episode of an event = first episode that STARTS at or after onset; flags chained to an episode already in
    progress at onset are skipped (and counted in `contaminated`). `detect_raw` keeps the old any-flag definition.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats as sps
from scipy.optimize import brentq

# --- constants (protocol sections 2/3/5) ------------------------------------------------------------------------
N = 20_000
WARM = 3_000
EP_START = 4_600
ONSET_LO, ONSET_HI = 6_000, 12_000          # onset uniform on the integers 6,000 .. 12,000 inclusive
H = 500
H_LIFE = 200            # in-lifetime horizon (H1c) and transient length
REBOUND = 100           # bars after a 200-bar shift ends over which rebound alerts are counted
GAP = 5
DMIN = 10
QUIET = 50
OMEGA, ALPHA, BETA, NU = 0.02, 0.05, 0.93, 5
BURN = 500
K_LEVEL = 0.5
WGRID = [50, 100, 250, 500, 1000, 4000]
POLS = {"static": 0, "rolling": 1, "periodic": 2, "gated1L": 3, "gated2L": 4}
CAL_SEED, CHECK_SEED, M_CAL = 12345, 54321, 120
TARGET_ALARM = 1.0     # episodes per 1,000 bars
TARGET_WATCH = 10.0    # 1 per 100 bars
ALERT_FLAG = 0.05      # in-control in-alert fraction above which a cell is flagged "long episodes"
RAMP_LEN = 5_000
MODEL_PRE = 300
INF = 1e18


# --- noise and streams -------------------------------------------------------------------------------------------
@njit(cache=True)
def _garch(z, omega, alpha, beta):
    n = len(z)
    r = np.empty(n)
    s2 = 1.0
    for t in range(n):
        r[t] = math.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] * r[t] + beta * s2
    return r


def noise(seed, arm, n=N):
    rng = np.random.default_rng(seed)
    if arm == "iid":
        return rng.standard_normal(n)
    if arm == "btclike":
        z = rng.standard_t(NU, n + BURN) * math.sqrt((NU - 2) / NU)
        return _garch(z, OMEGA, ALPHA, BETA)[BURN:]
    raise ValueError(arm)


def make_stream(seed, arm, shift=("none", None), persist=None, n=N):
    """Returns x, tau, end. shift = ("level", delta) | ("var", None) | ("ramp_level", delta) | ("ramp_var", None).
    The base noise depends only on (seed, arm); the onset only on seed, so every policy sees the same stream."""
    base = noise(seed, arm, n)
    tau = int(np.random.default_rng([seed, 1]).integers(ONSET_LO, ONSET_HI + 1))
    end = n if persist is None else min(n, tau + persist)
    x = base.copy()
    kind, delta = shift
    if kind == "level":
        x[tau:end] += delta
    elif kind == "var":
        x[tau:end] *= math.sqrt(2.0)
    elif kind in ("ramp_level", "ramp_var"):
        ramp = np.clip((np.arange(n) - tau) / float(RAMP_LEN), 0.0, 1.0)
        x = x + delta * ramp if kind == "ramp_level" else x * np.sqrt(1.0 + ramp)
    elif kind != "none":
        raise ValueError(kind)
    return x, tau, end


# --- detector + baseline policies (one per-bar loop) -------------------------------------------------------------
@njit(cache=True)
def _wstats(cs1, cs2, a, b):
    n = b - a
    mean = (cs1[b] - cs1[a]) / n
    var = (cs2[b] - cs2[a]) / n - mean * mean
    return mean, math.sqrt(max(var, 1e-12))


@njit(cache=True)
def run_stream(x, pol, W, R, K, k, h_a, h_w, mode, warm, alarm, watch, m_out, s_out):
    """pol: 0 static, 1 rolling, 2 periodic, 3 gated-1L, 4 gated-2L.  mode: 0 level (two-sided CUSUM on u),
    1 variance (one-sided CUSUM on y = (|u| - a) / b).  Returns (baseline updates, frozen bars)."""
    n = len(x)
    cs1 = np.zeros(n + 1)
    cs2 = np.zeros(n + 1)
    for i in range(n):
        cs1[i + 1] = cs1[i] + x[i]
        cs2[i + 1] = cs2[i] + x[i] * x[i]
    m = 0.0
    s = 1.0
    ya = 0.0
    yb = 1.0
    Sp = 0.0
    Sm = 0.0
    frozen = False
    frozen_until = 0
    quiet = 0
    anchor = 0
    updates = 0
    frozen_bars = 0
    for t in range(n):
        if t < warm:
            m_out[t] = 0.0
            s_out[t] = 1.0
            continue
        if t == warm:
            m, s = _wstats(cs1, cs2, 0, warm)
            sa = 0.0
            sa2 = 0.0
            for i in range(warm):
                v = abs(x[i] - m) / s
                sa += v
                sa2 += v * v
            ya = sa / warm
            yb = math.sqrt(max(sa2 / warm - ya * ya, 1e-12))
            updates = 1
        else:
            if pol == 1:
                a0 = max(t - W, 0)
                m, s = _wstats(cs1, cs2, a0, t)
                updates += 1
            elif pol == 2:
                if t % R == 0:
                    m, s = _wstats(cs1, cs2, max(t - W, 0), t)
                    Sp = 0.0
                    Sm = 0.0
                    updates += 1
            elif pol >= 3:
                if frozen:
                    if t >= frozen_until:
                        anchor = t - K
                        m, s = _wstats(cs1, cs2, max(t - W, anchor), t)
                        Sp = 0.0
                        Sm = 0.0
                        frozen = False
                        quiet = 0
                        updates += 1
                    else:
                        frozen_bars += 1
                else:
                    m, s = _wstats(cs1, cs2, max(t - W, anchor), t)
                    updates += 1
        m_out[t] = m
        s_out[t] = s
        u = (x[t] - m) / s
        if mode == 0:
            Sp = max(0.0, Sp + u - k)
            Sm = max(0.0, Sm - u - k)
            stat = max(Sp, Sm)
        else:
            y = (abs(u) - ya) / yb
            Sp = max(0.0, Sp + y - k)
            stat = Sp
        alarm[t] = 1 if stat > h_a else 0
        watch[t] = 1 if stat > h_w else 0
        if pol >= 3:
            hq = h_a if pol == 3 else h_w
            if frozen:
                if stat <= hq:
                    quiet += 1
                else:
                    quiet = 0
                if quiet >= QUIET:
                    frozen = False
                    quiet = 0
            elif stat > hq:
                frozen = True
                frozen_until = t + 1 + K
                quiet = 0
    return updates, frozen_bars


@njit(cache=True)
def count_eps(flag, start, gap):
    last = -1000000000
    c = 0
    for t in range(len(flag)):
        if flag[t] != 0:
            if t >= start and t - last > gap:
                c += 1
            last = t
    return c


@njit(cache=True)
def first_episode(alarm, tau, hz, gap):
    """First alert episode that STARTS at or after tau and within [tau, tau + hz).
    Flags chained (gap <= `gap`) to a flag just before tau belong to an episode already in progress and are skipped.
    Duration is censored at the horizon. Returns (detected, first, duration, contaminated); when nothing is
    detected `first` = hz and duration = 0."""
    prev = -1000000000
    lo = max(0, tau - gap)
    for i in range(tau - 1, lo - 1, -1):
        if alarm[i] != 0:
            prev = i
            break
    end = min(tau + hz, len(alarm))
    contaminated = 0
    t = tau
    start = -1
    while t < end:
        if alarm[t] != 0:
            if t - prev <= gap:
                contaminated = 1
                prev = t
            else:
                start = t
                break
        t += 1
    if start < 0:
        return 0, hz, 0, contaminated
    last = start
    u = start + 1
    while u < end:
        if u - last > gap:
            break
        if alarm[u] != 0:
            last = u
        u += 1
    return 1, start - tau, last - start + 1, contaminated


def run(x, pol, W, K, k, h_a, h_w, mode, warm=WARM):
    n = len(x)
    alarm = np.zeros(n, np.uint8)
    watch = np.zeros(n, np.uint8)
    m = np.zeros(n)
    s = np.ones(n)
    R = max(1, W // 4)
    upd, fz = run_stream(x, POLS[pol], int(W), R, int(K), float(k), float(h_a), float(h_w), int(mode),
                         int(warm), alarm, watch, m, s)
    return alarm, watch, m, s, upd, fz


# --- calibration (matched false-alarm rate, policy active) -------------------------------------------------------
def cal_streams(arm, base, M=M_CAL):
    return [noise(base + i, arm) for i in range(M)]


def rate(xs, pol, W, K, k, h_a, h_w, mode, which):
    tot, bars = 0, 0
    for x in xs:
        alarm, watch, *_ = run(x, pol, W, K, k, h_a, h_w, mode)
        tot += count_eps(watch if which == "watch" else alarm, EP_START, GAP)
        bars += len(x) - EP_START
    return 1000.0 * tot / bars


def rate_se(xs, pol, W, K, k, h_a, h_w, mode):
    """Standard error of the in-control episode rate, clustered by stream (the baseline's warm-up error is shared
    by all bars of a stream, so the number of independent units is the number of streams, not of bars)."""
    per = []
    for x in xs:
        alarm, *_ = run(x, pol, W, K, k, h_a, h_w, mode)
        per.append(1000.0 * count_eps(alarm, EP_START, GAP) / (len(x) - EP_START))
    return float(np.std(per, ddof=1) / math.sqrt(len(per)))


def alert_fraction(xs, pol, W, K, k, h_a, h_w, mode):
    """In-control fraction of scored bars that are IN ALERT (statistic above the alarm threshold)."""
    tot, bars = 0, 0
    for x in xs:
        alarm, *_ = run(x, pol, W, K, k, h_a, h_w, mode)
        tot += int(alarm[EP_START:].sum())
        bars += len(x) - EP_START
    return tot / bars


def control_cost(xs, pol, W, K, k, h_a, h_w, mode):
    """In-control cost: baseline updates per 1,000 scored bars and fraction of bars the baseline is frozen."""
    upd = fz = bars = 0
    for x in xs:
        *_, u, f = run(x, pol, W, K, k, h_a, h_w, mode)
        upd += u
        fz += f
        bars += len(x) - WARM
    return 1000.0 * upd / bars, fz / bars


def bisect_threshold(f, target, lo=0.3, hi=3000.0, iters=40):
    """f(h) = episodes per 1,000 bars, decreasing in h. Returns the smallest h found with f(h) <= target."""
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def calibrate_cell(xs_cal, xs_chk, pol, W, K, k, mode):
    out = {"pol": pol, "W": W, "K": K, "mode": mode, "k": k}
    if pol == "gated2L":
        h_w = bisect_threshold(lambda h: rate(xs_cal, pol, W, K, k, INF, h, mode, "watch"), TARGET_WATCH)
        h_a = bisect_threshold(lambda h: rate(xs_cal, pol, W, K, k, h, h_w, mode, "alarm"), TARGET_ALARM)
        out.update(h_a=h_a, h_w=h_w,
                   watch_rate_cal=rate(xs_cal, pol, W, K, k, INF, h_w, mode, "watch"),
                   watch_rate_chk=rate(xs_chk, pol, W, K, k, INF, h_w, mode, "watch"))
    else:
        h_a = bisect_threshold(lambda h: rate(xs_cal, pol, W, K, k, h, INF, mode, "alarm"), TARGET_ALARM)
        out.update(h_a=h_a, h_w=INF)
    out["rate_cal"] = rate(xs_cal, pol, W, K, k, out["h_a"], out["h_w"], mode, "alarm")
    out["rate_chk"] = rate(xs_chk, pol, W, K, k, out["h_a"], out["h_w"], mode, "alarm")
    out["rate_chk_se"] = rate_se(xs_chk, pol, W, K, k, out["h_a"], out["h_w"], mode)
    out["alert_frac_cal"] = alert_fraction(xs_cal, pol, W, K, k, out["h_a"], out["h_w"], mode)
    out["alert_frac_chk"] = alert_fraction(xs_chk, pol, W, K, k, out["h_a"], out["h_w"], mode)
    out["long_episodes"] = bool(out["alert_frac_chk"] > ALERT_FLAG)
    out["upd_per_1000_chk"], out["frozen_frac_chk"] = control_cost(xs_chk, pol, W, K, k, out["h_a"], out["h_w"], mode)
    out["ok"] = bool(0.8 * TARGET_ALARM <= out["rate_cal"] <= 1.2 * TARGET_ALARM and
                     0.8 * TARGET_ALARM <= out["rate_chk"] <= 1.2 * TARGET_ALARM)
    return out


def cal_key(mode, pol, W, K):
    return f"{mode}|{pol}|{W}|{K}"


def calibrate_all(arm, k_var, Ks=(250, 1000), Ws=WGRID, M=M_CAL, verbose=True):
    xs_cal = cal_streams(arm, CAL_SEED, M)
    xs_chk = cal_streams(arm, CHECK_SEED, M)
    cells = {}
    for mode, k in ((0, K_LEVEL), (1, k_var)):
        jobs = [("static", 0, 0)]
        jobs += [(p, W, 0) for p in ("rolling", "periodic") for W in Ws]
        jobs += [(p, W, K) for p in ("gated1L", "gated2L") for W in Ws for K in Ks]
        for pol, W, K in jobs:
            c = calibrate_cell(xs_cal, xs_chk, pol, W, K, k, mode)
            cells[cal_key(mode, pol, W, K)] = c
            if verbose:
                print(f"  mode={mode} {pol:<8} W={W:<5} K={K:<5} h_a={c['h_a']:8.3f} "
                      f"rate cal/chk = {c['rate_cal']:.2f}/{c['rate_chk']:.2f} (se {c['rate_chk_se']:.2f}) "
                      f"in-alert {100 * c['alert_frac_chk']:5.1f}% "
                      f"{'ok' if c['ok'] else 'OUT OF TOLERANCE'}{' LONG-EPISODES' if c['long_episodes'] else ''}",
                      flush=True)
    return {"arm": arm, "k_var": k_var, "k_level": K_LEVEL, "cells": cells}


# --- metrics -----------------------------------------------------------------------------------------------------
def event_metrics(alarm, m, s, tau, end, shift, persist, H_=H, dmin=DMIN, gap=GAP):
    hz = H_ if persist is None else min(H_, end - tau)
    det, first, dur, contam = first_episode(alarm, tau, hz, gap)
    h2 = min(H_LIFE, hz)
    det2, _, dur2, _ = first_episode(alarm, tau, h2, gap)
    kind, delta = shift
    tta = np.nan
    if persist is None and kind in ("level", "var") and tau > 0:
        if kind == "level":
            ref = m[tau - 1]
            hit = np.flatnonzero(m[tau:] - ref >= 0.9 * delta)
        else:
            ref = s[tau - 1]
            hit = np.flatnonzero(s[tau:] >= ref + 0.9 * (math.sqrt(2.0) - 1.0) * ref)
        if hit.size:
            tta = float(hit[0])
    rebound = np.nan
    if persist is not None and end + REBOUND <= len(alarm):
        idx = np.flatnonzero(alarm[:end + REBOUND])
        if idx.size:
            starts = idx[np.r_[True, np.diff(idx) > gap]]
            rebound = float(((starts >= end) & (starts < end + REBOUND)).sum())
        else:
            rebound = 0.0
    return {"horizon": hz, "detect": int(det), "detect_raw": int(alarm[tau:tau + hz].any()), "delay": int(first),
            "duration": int(dur), "contaminated": int(contam), "coverage": float(alarm[tau:tau + hz].mean()),
            "blind": int((not det) or dur < dmin), "blind200": int((not det2) or dur2 < dmin),
            "alert200": float(alarm[tau:tau + h2].mean()), "tta": tta, "rebound": rebound}


def measure_k_var(arm, seeds):
    """k for the variance statistic: 0.5 x the measured shift in y under a x2 variance shift, static baseline."""
    shifts = []
    for seed in seeds:
        x, tau, _ = make_stream(seed, arm, ("var", None))
        m0, s0 = x[:WARM].mean(), x[:WARM].std()
        v = np.abs(x - m0) / s0
        ya, yb = v[:WARM].mean(), v[:WARM].std()
        y = (v - ya) / yb
        shifts.append(y[tau:tau + 500].mean() - y[WARM:tau].mean())
    return 0.5 * float(np.mean(shifts)), float(np.mean(shifts))


# --- statistics helpers ------------------------------------------------------------------------------------------
def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_compare(a, b, n_boot=10_000, seed=2026):
    """a, b: 0/1 arrays over the same seeds. Returns diff (a - b), bootstrap 95% CI, exact McNemar p."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    p = 1.0 if n01 + n10 == 0 else float(sps.binomtest(n01, n01 + n10, 0.5).pvalue)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), p


def trend_test(df, shift, pol, col, direction=+1, n_perm=10_000, seed=2026):
    """Does `col` increase with log W (direction=+1) or decrease (direction=-1)?
    Within-seed permutation test on sum_seeds sum_W (log W - mean log W) * y (well defined even when a seed's
    outcomes are all equal), one-sided; plus a GEE logistic slope (exchangeable, clustered by seed) when
    statsmodels is installed (protocol H1b: stand-in for the seed random effect)."""
    g = df[(df["shift"] == shift) & (df["pol"] == pol)]
    Y = g.pivot_table(index="seed", columns="W", values=col).dropna(axis=0)
    if Y.shape[0] < 2 or Y.shape[1] < 3:
        return dict(shift=shift, pol=pol, metric=col, direction=direction, p_perm=np.nan, slope_gee=np.nan, p_gee=np.nan)
    Ws = np.array(Y.columns, float)
    c = np.log(Ws) - np.log(Ws).mean()
    Yv = Y.values.astype(float)
    obs = direction * float((Yv * c).sum())
    rng = np.random.default_rng(seed)
    cnt = sum(direction * float((rng.permuted(Yv, axis=1) * c).sum()) >= obs - 1e-12 for _ in range(n_perm))
    slope = p_gee = float("nan")
    try:
        import warnings
        import statsmodels.api as sm
        warnings.simplefilter("ignore")
        long = g[["seed", "W", col]].copy()
        long["logW"] = np.log(long["W"].astype(float))
        fit = sm.GEE(long[col].astype(float), sm.add_constant(long["logW"]), groups=long["seed"],
                     family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        slope = float(fit.params["logW"])
        p_gee = float(sps.norm.sf(direction * float(fit.tvalues["logW"])))
    except Exception:
        pass
    return dict(shift=shift, pol=pol, metric=col, direction=direction, p_perm=(cnt + 1) / (n_perm + 1),
                slope_gee=slope, p_gee=p_gee)


# --- dev / go-no-go runner ---------------------------------------------------------------------------------------
def parse_seeds(txt):
    if "-" in txt and "," not in txt:
        a, b = txt.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in txt.split(",")]


def dev_run(arm, seeds, cal, shifts, pols, Ws, Ks, persist=None):
    rows = []
    for seed in seeds:
        for shift in shifts:
            x, tau, end = make_stream(seed, arm, shift, persist=persist)
            mode = 0 if shift[0] in ("level", "ramp_level") else 1
            k = cal["k_level"] if mode == 0 else cal["k_var"]
            for pol in pols:
                for W in ([0] if pol == "static" else Ws):
                    for K in (Ks if pol.startswith("gated") else [0]):
                        c = cal["cells"][cal_key(mode, pol, W, K)]
                        alarm, watch, m, s, upd, fz = run(x, pol, W, K, k, c["h_a"], c["h_w"], mode)
                        met = event_metrics(alarm, m, s, tau, end, shift, persist)
                        rows.append(dict(seed=seed, shift=f"{shift[0]}{'' if shift[1] is None else shift[1]}",
                                         pol=pol, W=W, K=K, tau=tau,
                                         upd_per_1000=1000.0 * upd / (len(x) - WARM),
                                         frozen_frac=fz / (len(x) - WARM), **met))
    return pd.DataFrame(rows)


def go_no_go(df, family_shifts, excl_contam=False, col="detect"):
    """H1a on dev seeds: cells (shift, W, pol in rolling/periodic) with P(detect) < 0.5 x static and a Holm-adjusted
    exact McNemar p < 0.05 over the family. excl_contam=True drops events contaminated for either policy."""
    recs = []
    for shift in family_shifts:
        stf = df[(df["shift"] == shift) & (df["pol"] == "static")].set_index("seed")
        for pol in ("rolling", "periodic"):
            for W in sorted(df["W"].unique()):
                if W == 0:
                    continue
                gf = df[(df["shift"] == shift) & (df["pol"] == pol) & (df["W"] == W)].set_index("seed")
                common = stf.index.intersection(gf.index)
                if excl_contam:
                    common = common[(stf.loc[common, "contaminated"] == 0).values & (gf.loc[common, "contaminated"] == 0).values]
                if len(common) == 0:
                    continue
                st, g = stf.loc[common, col], gf.loc[common, col]
                d, lo, hi, p = paired_compare(g.values, st.values)
                recs.append(dict(shift=shift, pol=pol, W=W, n=len(common), P_policy=g.mean(), P_static=st.mean(),
                                 diff=d, ci_lo=lo, ci_hi=hi, p_mcnemar=p))
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_mcnemar"].values)
    out["h1a_cell"] = (out["P_policy"] < 0.5 * out["P_static"]) & (out["p_holm"] < 0.05)
    return out


def equivalence_table(df, tol=0.05):
    """Where are policies indistinguishable? Paired difference with a bootstrap 95% CI inside +/- tol."""
    pairs = [("rolling", "periodic"), ("gated2L", "rolling"), ("gated1L", "rolling"), ("gated2L", "static"),
             ("rolling", "static"), ("periodic", "static")]
    recs = []
    for shift in sorted(df["shift"].unique()):
        for metric in ("detect", "blind"):
            for pa, pb in pairs:
                for W in sorted(df["W"].unique()):
                    for K in (sorted(df[df["pol"].str.startswith("gated")]["K"].unique()) if pa.startswith("gated") else [0]):
                        wa = W if pa != "static" else 0
                        wb = W if pb != "static" else 0
                        if W == 0:
                            continue
                        a = df[(df["shift"] == shift) & (df["pol"] == pa) & (df["W"] == wa) & (df["K"] == K)].set_index("seed")[metric]
                        b = df[(df["shift"] == shift) & (df["pol"] == pb) & (df["W"] == wb) & (df["K"] == 0)].set_index("seed")[metric]
                        common = a.index.intersection(b.index)
                        if len(common) < 5:
                            continue
                        d, lo, hi, _ = paired_compare(a.loc[common].values, b.loc[common].values)
                        recs.append(dict(shift=shift, metric=metric, a=pa, b=pb, K=K, W=W, diff=d, ci_lo=lo, ci_hi=hi,
                                         equivalent=bool(lo > -tol and hi < tol)))
    return pd.DataFrame(recs)


def summarise(df):
    return df.groupby(["shift", "pol", "W", "K"]).agg(
        n=("detect", "size"), P_detect=("detect", "mean"), P_detect_raw=("detect_raw", "mean"),
        contam=("contaminated", "mean"), P_blind=("blind", "mean"), P_blind200=("blind200", "mean"),
        delay=("delay", "mean"), duration=("duration", "mean"), coverage=("coverage", "mean"),
        alert200=("alert200", "mean"), tta=("tta", "median"), rebound=("rebound", "mean"),
        upd_per_1000=("upd_per_1000", "mean"), frozen_frac=("frozen_frac", "mean")).reset_index()


def cmd_dev(a):
    cal = json.load(open(a.cal))
    seeds = parse_seeds(a.seeds)
    shifts = [("level", 1.0), ("level", 2.0), ("var", None)] + [("level", d) for d in a.extra_deltas]
    pols = ["static", "rolling", "periodic", "gated1L", "gated2L"]
    df = dev_run(a.arm, seeds, cal, shifts, pols, WGRID, a.ks, persist=a.persist)
    os.makedirs(a.out, exist_ok=True)
    tag = "" if a.persist is None else f"_persist{a.persist}"
    df.to_csv(os.path.join(a.out, f"dev_events{tag}.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(a.out, f"dev_summary{tag}.csv"), index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(summ.round(3).to_string(index=False))
    print("\nEvents with an alert already in progress at onset (flags chained to it are skipped):")
    print(df.groupby(["shift", "pol"])["contaminated"].mean().round(2).to_string())
    if a.persist is None:
        for name, fam in (("primary", ["level1.0", "var"]), ("secondary (delta = 0.75)", ["level0.75"])):
            if not set(fam) & set(df["shift"]):
                continue
            res = go_no_go(df, [f for f in fam if f in set(df["shift"])])
            res.to_csv(os.path.join(a.out, f"go_no_go_{name.split()[0]}.csv"), index=False)
            exc = go_no_go(df, [f for f in fam if f in set(df["shift"])], excl_contam=True)
            exc.to_csv(os.path.join(a.out, f"go_no_go_{name.split()[0]}_excl_contam.csv"), index=False)
            print(f"\nH1a family: {name} (rolling/periodic vs static, P(detect), Holm over {len(res)})")
            print(res.round(3).to_string(index=False))
            n_cells = int(res["h1a_cell"].sum())
            practical = res[res["h1a_cell"] & (res["W"] >= 250)]
            print(f"H1a [{name}]: {'PASS' if n_cells else 'FAIL'}  ({n_cells} cells; {len(practical)} with W >= 250; "
                  f"{int(exc['h1a_cell'].sum())} when contaminated events are excluded)")
        eq = equivalence_table(df)
        eq.to_csv(os.path.join(a.out, "equivalence.csv"), index=False)
        print(f"\nequivalence.csv written ({int(eq['equivalent'].sum())} of {len(eq)} comparisons inside +/-0.05).")


def cmd_figure1(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cal = json.load(open(a.cal))
    seeds = parse_seeds(a.seeds)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, delta in zip(axes, (1.0, 2.0)):
        for pol, W in (("static", 0), ("rolling", 250), ("rolling", 1000), ("rolling", 4000)):
            frac = np.zeros(1000)
            for seed in seeds:
                x, tau, end = make_stream(seed, a.arm, ("level", delta))
                c = cal["cells"][cal_key(0, pol, W, 0)]
                alarm, *_ = run(x, pol, W, 0, cal["k_level"], c["h_a"], c["h_w"], 0)
                frac += alarm[tau:tau + 1000]
            ax.plot(frac / len(seeds), label="static" if pol == "static" else f"rolling W={W}")
        ax.set_title(f"+{delta:g} sigma level shift ({a.arm})")
        ax.set_xlabel("bars after onset")
    axes[0].set_ylabel("fraction of streams in alert")
    axes[0].legend()
    os.makedirs(a.out, exist_ok=True)
    fig.tight_layout()
    path = os.path.join(a.out, "figure1.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


# --- H3f: benign-drift ramp --------------------------------------------------------------------------------------
RAMP_KINDS = (("ramp_level", 2.0, 0), ("ramp_var", None, 1))


def ramp_run(arm, seeds, cal, pols=("static", "rolling", "periodic", "gated2L"), Ws=WGRID, K=250, kinds=RAMP_KINDS):
    """Fraction of ramp-window bars in alert (primary), against the same fraction on the stationary stream with
    the same seed and the same bar positions; alarm episodes per 1,000 bars (secondary)."""
    rows = []
    for seed in seeds:
        for kind, delta, mode in kinds:
            xr, tau, _ = make_stream(seed, arm, (kind, delta))
            xc, _, _ = make_stream(seed, arm, ("none", None))
            k = cal["k_level"] if mode == 0 else cal["k_var"]
            lo, hi = tau, tau + RAMP_LEN
            for pol in pols:
                for W in ([0] if pol == "static" else Ws):
                    KK = K if pol.startswith("gated") else 0
                    c = cal["cells"][cal_key(mode, pol, W, KK)]
                    rec = dict(seed=seed, kind=kind, pol=pol, W=W)
                    for tag, x in (("ramp", xr), ("ctrl", xc)):
                        alarm, *_ = run(x, pol, W, KK, k, c["h_a"], c["h_w"], mode)
                        rec[f"A_{tag}"] = float(alarm[lo:hi].mean())
                        rec[f"eps_{tag}"] = 1000.0 * count_eps(alarm[:hi], lo, GAP) / RAMP_LEN
                    rows.append(rec)
    return pd.DataFrame(rows)


def h3f_table(df, floor=1e-3):
    """H3f: static floods (A_ramp >= 5 x A_ctrl); gated-2L stays within 2 x max(rolling A_ramp, own A_ctrl)."""
    g = df.groupby(["kind", "pol", "W"])[["A_ramp", "A_ctrl", "eps_ramp", "eps_ctrl"]].mean().reset_index()
    recs = []
    for kind in g["kind"].unique():
        s = g[(g["kind"] == kind) & (g["pol"] == "static")].iloc[0]
        static_floods = bool(s["A_ramp"] >= 5 * max(s["A_ctrl"], floor))
        for W in sorted(w for w in g["W"].unique() if w):
            r = g[(g["kind"] == kind) & (g["pol"] == "rolling") & (g["W"] == W)]
            q = g[(g["kind"] == kind) & (g["pol"] == "gated2L") & (g["W"] == W)]
            if r.empty or q.empty:
                continue
            r, q = r.iloc[0], q.iloc[0]
            recs.append(dict(kind=kind, W=W, A_static=s["A_ramp"], A_ctrl_static=s["A_ctrl"], A_rolling=r["A_ramp"],
                             A_gated2L=q["A_ramp"], A_ctrl_gated2L=q["A_ctrl"], static_floods=static_floods,
                             gated_ok=bool(q["A_ramp"] <= 2 * max(r["A_ramp"], q["A_ctrl"], floor)),
                             eps_static=s["eps_ramp"], eps_rolling=r["eps_ramp"], eps_gated2L=q["eps_ramp"]))
    return g, pd.DataFrame(recs)


def cmd_ramp(a):
    cal = json.load(open(a.cal))
    df = ramp_run(a.arm, parse_seeds(a.seeds), cal)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "ramp_events.csv"), index=False)
    g, h3 = h3f_table(df)
    g.to_csv(os.path.join(a.out, "ramp_summary.csv"), index=False)
    h3.to_csv(os.path.join(a.out, "h3f.csv"), index=False)
    pd.set_option("display.width", 220)
    print("Fraction of ramp-window bars in alert (A_ramp) versus the stationary stream (A_ctrl):")
    print(g.round(3).to_string(index=False))
    print("\nH3f:")
    print(h3.round(3).to_string(index=False))
    for kind, d in h3.groupby("kind"):
        print(f"  {kind}: static floods = {bool(d['static_floods'].iloc[0])}; gated-2L within tolerance at "
              f"{int(d['gated_ok'].sum())}/{len(d)} windows")


# --- H2b: model stream -------------------------------------------------------------------------------------------
@njit(cache=True)
def _model_reps(n_rep, mode, pol, delta, W, R, h, k, ya, yb, pre, hz, gap, seed, o_det, o_dur, o_contam):
    np.random.seed(seed)
    T = pre + hz
    tau = pre
    alarm = np.zeros(T, np.uint8)
    sq2 = math.sqrt(2.0)
    for rep in range(n_rep):
        phase = 0
        if pol == 2:
            phase = np.random.randint(0, R)
        Sp = 0.0
        Sm = 0.0
        for t in range(T):
            if pol == 2 and (t - phase) % R == 0:
                Sp = 0.0
                Sm = 0.0
            mu = 0.0
            sc = 1.0
            if t >= tau:
                if pol == 0:
                    if mode == 0:
                        mu = delta
                    else:
                        sc = sq2
                else:
                    if pol == 1:
                        p = min((t - tau) / W, 1.0)
                    else:
                        r = t - ((t - phase) % R)
                        p = max(0.0, min((r - tau) / W, 1.0))
                    if mode == 0:
                        sd = math.sqrt(1.0 + p * (1.0 - p) * delta * delta)
                        mu = delta * (1.0 - p) / sd
                        sc = 1.0 / sd
                    else:
                        sc = sq2 / math.sqrt(1.0 + p)
            u = mu + sc * np.random.standard_normal()
            if mode == 0:
                Sp = max(0.0, Sp + u - k)
                Sm = max(0.0, Sm - u - k)
                stat = max(Sp, Sm)
            else:
                y = (abs(u) - ya) / yb
                Sp = max(0.0, Sp + y - k)
                stat = Sp
            alarm[t] = 1 if stat > h else 0
        d, f, du, co = first_episode(alarm, tau, hz, gap)
        o_det[rep] = d
        o_dur[rep] = du
        o_contam[rep] = co


def model_cell(shift, pol, W, h, k, n_rep=4000, seed=2026, hz=H):
    """Monte Carlo of the Gaussian model stream: the detector (same k, calibrated h) on a stream whose standardised
    mean/scale follow the absorption model for the policy's update schedule. Ignores the baseline's estimation
    noise. Returns P(detect), P(blind), mean/median duration of the first episode."""
    kind, delta = shift
    mode = 0 if kind == "level" else 1
    R = max(1, W // 4)
    ya = math.sqrt(2.0 / math.pi)
    yb = math.sqrt(1.0 - 2.0 / math.pi)
    det = np.zeros(n_rep, np.int64)
    dur = np.zeros(n_rep, np.int64)
    con = np.zeros(n_rep, np.int64)
    _model_reps(n_rep, mode, POLS[pol], float(delta or 0.0), float(max(W, 1)), R, float(h), float(k), ya, yb,
                MODEL_PRE, hz, GAP, seed, det, dur, con)
    blind = (det == 0) | (dur < DMIN)
    return dict(P_detect=float(det.mean()), P_blind=float(blind.mean()), dur_mean=float(dur.mean()),
                dur_median=float(np.median(dur)), contam=float(con.mean()))


def model_predict(cal, deltas=(0.75, 1.0, 2.0), n_rep=4000, seed=2026):
    out = {}
    jobs = [(("level", d), 0, cal["k_level"], f"level{d}") for d in deltas] + [(("var", None), 1, cal["k_var"], "var")]
    for shift, mode, k, name in jobs:
        for pol in ("static", "rolling", "periodic"):
            for W in ([0] if pol == "static" else WGRID):
                h = cal["cells"][cal_key(mode, pol, W, 0)]["h_a"]
                out[f"{name}|{pol}|{W}"] = model_cell(shift, pol, W, h, k, n_rep, seed)
    return out


def cmd_model(a):
    cal = json.load(open(a.cal))
    pred = model_predict(cal, n_rep=a.reps)
    json.dump({"cal_arm": cal["arm"], "n_rep": a.reps, "seed": 2026, "pred": pred}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({len(pred)} cells, calibrated on arm={cal['arm']})")
    for key in sorted(pred):
        if key.startswith(("level1.0", "var")) and ("|rolling|" in key or "|periodic|" in key or "|static|" in key):
            p = pred[key]
            print(f"  {key:<22} P(detect) {p['P_detect']:.2f}  P(blind) {p['P_blind']:.2f}  duration median {p['dur_median']:.0f}")


def cmd_compare_model(a):
    mp = json.load(open(a.model))["pred"]
    ev = pd.read_csv(a.events)
    sim = ev.groupby(["shift", "pol", "W"])[["blind", "detect", "duration"]].mean().reset_index()
    rows = []
    for _, r in sim.iterrows():
        key = f"{r['shift']}|{r['pol']}|{int(r['W'])}"
        if r["pol"] in ("rolling", "periodic") and r["shift"] in ("level1.0", "var") and key in mp:
            rows.append(dict(cell=key, W=int(r["W"]), sim_blind=r["blind"], model_blind=mp[key]["P_blind"],
                             sim_detect=r["detect"], model_detect=mp[key]["P_detect"],
                             sim_dur=r["duration"], model_dur=mp[key]["dur_mean"]))
    t = pd.DataFrame(rows)
    if t.empty:
        print("no comparable cells")
        return
    t["abs_err"] = (t["sim_blind"] - t["model_blind"]).abs()
    rho = sps.spearmanr(t["sim_blind"], t["model_blind"])[0]
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    print(f"\nH2b on {len(t)} primary cells: MAE(P(blind)) = {t['abs_err'].mean():.3f} (pass <= 0.10), "
          f"Spearman = {rho:.2f} (pass >= 0.80); worst cell {t.loc[t['abs_err'].idxmax(), 'cell']} "
          f"(err {t['abs_err'].max():.2f})")
    small = t[t["W"] <= 100]
    large = t[t["W"] > 100]
    print(f"  W <= 100: MAE {small['abs_err'].mean():.3f} (n={len(small)});  W > 100: MAE {large['abs_err'].mean():.3f} (n={len(large)})")
    print(f"  H2b: {'PASS' if (t['abs_err'].mean() <= 0.10 and rho >= 0.8) else 'FAIL'}")


# --- blind cells, trend tests, gate evaluation -------------------------------------------------------------------
def blind_cells(events_csv, model_json=None, thr=0.25):
    """Frozen list used by H3a / H3g: (shift, reference policy, W) with P(blind | policy) - P(blind | static) >= thr,
    from the dev events and/or the model stream. Write it BEFORE the test run."""
    ev = pd.read_csv(events_csv)
    out = {"threshold": thr, "dev": [], "model": []}
    pb = ev.groupby(["shift", "pol", "W"])["blind"].mean()
    for shift in sorted(ev["shift"].unique()):
        st = pb.get((shift, "static", 0), np.nan)
        for pol in ("rolling", "periodic"):
            for W in WGRID:
                v = pb.get((shift, pol, W), np.nan)
                if v - st >= thr:
                    out["dev"].append(dict(shift=shift, pol=pol, W=W, gap=float(v - st)))
    if model_json:
        mp = json.load(open(model_json))["pred"]
        for key, v in mp.items():
            shift, pol, W = key.split("|")
            if pol in ("rolling", "periodic"):
                st = mp.get(f"{shift}|static|0")
                if st is not None and v["P_blind"] - st["P_blind"] >= thr:
                    out["model"].append(dict(shift=shift, pol=pol, W=int(W), gap=float(v["P_blind"] - st["P_blind"])))
    seen, union = set(), []
    for c in out["dev"] + out["model"]:
        kk = (c["shift"], c["pol"], c["W"])
        if kk not in seen:
            seen.add(kk)
            union.append(dict(shift=c["shift"], pol=c["pol"], W=c["W"]))
    out["union"] = union
    return out


def cmd_trend(a):
    ev = pd.read_csv(a.events)
    recs = []
    for shift in sorted(ev["shift"].unique()):
        for pol in ("rolling", "periodic"):
            recs.append(trend_test(ev, shift, pol, "detect", +1))
            recs.append(trend_test(ev, shift, pol, "blind200", -1))
            recs.append(trend_test(ev, shift, pol, "alert200", +1))
    t = pd.DataFrame(recs)
    pd.set_option("display.width", 200)
    print("H1b: P(detect) rises with W (direction +1). H1c: P(blind within 200 bars) falls with W (-1),")
    print("fraction of the first 200 bars in alert rises with W (+1). One-sided p; permutation and GEE.\n")
    print(t.round(4).to_string(index=False))
    if a.out:
        t.to_csv(a.out, index=False)


def gate_eval(ev, cells):
    """H3a (1L) and H3g (2L) on the frozen blind cells: share of the P(blind) gap (reference policy - static)
    that the gate closes, alert duration / coverage against the reference, P(detect) recovery against static."""
    rows = []
    for c in cells:
        sub = ev[ev["shift"] == c["shift"]]
        st = sub[sub["pol"] == "static"]
        ref = sub[(sub["pol"] == c["pol"]) & (sub["W"] == c["W"])]
        if st.empty or ref.empty:
            continue
        gap = ref["blind"].mean() - st["blind"].mean()
        for g in ("gated1L", "gated2L"):
            for K in sorted(sub[sub["pol"] == g]["K"].unique()):
                q = sub[(sub["pol"] == g) & (sub["W"] == c["W"]) & (sub["K"] == K)]
                if q.empty or gap <= 0:
                    continue
                closed = (ref["blind"].mean() - q["blind"].mean()) / gap
                rec = dict(shift=c["shift"], ref=c["pol"], W=c["W"], gate=g, K=K, gap=gap, closed=closed,
                           P_detect_gate=q["detect"].mean(), P_detect_static=st["detect"].mean(),
                           dur_gate=q["duration"].mean(), dur_ref=ref["duration"].mean(),
                           cov_gate=q["coverage"].mean(), cov_ref=ref["coverage"].mean())
                rec["recovers_detect"] = bool(rec["P_detect_gate"] >= 0.9 * rec["P_detect_static"])
                rec["H3a_ok"] = bool(closed >= 0.5 and rec["dur_gate"] > rec["dur_ref"] and rec["cov_gate"] > rec["cov_ref"])
                rec["H3g_ok"] = bool(closed >= 0.5 and rec["recovers_detect"])
                rows.append(rec)
    return pd.DataFrame(rows)


def cmd_gate_eval(a):
    cells = json.load(open(a.cells))["union"]
    res = gate_eval(pd.read_csv(a.events), cells)
    pd.set_option("display.width", 220)
    if res.empty:
        print("no blind cells with a positive gap (nothing to evaluate)")
        return
    print(res.round(3).to_string(index=False))
    for g, flag in (("gated1L", "H3a_ok"), ("gated2L", "H3g_ok")):
        d = res[(res["gate"] == g) & (res["K"] == 250)]
        if len(d):
            print(f"{'H3a' if g == 'gated1L' else 'H3g'} ({g}, K=250): {int(d[flag].sum())}/{len(d)} blind cells pass")
    if a.out:
        res.to_csv(a.out, index=False)


# --- unit tests (protocol section 7) -----------------------------------------------------------------------------
def _ok(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")
    return bool(cond)


def _ref_first_episode(alarm, tau, hz, gap):
    """Plain-python reference for first_episode (episodes over the whole array, then censoring at the horizon)."""
    idx = np.flatnonzero(alarm)
    end = tau + hz
    if idx.size == 0:
        return 0, hz, 0, 0
    ep = np.cumsum(np.r_[True, np.diff(idx) > gap])
    contam = 0
    first_ep = None
    for e in np.unique(ep):
        fl = idx[ep == e]
        if fl[0] < tau:
            if ((fl >= tau) & (fl < end)).any():
                contam = 1
        elif first_ep is None and fl[0] < end:
            first_ep = fl[fl < end]
    if first_ep is None:
        return 0, hz, 0, contam
    return 1, int(first_ep[0] - tau), int(first_ep[-1] - first_ep[0] + 1), contam


def run_tests():
    ok = True
    print("Generators")
    xi = np.concatenate([noise(s, "iid") for s in range(1, 21)])
    ok &= _ok("iid mean ~ 0, var ~ 1", abs(xi.mean()) < 0.01 and abs(xi.var() - 1) < 0.02,
              f"(mean {xi.mean():.4f}, var {xi.var():.4f})")
    xb = np.concatenate([noise(s, "btclike") for s in range(1, 101)])
    q = np.percentile(xb, [0.1, 1, 25, 75, 99, 99.9])
    qr1, qr2 = (q[4] - q[1]) / (q[3] - q[2]), (q[5] - q[0]) / (q[3] - q[2])
    ac = np.corrcoef(np.abs(xb[:-1]), np.abs(xb[1:]))[0, 1]
    ok &= _ok("btclike var ~ 1", abs(xb.var() - 1) < 0.1, f"(var {xb.var():.3f})")
    ok &= _ok("btclike heavier tails than Gaussian (3.45, 4.58)", qr1 > 3.6 and qr2 > 5.0,
              f"(q99-1/IQR {qr1:.2f}, q99.9-0.1/IQR {qr2:.2f})")
    ok &= _ok("btclike |r| lag-1 autocorrelation > 0.1", ac > 0.1, f"({ac:.3f})")
    ons = [make_stream(s, "iid")[1] for s in range(1, 2001)]
    ok &= _ok("onset on 6,000 .. 12,000 inclusive", min(ons) >= 6000 and max(ons) <= 12000 and max(ons) > 11900,
              f"(min {min(ons)}, max {max(ons)})")

    print("Baselines")
    n = 8000
    xs = np.zeros(n)
    tau = 5000
    xs[tau:] = 1.0
    alarm, watch, m, s, *_ = run(xs + 1e-3 * np.random.default_rng(0).standard_normal(n), "rolling", 400, 0, 0.5, INF, INF, 0, warm=3000)
    sidx = np.flatnonzero(m[tau:] >= 0.9)[0]
    ok &= _ok("rolling baseline mean reaches 0.9 delta at 0.9 W", abs(sidx - 0.9 * 400) <= 3, f"(s = {sidx} vs {0.9 * 400:.0f})")

    x, tau, end = make_stream(7, "btclike", ("level", 1.0))
    t0 = 9000
    x2 = x.copy()
    x2[t0:] = np.random.default_rng(1).standard_normal(len(x) - t0) * 3
    causal = True
    for pol, K in (("rolling", 0), ("periodic", 0), ("gated1L", 250), ("gated2L", 250)):
        a1, w1, m1, s1, *_ = run(x, pol, 250, K, 0.5, 6.0, 3.0, 0)
        a2, w2, m2, s2, *_ = run(x2, pol, 250, K, 0.5, 6.0, 3.0, 0)
        causal &= bool(np.array_equal(m1[:t0 + 1], m2[:t0 + 1]) and np.array_equal(s1[:t0 + 1], s2[:t0 + 1])
                       and np.array_equal(a1[:t0], a2[:t0]))
    ok &= _ok("baselines are causal (future bars do not change earlier (m, s) or flags)", causal)

    print("Gates")
    same = True
    for seed in range(1, 8):
        x, tau, _ = make_stream(seed, "btclike", ("level", 2.0))
        ar, _, mr, sr, *_ = run(x, "rolling", 250, 0, 0.5, 8.0, INF, 0)
        ag, _, mg, sg, *_ = run(x, "gated1L", 250, 250, 0.5, 8.0, INF, 0)
        fa = np.flatnonzero(ag)
        if fa.size == 0 or fa[0] < WARM:
            continue
        i = fa[0]
        same &= bool(np.array_equal(ar[:i + 1], ag[:i + 1]) and np.array_equal(mr[:i + 1], mg[:i + 1]))
    ok &= _ok("gated-1L has identical flags and baseline to rolling up to the first alarm (same thresholds)", same)

    print("Episodes")
    rng = np.random.default_rng(5)
    agree = True
    for _ in range(600):
        al = (rng.random(400) < rng.choice([0.02, 0.05, 0.15])).astype(np.uint8)
        tau_, hz_ = int(rng.integers(30, 120)), int(rng.integers(20, 200))
        agree &= tuple(int(v) for v in first_episode(al, tau_, hz_, GAP)) == _ref_first_episode(al, tau_, hz_, GAP)
    ok &= _ok("first_episode agrees with a plain-python reference on 600 random alarm arrays", agree)
    al = np.zeros(300, np.uint8)
    al[95:110] = 1                      # in progress at onset (tau = 100), continues to 109
    al[150:160] = 1                     # a new episode
    d, f, du, co = first_episode(al, 100, 100, GAP)
    ok &= _ok("an alert already in progress at onset is skipped; the next episode is the first",
              (d, f, du, co) == (1, 50, 10, 1), f"({d}, {f}, {du}, {co})")
    al = np.zeros(300, np.uint8)
    al[80:90] = 1                       # ended before onset
    al[103:106] = 1
    d, f, du, co = first_episode(al, 100, 100, GAP)
    ok &= _ok("an alert that ended before onset does not contaminate", (d, f, du, co) == (1, 3, 3, 0), f"({d}, {f}, {du}, {co})")

    print("Implementation")
    x, tau, _ = make_stream(3, "btclike", ("level", 1.0))
    x = x[:900]
    res = []
    for fn in (run_stream, run_stream.py_func):
        alarm = np.zeros(len(x), np.uint8)
        watch = np.zeros(len(x), np.uint8)
        m = np.zeros(len(x))
        s = np.ones(len(x))
        fn(x, 4, 40, 10, 100, 0.5, 4.0, 2.0, 0, 200, alarm, watch, m, s)
        res.append((alarm.copy(), watch.copy(), m.copy(), s.copy()))
    ok &= _ok("compiled and pure-python paths agree (gated-2L)", all(np.allclose(a, b) for a, b in zip(res[0], res[1])))

    print("Theory")
    def p_star(d, k):
        return brentq(lambda p: d * (1 - p) / math.sqrt(1 + p * (1 - p) * d * d) - k, 0.0, 0.999)
    p1, p2 = p_star(1.0, 0.5), p_star(2.0, 0.5)
    ok &= _ok("CUSUM visibility limits p* ~ 0.44 (delta=1) and ~0.66 (delta=2)",
              abs(p1 - 0.44) < 0.02 and abs(p2 - 0.66) < 0.02, f"({p1:.3f}, {p2:.3f})")
    def med_abs(p, d):
        return brentq(lambda mm: (1 - p) * sps.norm.cdf(mm) + p * sps.norm.cdf(mm - d) - 0.5, -5, 8)
    p90 = brentq(lambda p: med_abs(p, 1.0) - 0.9, 0.01, 0.999)
    p90_2 = brentq(lambda p: med_abs(p, 2.0) - 1.8, 0.01, 0.999)
    ok &= _ok("mixture-median reaches 90% of the shift at p ~ 0.89 (delta=1) and ~ 0.85 (delta=2)",
              abs(p90 - 0.89) < 0.01 and abs(p90_2 - 0.85) < 0.01, f"({p90:.3f}, {p90_2:.3f})")

    print("Calibration and model stream (small)")
    xs_cal, xs_chk = cal_streams("iid", 777, 8), cal_streams("iid", 888, 8)
    c = calibrate_cell(xs_cal, xs_chk, "rolling", 250, 0, 0.5, 0)
    ok &= _ok("rolling W=250 iid calibrates to ~1 episode / 1,000 bars on an independent stream",
              0.6 <= c["rate_chk"] <= 1.5, f"(cal {c['rate_cal']:.2f}, chk {c['rate_chk']:.2f}, h={c['h_a']:.2f})")
    ok &= _ok("level-statistic in-control in-alert fraction is small (iid)", c["alert_frac_chk"] < ALERT_FLAG,
              f"({100 * c['alert_frac_chk']:.2f}%)")
    h = c["h_a"]
    ms = model_cell(("level", 2.0), "static", 0, h, 0.5, n_rep=1000)
    mr = model_cell(("level", 1.0), "rolling", 50, h, 0.5, n_rep=1000)
    mst = model_cell(("level", 1.0), "static", 0, h, 0.5, n_rep=1000)
    mp = model_cell(("level", 1.0), "periodic", 50, h, 0.5, n_rep=1000)
    ok &= _ok("model stream: static detects a +2 sigma shift essentially always", ms["P_detect"] > 0.99, f"({ms['P_detect']:.3f})")
    ok &= _ok("model stream: W=50 rolling and periodic detect less than static at +1 sigma",
              mr["P_detect"] < mst["P_detect"] and mp["P_detect"] < mr["P_detect"] + 0.05,
              f"(static {mst['P_detect']:.2f}, rolling {mr['P_detect']:.2f}, periodic {mp['P_detect']:.2f})")
    ramp_cal = {"k_level": 0.5, "k_var": 0.2, "cells": {
        cal_key(0, "static", 0, 0): calibrate_cell(xs_cal, xs_chk, "static", 0, 0, 0.5, 0),
        cal_key(0, "rolling", 250, 0): c}}
    rr = ramp_run("iid", [1, 2], ramp_cal, pols=("static", "rolling"), Ws=[250], kinds=RAMP_KINDS[:1])
    A_s = rr[rr["pol"] == "static"]["A_ramp"].mean()
    A_r = rr[rr["pol"] == "rolling"]["A_ramp"].mean()
    ok &= _ok("ramp runner: a static baseline sits in alert on a +2 sigma ramp, a rolling W=250 baseline mostly does not",
              A_s > 0.5 and A_r < A_s, f"(static {A_s:.2f}, rolling {A_r:.2f})")
    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    return 0 if ok else 1


# --- CLI ---------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    p = sub.add_parser("kvar")
    p.add_argument("--arm", default="btclike")
    p.add_argument("--seeds", default="1-30")
    p = sub.add_parser("calibrate")
    p.add_argument("--arm", default="btclike")
    p.add_argument("--kvar-seeds", default="1-30")
    p.add_argument("--out", default=None)
    p.add_argument("--m", type=int, default=M_CAL, help="number of 20,000-bar calibration streams")
    p = sub.add_parser("dev")
    p.add_argument("--arm", default="btclike")
    p.add_argument("--cal", required=True)
    p.add_argument("--seeds", default="1-30")
    p.add_argument("--out", default="dev_out")
    p.add_argument("--persist", type=int, default=None)
    p.add_argument("--extra-deltas", type=float, nargs="*", default=[0.75])
    p.add_argument("--ks", type=int, nargs="*", default=[250], help="gate freeze lengths (250 primary, 1000 secondary)")
    p = sub.add_parser("figure1")
    p.add_argument("--arm", default="btclike")
    p.add_argument("--cal", required=True)
    p.add_argument("--seeds", default="1-30")
    p.add_argument("--out", default="dev_out")
    p = sub.add_parser("ramp")
    p.add_argument("--arm", default="btclike")
    p.add_argument("--cal", required=True)
    p.add_argument("--seeds", default="1-30")
    p.add_argument("--out", default="dev_out")
    p = sub.add_parser("model")
    p.add_argument("--cal", required=True, help="calibration of the arm the model is compared with (iid for the pure Gaussian check)")
    p.add_argument("--out", default="model_pred.json")
    p.add_argument("--reps", type=int, default=4000)
    p = sub.add_parser("blindcells")
    p.add_argument("--events", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--thr", type=float, default=0.25)
    p.add_argument("--out", default="blind_cells.json")
    p = sub.add_parser("compare-model")
    p.add_argument("--events", required=True)
    p.add_argument("--model", required=True)
    p = sub.add_parser("trend")
    p.add_argument("--events", required=True)
    p.add_argument("--out", default=None)
    p = sub.add_parser("gate-eval")
    p.add_argument("--events", required=True)
    p.add_argument("--cells", required=True)
    p.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.cmd == "test":
        sys.exit(run_tests())
    if a.cmd == "kvar":
        kv, shift = measure_k_var(a.arm, parse_seeds(a.seeds))
        print(f"arm={a.arm}: measured shift in y = {shift:.3f}  ->  k_var = {kv:.3f}")
    elif a.cmd == "calibrate":
        kv, shift = measure_k_var(a.arm, parse_seeds(a.kvar_seeds))
        print(f"k_var = {kv:.3f} (shift in y {shift:.3f})")
        cal = calibrate_all(a.arm, kv, M=a.m)
        out = a.out or f"cal_{a.arm}.json"
        json.dump(cal, open(out, "w"), indent=1)
        bad = [k for k, v in cal["cells"].items() if not v["ok"]]
        longs = [k for k, v in cal["cells"].items() if v["long_episodes"]]
        print(f"wrote {out}; {len(bad)} cell(s) out of tolerance: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
        print(f"{len(longs)} cell(s) with in-control in-alert fraction > {100 * ALERT_FLAG:.0f}% "
              f"(long episodes; episode rate is matched, time in alert is not): {longs[:10]}{' ...' if len(longs) > 10 else ''}")
    elif a.cmd == "dev":
        cmd_dev(a)
    elif a.cmd == "figure1":
        cmd_figure1(a)
    elif a.cmd == "ramp":
        cmd_ramp(a)
    elif a.cmd == "model":
        cmd_model(a)
    elif a.cmd == "blindcells":
        res = blind_cells(a.events, a.model, a.thr)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}: {len(res['dev'])} cells from dev events, {len(res['model'])} from the model stream, "
              f"{len(res['union'])} in the union")
    elif a.cmd == "compare-model":
        cmd_compare_model(a)
    elif a.cmd == "trend":
        cmd_trend(a)
    elif a.cmd == "gate-eval":
        cmd_gate_eval(a)


if __name__ == "__main__":
    main()