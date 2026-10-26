"""sim.py -- adaptive-baseline blindness experiments (Day-1 code).

Pieces
  1. Stream generator      gen_control / gen_stream
  2. Baseline policies     Baseline(PolicySpec): static, rolling, periodic,
                           triggered, gated, dspot
  3. Detectors             ZScore, EWMA, CUSUM, IForestDetector  (update(v)->(alarm, score))
  4. Runner                run_stream(x, spec, detector, kind)
  5. Calibration           calibrate(...)  bisects the threshold to a target
                           alarm-EPISODE rate on a long in-control stream
  6. Metrics               evaluate_event, absorption_curve
  7. Theory                mu_path, cusum_D, cusum_crit_p, p_detect_periodic,
                           zscore_alarm_prob

Usage
  python sim.py --selftest
  python sim.py --calibrate cusum rolling iid        (W via --W)

Conventions
  * Bars 0..WARM-1 are warm-up. Detection starts at bar WARM.
  * All detectors are scored from WARM, but alarm episodes are only counted from
    WARM + BURN so every detector (incl. Isolation Forest, which needs ~1,500
    rows of history) is judged on the same bars.
  * CUSUM / EWMA are STATE-based: they stay "in alarm" while the statistic is
    above threshold (no reset on alarm). Only periodic(W, R) resets them.
  * 'retrains' counts baseline recomputations: rolling/dspot = one per bar,
    periodic = one per R bars, gated = one per bar while not frozen plus one per
    timed-out re-baseline, triggered = one per ADWIN/PH firing.
  * Baseline windows are truncated at bar 0 (W > bars available -> shorter window).
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import lfilter
except Exception:  # pragma: no cover
    lfilter = None

try:  # ADWIN / Page-Hinkley from river, with a small fallback for PH
    from river.drift import ADWIN, PageHinkley
    HAVE_RIVER = True
except Exception:  # pragma: no cover
    HAVE_RIVER = False

N = 20_000
WARM = 3_000
BURN = 1_600
H = 500
MERGE = 5
SQ2 = math.sqrt(2.0)


# --------------------------------------------------------------------------
# 1. Stream generator
# --------------------------------------------------------------------------
def gen_noise(n, noise, rng, phi=0.5):
    """Unit-variance noise: 'iid' or 'ar1' (lag-1 autocorrelation phi)."""
    if noise == "iid":
        return rng.standard_normal(n)
    if noise != "ar1":
        raise ValueError(noise)
    e = rng.standard_normal(n) * math.sqrt(1 - phi * phi)
    x0 = rng.standard_normal()
    if lfilter is not None:
        y, _ = lfilter([1.0], [1.0, -phi], e[1:], zi=[phi * x0])
        return np.concatenate([[x0], y])
    x = np.empty(n)  # pragma: no cover
    x[0] = x0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def gen_control(n, noise, rng):
    return gen_noise(n, noise, rng)


def gen_stream(rng, n=N, noise="iid", kind="level", delta=1.0, persistence=None,
               onset=None, onset_range=(6000, 12000)):
    """Return (x, onset). kind='level': +delta sigma; kind='var': variance x2
    (delta ignored). persistence=None -> permanent, else shift lasts that many bars."""
    x = gen_noise(n, noise, rng)
    if onset is None:
        onset = int(rng.integers(onset_range[0], onset_range[1] + 1))
    end = n if persistence is None else min(n, onset + persistence)
    if kind == "level":
        x[onset:end] += delta
    elif kind == "var":
        x[onset:end] *= SQ2
    else:
        raise ValueError(kind)
    return x, onset


# --------------------------------------------------------------------------
# 2. Baseline policies
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PolicySpec:
    name: str            # static | rolling | periodic | triggered | gated | dspot
    W: int = 250
    R: int | None = None  # periodic reset interval (default W // 4)
    K: int = 250          # gated freeze length

    @property
    def feedback(self):   # baseline depends on detector alarms / change detectors
        return self.name in ("triggered", "gated", "dspot")


class _SimplePH:
    """Two-sided Page-Hinkley fallback (used only if river is missing)."""

    def __init__(self, delta=0.005, lam=50.0):
        self.delta, self.lam = delta, lam
        self.n = 0; self.mean = 0.0; self.up = 0.0; self.dn = 0.0
        self.drift_detected = False

    def update(self, v):
        self.n += 1
        self.mean += (v - self.mean) / self.n
        self.up = max(0.0, self.up + v - self.mean - self.delta)
        self.dn = max(0.0, self.dn - v + self.mean - self.delta)
        self.drift_detected = self.up > self.lam or self.dn > self.lam
        return self


class _Trigger:
    """ADWIN or Page-Hinkley fires -> True. Both are reset after a firing."""

    def __init__(self):
        self._make()

    def _make(self):
        if HAVE_RIVER:
            self.adwin = ADWIN(delta=0.002)
            self.ph = PageHinkley(min_instances=30, delta=0.005, threshold=50.0, mode="both")
        else:  # pragma: no cover
            self.adwin = None
            self.ph = _SimplePH()

    def update(self, v):
        fired = False
        if self.adwin is not None:
            self.adwin.update(v)
            fired = bool(self.adwin.drift_detected)
        self.ph.update(v)
        fired = fired or bool(self.ph.drift_detected)
        if fired:
            self._make()
        return fired


class Baseline:
    """Computes (m, s) for bar t from bars < t only.

    Generic path:  m, s = begin(t); ... score ...; end(t, x_t, flag, feed)
    Vector path :  vector()  (only for alarm-independent policies)
    """

    def __init__(self, spec: PolicySpec, x, warm=WARM):
        self.spec, self.x, self.warm = spec, x, warm
        self.W, self.K = spec.W, spec.K
        self.R = spec.R if spec.R else max(1, spec.W // 4)
        self.cs = np.concatenate([[0.0], np.cumsum(x)])
        self.cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
        self.retrains = 0
        self.reset = False
        self.m0, self.s0 = self.stats(0, warm)
        self.m, self.s = self.m0, self.s0
        # gated state
        self.frozen = False; self.fz_start = 0; self.last_flag = -10**9
        self.fm = self.fs = None; self.n_freezes = 0
        # dspot state
        if spec.name == "dspot":
            a = max(0, warm - self.W)
            seg = x[a:warm]
            self.buf = deque(seg.tolist())
            self.bsum = float(seg.sum()); self.bsum2 = float((seg * seg).sum())
        # triggered state
        if spec.name == "triggered":
            self.trig = _Trigger(); self.pending = False

    # -- helpers ----------------------------------------------------------
    def stats(self, a, b):
        n = b - a
        mu = (self.cs[b] - self.cs[a]) / n
        var = (self.cs2[b] - self.cs2[a]) / n - mu * mu
        return mu, math.sqrt(max(var, 1e-8))

    def _roll(self, t):
        self.m, self.s = self.stats(max(0, t - self.W), t)
        self.retrains += 1

    # -- generic path -----------------------------------------------------
    def begin(self, t):
        self.reset = False
        name = self.spec.name
        if name == "static":
            return self.m0, self.s0
        if name == "rolling":
            self._roll(t)
        elif name == "periodic":
            if t == self.warm or (t - self.warm) % self.R == 0:
                self.m, self.s = self.stats(max(0, t - self.W), t)
                self.retrains += 1
                self.reset = True
        elif name == "triggered":
            if self.pending:
                self.m, self.s = self.stats(max(0, t - self.W), t)
                self.retrains += 1; self.reset = True; self.pending = False
        elif name == "gated":
            if self.frozen:
                if t - self.fz_start >= self.K:       # timed out: re-baseline on last K bars
                    self.frozen = False
                    self.m, self.s = self.stats(max(0, t - self.K), t)
                    self.retrains += 1
                    return self.m, self.s
                if t - self.last_flag < 50:           # still flagging: hold
                    return self.fm, self.fs
                self.frozen = False                    # 50 quiet bars: resume rolling early
            self._roll(t)
        elif name == "dspot":
            n = len(self.buf)
            if n >= 20:
                mu = self.bsum / n
                self.m = mu
                self.s = math.sqrt(max(self.bsum2 / n - mu * mu, 1e-8))
            self.retrains += 1
        return self.m, self.s

    def end(self, t, xt, flag, feed):
        name = self.spec.name
        if name == "gated":
            if flag:
                if not self.frozen:
                    self.frozen = True; self.fz_start = t + 1
                    self.fm, self.fs = self.m, self.s    # pre-alarm values
                    self.n_freezes += 1
                self.last_flag = t
        elif name == "dspot":
            if not flag:
                self.buf.append(xt); self.bsum += xt; self.bsum2 += xt * xt
                if len(self.buf) > self.W:
                    old = self.buf.popleft(); self.bsum -= old; self.bsum2 -= old * old
        elif name == "triggered":
            if self.trig.update(feed):
                self.pending = True

    # -- vector path ------------------------------------------------------
    def vector(self):
        n = len(self.x); w = self.warm
        t = np.arange(n)
        m = np.full(n, np.nan); s = np.full(n, np.nan)
        rst = np.zeros(n, bool)
        name = self.spec.name
        if name == "static":
            m[w:], s[w:] = self.m0, self.s0
            return m, s, rst, 0

        def stats_v(a, b):
            cnt = (b - a).astype(float)
            mu = (self.cs[b] - self.cs[a]) / cnt
            var = (self.cs2[b] - self.cs2[a]) / cnt - mu * mu
            return mu, np.sqrt(np.maximum(var, 1e-8))

        tt = t[w:]
        if name == "rolling":
            m[w:], s[w:] = stats_v(np.maximum(0, tt - self.W), tt)
            return m, s, rst, n - w
        if name == "periodic":
            idx = w + ((tt - w) // self.R) * self.R
            m[w:], s[w:] = stats_v(np.maximum(0, idx - self.W), idx)
            rst[w:] = ((tt - w) % self.R) == 0
            return m, s, rst, int(rst.sum())
        raise ValueError("vector() only for static/rolling/periodic")


# --------------------------------------------------------------------------
# 3. Detectors
# --------------------------------------------------------------------------
class ZScore:
    def __init__(self, thr): self.thr = thr
    def reset(self): pass
    def update(self, v):
        a = abs(v)
        return a > self.thr, a


class EWMA:
    def __init__(self, thr, lam=0.1):
        self.lam = lam; self.thr = thr          # thr = L
        self.limit_unit = math.sqrt(lam / (2 - lam)); self.e = 0.0
    def reset(self): self.e = 0.0
    def update(self, v):
        self.e = self.lam * v + (1 - self.lam) * self.e
        sc = abs(self.e) / self.limit_unit
        return sc > self.thr, sc


class CUSUM:
    def __init__(self, thr, k=0.5):
        self.thr = thr; self.k = k; self.sp = 0.0; self.sm = 0.0   # thr = h
    def reset(self): self.sp = self.sm = 0.0
    def update(self, v):
        k = self.k
        sp = self.sp + v - k; sm = self.sm - v - k
        self.sp = sp if sp > 0.0 else 0.0
        self.sm = sm if sm > 0.0 else 0.0
        sc = self.sp if self.sp > self.sm else self.sm
        return sc > self.thr, sc


class IForestDetector:
    """Windowed features (x_t, mean, std, range over 30 bars); sklearn
    IsolationForest trained on `train_rows` rows, refit every `refit` bars. Threshold =
    q-quantile of out-of-sample scores on the freshest `oos_rows` rows (rows the
    model has not seen). thr = q. Lower score = more anomalous.
    Block-scoring (run_batch) and per-bar scoring (update) give identical results."""

    FEAT = 30

    def __init__(self, thr, train_rows=1000, oos_rows=500, refit=250, n_est=50, seed=0):
        self.q = thr; self.train, self.oos, self.refit = train_rows, oos_rows, refit
        self.n_est, self.seed = n_est, seed
        self.T0 = train_rows + oos_rows
        self.reset_state()

    def reset_state(self):
        self.vals = deque(maxlen=self.FEAT); self.rows = []
        self.model = None; self.cut = None

    def reset(self): pass   # IF is not accumulator-based

    def _fit(self, F_train, F_oos):
        from sklearn.ensemble import IsolationForest
        mdl = IsolationForest(n_estimators=self.n_est, random_state=self.seed).fit(F_train)
        cut = float(np.quantile(mdl.score_samples(F_oos), self.q))
        return mdl, cut

    def update(self, v):
        self.vals.append(v)
        if len(self.vals) < self.FEAT:
            return False, 0.0
        w = np.fromiter(self.vals, float)
        row = [v, w.mean(), w.std(), w.max() - w.min()]
        idx = len(self.rows)
        if idx >= self.T0 and (idx - self.T0) % self.refit == 0:
            R = np.asarray(self.rows)
            self.model, self.cut = self._fit(R[idx - self.oos - self.train: idx - self.oos],
                                             R[idx - self.oos: idx])
        self.rows.append(row)
        if self.model is None:
            return False, 0.0
        sc = float(self.model.score_samples(np.asarray([row]))[0])
        return sc < self.cut, -sc

    def run_batch(self, v):
        v = np.asarray(v, float); n = len(v); out = np.zeros(n, bool)
        if n < self.FEAT + self.T0:
            return out
        win = np.lib.stride_tricks.sliding_window_view(v, self.FEAT)
        F = np.column_stack([v[self.FEAT - 1:], win.mean(1), win.std(1), win.max(1) - win.min(1)])
        for b in range(self.T0, len(F), self.refit):
            mdl, cut = self._fit(F[b - self.oos - self.train: b - self.oos], F[b - self.oos: b])
            e = min(b + self.refit, len(F))
            out[self.FEAT - 1 + b: self.FEAT - 1 + e] = mdl.score_samples(F[b:e]) < cut
        return out


def make_detector(name, thr, kind="level"):
    return {"zscore": ZScore, "ewma": EWMA, "cusum": CUSUM, "iforest": IForestDetector}[name](thr)


# --------------------------------------------------------------------------
# 4. Runner
# --------------------------------------------------------------------------
def run_stream(x, spec: PolicySpec, detector, kind="level", warm=WARM, force_loop=False):
    """kind='level' feeds u_t; kind='var' feeds u_t^2 - 1 (statistical detectors
    and the IF alike). Returns dict(flags, m, s, retrains)."""
    n = len(x)
    base = Baseline(spec, x, warm)
    flags = np.zeros(n, bool)
    if not spec.feedback and not force_loop:
        m, s, rst, retr = base.vector()
        u = (x - m) / s
        feed = u if kind == "level" else u * u - 1.0
        if isinstance(detector, IForestDetector):
            flags[warm:] = detector.run_batch(feed[warm:])
        elif isinstance(detector, ZScore):
            flags[warm:] = np.abs(feed[warm:]) > detector.thr
        else:
            fl = feed.tolist(); rl = rst.tolist(); upd = detector.update
            for t in range(warm, n):
                if rl[t]:
                    detector.reset()
                flags[t] = upd(fl[t])[0]
        return dict(flags=flags, m=m, s=s, retrains=retr, freezes=0)
    m_arr = np.full(n, np.nan); s_arr = np.full(n, np.nan)
    xl = x.tolist()
    for t in range(warm, n):
        m, s = base.begin(t)
        m_arr[t] = m; s_arr[t] = s
        if base.reset:
            detector.reset()
        u = (xl[t] - m) / s
        v = u if kind == "level" else u * u - 1.0
        a = detector.update(v)[0]
        flags[t] = a
        base.end(t, xl[t], a, v)
    return dict(flags=flags, m=m_arr, s=s_arr, retrains=base.retrains, freezes=base.n_freezes)


# --------------------------------------------------------------------------
# 5. Episodes and calibration
# --------------------------------------------------------------------------
def episodes(flags, merge=MERGE, start=0):
    """Start indices of alarm episodes; flags with gap <= merge bars are one episode."""
    idx = np.flatnonzero(flags[start:]) + start
    if idx.size == 0:
        return idx
    return idx[np.r_[True, np.diff(idx) > merge]]


# name -> (lo, hi, alarms_decrease_with_param)
SEARCH = {"zscore": (0.5, 8.0, True), "ewma": (0.5, 8.0, True),
          "cusum": (0.5, 60.0, True), "iforest": (0.0, 0.05, False)}


def calibrate(det, spec, noise="iid", kind="level", n=300_000, target=1e-3,
              iters=14, seed=12345, tol=0.2):
    """Bisect threshold so alarm-episode rate on an in-control stream (policy active)
    equals `target` per bar. Common random numbers across iterations.
    Returns dict(thr, rate, ok). ok=False -> rate not within +-tol of target."""
    x = gen_control(n, noise, np.random.default_rng(seed))
    lo, hi, dec = SEARCH[det]
    start = WARM + BURN
    nbars = n - start

    def rate(thr):
        r = run_stream(x, spec, make_detector(det, thr, kind), kind)
        return len(episodes(r["flags"], start=start)) / nbars

    for _ in range(iters):
        mid = (lo + hi) / 2
        too_many = rate(mid) > target
        if dec:
            lo, hi = (mid, hi) if too_many else (lo, mid)
        else:
            lo, hi = (lo, mid) if too_many else (mid, hi)
    thr = hi if dec else lo
    r = rate(thr)
    return dict(det=det, policy=spec.name, W=spec.W, noise=noise, kind=kind, thr=thr,
                rate=r, ok=abs(r / target - 1) <= tol)


# --------------------------------------------------------------------------
# 6. Metrics
# --------------------------------------------------------------------------
def evaluate_event(x, onset, spec, det_name, thr, kind="level", delta=1.0, horizon=H):
    r = run_stream(x, spec, make_detector(det_name, thr, kind), kind)
    seg = r["flags"][onset:onset + horizon]
    hit = np.flatnonzero(seg)
    detected = hit.size > 0
    delay = int(hit[0]) if detected else horizon          # misses censored at H
    m, s = r["m"], r["s"]
    if kind == "level":
        moved = (m[onset:] - m[onset - 1]) >= 0.9 * delta
    else:
        moved = s[onset:] >= s[onset - 1] * (1 + 0.9 * (SQ2 - 1))
    tta = int(np.argmax(moved)) if moved.any() else float("nan")
    return dict(detected=detected, delay=delay, coverage=float(seg.mean()),
                tta=tta, retrains=r["retrains"])


def absorption_curve(seeds, spec, det_name, thr, noise="iid", kind="level", delta=1.0,
                     persistence=None, horizon=H):
    """Fraction of streams flagging at each bar after onset (Figure 1)."""
    acc = np.zeros(horizon)
    for sd in seeds:
        x, on = gen_stream(np.random.default_rng(sd), noise=noise, kind=kind,
                           delta=delta, persistence=persistence)
        acc += run_stream(x, spec, make_detector(det_name, thr, kind), kind)["flags"][on:on + horizon]
    return acc / len(seeds)


# --------------------------------------------------------------------------
# 7. Theory (section 4 of the design)
# --------------------------------------------------------------------------
def _phi_sf(z):
    return 0.5 * math.erfc(z / SQ2)


def mu_path(delta, W, s):
    """Expected standardised level of a shifted bar s bars after onset
    (absorption fraction p = s/W)."""
    p = min(s / W, 1.0)
    return delta * (1 - p) / math.sqrt(1 + p * (1 - p) * delta ** 2)


def zscore_alarm_prob(delta, W, s, c):
    """P(|u|>c) upper tail only, 1 - Phi(c*sqrt(1+p(1-p)d^2) - d(1-p))."""
    p = min(s / W, 1.0)
    return _phi_sf(c * math.sqrt(1 + p * (1 - p) * delta ** 2) - delta * (1 - p))


def cusum_crit_p(delta, k=0.5):
    """Absorption fraction p at which the CUSUM drift falls to k."""
    lo, hi = 0.0, 1.0
    f = lambda p: delta * (1 - p) / math.sqrt(1 + p * (1 - p) * delta ** 2) - k
    if f(0.0) <= 0:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return lo


def cusum_D(delta, W, h, k=0.5):
    """Detection time of the noise-free mean-path CUSUM: first s with S>=h
    (inf if the accumulator never gets there). Noise-free => an approximation."""
    S = 0.0
    for s in range(1, W + 1):
        S = max(0.0, S + mu_path(delta, W, s) - k)
        if S >= h:
            return s
    return math.inf


def p_detect_periodic(delta, W, R, h, k=0.5):
    """Mechanism 2: P(detect) ~ max(0, 1 - D/R), uniform onset phase."""
    D = cusum_D(delta, W, h, k)
    return 0.0 if math.isinf(D) else max(0.0, 1 - D / R)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def selftest():
    ok_all = True

    def check(name, cond, info=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

    rng = np.random.default_rng(0)
    # generator: in-control moments
    for noise, rho in (("iid", 0.0), ("ar1", 0.5)):
        x = gen_control(400_000, noise, rng)
        r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        check(f"control {noise}: mean/var/lag1",
              abs(x.mean()) < 0.02 and abs(x.var() - 1) < 0.02 and abs(r1 - rho) < 0.01,
              f"mean={x.mean():.3f} var={x.var():.3f} r1={r1:.3f}")
    # generator: shifts
    x, on = gen_stream(rng, n=200_000, kind="level", delta=2.0, onset=6000)
    check("level shift size", abs(x[on:].mean() - 2.0) < 0.03 and abs(x[:on].mean()) < 0.05)
    x, on = gen_stream(rng, n=200_000, kind="var", onset=6000)
    check("variance x2", abs(x[on:].var() - 2.0) < 0.06)
    x, on = gen_stream(rng, n=50_000, kind="level", delta=1.0, persistence=200, onset=6000)
    check("persistence reverts", abs(x[on:on + 200].mean() - 1) < 0.3 and abs(x[on + 200:].mean()) < 0.05)
    # onset range
    ons = [gen_stream(np.random.default_rng(i))[1] for i in range(200)]
    check("onset in [6000,12000]", min(ons) >= 6000 and max(ons) <= 12000)
    # episodes
    f = np.zeros(100, bool); f[[10, 12, 15, 30, 31]] = True
    check("episode merging (gap<=5)", list(episodes(f)) == [10, 30])
    # causality: changing future bars must not change earlier m,s
    x = gen_control(8000, "iid", np.random.default_rng(1)); x2 = x.copy(); x2[5000:] += 5
    for sp in (PolicySpec("rolling", 100), PolicySpec("periodic", 100), PolicySpec("gated", 100),
               PolicySpec("dspot", 100), PolicySpec("triggered", 100)):
        a = run_stream(x, sp, CUSUM(5.0)); b = run_stream(x2, sp, CUSUM(5.0))
        same = np.allclose(a["m"][:5000], b["m"][:5000], equal_nan=True)
        check(f"causal baseline: {sp.name}", same)
    # vector path == generic path
    for sp in (PolicySpec("static"), PolicySpec("rolling", 100), PolicySpec("periodic", 100, R=25)):
        x, on = gen_stream(np.random.default_rng(2), n=9000, onset=6500, delta=1.5)
        for det in (CUSUM(6.0), EWMA(3.0), ZScore(3.0)):
            a = run_stream(x, sp, det.__class__(det.thr)); b = run_stream(x, sp, det.__class__(det.thr), force_loop=True)
            check(f"vector==loop {sp.name}/{det.__class__.__name__}",
                  np.array_equal(a["flags"], b["flags"]) and a["retrains"] == b["retrains"])
    # IF: batch == per-bar (short run)
    try:
        v = np.random.default_rng(3).standard_normal(2300)
        v[2100:2150] += 4
        d1 = IForestDetector(0.01); d2 = IForestDetector(0.01)
        bat = d1.run_batch(v)
        per = np.array([d2.update(z)[0] for z in v])
        check("IsolationForest batch==per-bar", np.array_equal(bat, per), f"alarms={int(bat.sum())}")
    except ImportError:
        print("[SKIP] sklearn not installed")
    # gated behaviour: holds baseline during a persistent shift, rolling absorbs it
    x, on = gen_stream(np.random.default_rng(4), n=9000, onset=6000, delta=2.0)
    g = run_stream(x, PolicySpec("gated", 100, K=250), CUSUM(8.0))
    r = run_stream(x, PolicySpec("rolling", 100), CUSUM(8.0))
    check("gated holds baseline after alarm", abs(g["m"][on + 100] - g["m"][on - 1]) < 0.5
          and r["m"][on + 150] > 1.5, f"gated m={g['m'][on+100]:.2f} rolling m={r['m'][on+150]:.2f}")
    # theory
    check("CUSUM critical absorption p (d=1: ~0.44, d=2: ~0.66)",
          abs(cusum_crit_p(1) - 0.44) < 0.01 and abs(cusum_crit_p(2) - 0.66) < 0.02,
          f"{cusum_crit_p(1):.3f}, {cusum_crit_p(2):.3f}")
    check("0.5 sigma shift never detected by CUSUM(k=.5)", math.isinf(cusum_D(0.5, 250, 5.0)))
    check("D finite for d=1,W=250,h=5", math.isfinite(cusum_D(1.0, 250, 5.0)),
          f"D={cusum_D(1.0, 250, 5.0)}")
    # calibration (small, quick; real runs use n=300_000)
    t0 = time.time()
    for det in ("cusum", "ewma", "zscore"):
        c = calibrate(det, PolicySpec("rolling", 250), "iid", n=60_000, iters=12, tol=0.35)
        print(f"   calib {det}: thr={c['thr']:.3f} rate={c['rate']:.2e} ok={c['ok']}")
        ok_all &= c["ok"]
    print(f"   calibration time {time.time()-t0:.1f}s")
    print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", nargs=3, metavar=("DET", "POLICY", "NOISE"))
    ap.add_argument("--W", type=int, default=250)
    ap.add_argument("--K", type=int, default=250)
    ap.add_argument("--kind", default="level")
    ap.add_argument("--n", type=int, default=300_000)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.calibrate:
        d, p, nz = a.calibrate
        print(calibrate(d, PolicySpec(p, a.W, K=a.K), nz, a.kind, n=a.n))
    else:
        ap.print_help()