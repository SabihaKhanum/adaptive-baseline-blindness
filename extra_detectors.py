"""
extra_detectors.py -- addendum A1 to the protocol: two more detectors under the same baseline policies.

Uses sim.py (same generator, calibration rule, episode rule and metrics). Detectors, all two-sided, on u = (x - m) / s
where (m, s) is the baseline of the policy in force:
    ewma      z_t = lam * u_t + (1 - lam) * z_{t-1}, lam = 0.1; alarm when |z_t| > h
    ma20      moving-average z-score: stat = sqrt(L) * |mean of the last L = 20 values of u|; alarm when stat > h
    cusum     the primary detector, re-implemented here ONLY as a consistency check against sim.py
Policies: static, rolling(W), periodic(W, R = W/4; statistic reset at each recompute). No gated policy.
h is calibrated per (detector, policy, W) to 1 alarm episode per 1,000 in-control bars (sim.TARGET_ALARM),
on the same calibration streams and the same check streams as the primary study.

    python extra_detectors.py test
    python extra_detectors.py calibrate --out cal_extra.json
    python extra_detectors.py run --cal cal_extra.json --seeds 1-30   --out extra_dev
    python extra_detectors.py run --cal cal_extra.json --seeds 101-300 --out extra_test
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

import sim

DETS = {"cusum": 0, "ewma": 1, "ma20": 2}
LAM = 0.1
L_MA = 20
POL = {"static": 0, "rolling": 1, "periodic": 2}
BOUNDS = {"cusum": (0.3, 3000.0), "ewma": (0.02, 50.0), "ma20": (0.3, 500.0)}


@njit(cache=True)
def run_x(x, pol, W, R, det, lam, k, h, warm, alarm, m_out, s_out, L):
    n = len(x)
    cs1 = np.zeros(n + 1)
    cs2 = np.zeros(n + 1)
    for i in range(n):
        cs1[i + 1] = cs1[i] + x[i]
        cs2[i + 1] = cs2[i] + x[i] * x[i]
    m = 0.0
    s = 1.0
    Sp = 0.0
    Sm = 0.0
    z = 0.0
    ubuf = np.zeros(n)
    rs = 0.0
    for t in range(n):
        if t < warm:
            m_out[t] = 0.0
            s_out[t] = 1.0
            continue
        if t == warm:
            m, s = sim._wstats(cs1, cs2, 0, warm)
        else:
            if pol == 1:
                m, s = sim._wstats(cs1, cs2, max(t - W, 0), t)
            elif pol == 2:
                if t % R == 0:
                    m, s = sim._wstats(cs1, cs2, max(t - W, 0), t)
                    Sp = 0.0
                    Sm = 0.0
                    z = 0.0
        m_out[t] = m
        s_out[t] = s
        u = (x[t] - m) / s
        if det == 0:
            Sp = max(0.0, Sp + u - k)
            Sm = max(0.0, Sm - u - k)
            stat = max(Sp, Sm)
        elif det == 1:
            z = lam * u + (1.0 - lam) * z
            stat = abs(z)
        else:
            ubuf[t] = u
            rs += u
            if t - L >= warm:
                rs -= ubuf[t - L]
            cnt = min(L, t - warm + 1)
            stat = math.sqrt(cnt) * abs(rs / cnt)
        alarm[t] = 1 if stat > h else 0


def run_det(x, det, pol, W, h):
    n = len(x)
    alarm = np.zeros(n, np.uint8)
    m = np.zeros(n)
    s = np.ones(n)
    run_x(x, POL[pol], int(W), max(1, int(W) // 4), DETS[det], LAM, sim.K_LEVEL, float(h), sim.WARM, alarm, m, s, L_MA)
    return alarm, m, s


def rate(xs, det, pol, W, h):
    tot = bars = 0
    for x in xs:
        alarm, *_ = run_det(x, det, pol, W, h)
        tot += sim.count_eps(alarm, sim.EP_START, sim.GAP)
        bars += len(x) - sim.EP_START
    return 1000.0 * tot / bars


def calibrate(det, pol, W, xs_cal, xs_chk):
    lo, hi = BOUNDS[det]
    h = sim.bisect_threshold(lambda v: rate(xs_cal, det, pol, W, v), sim.TARGET_ALARM, lo=lo, hi=hi)
    rc, rk = rate(xs_cal, det, pol, W, h), rate(xs_chk, det, pol, W, h)
    return dict(det=det, pol=pol, W=W, h=h, rate_cal=rc, rate_chk=rk,
                ok=bool(0.8 <= rc <= 1.2 and 0.8 <= rk <= 1.2))


def cells():
    out = [("static", 0)]
    for pol in ("rolling", "periodic"):
        out += [(pol, W) for W in sim.WGRID]
    return out


def cmd_calibrate(a):
    xs_cal = sim.cal_streams(a.arm, sim.CAL_SEED)
    xs_chk = sim.cal_streams(a.arm, sim.CHECK_SEED)
    res = {}
    for det in a.dets:
        for pol, W in cells():
            c = calibrate(det, pol, W, xs_cal, xs_chk)
            res[f"{det}|{pol}|{W}"] = c
            print(f"{det:9s} {pol:9s} W={W:4d}  h={c['h']:.4g}  rate cal {c['rate_cal']:.2f}  chk {c['rate_chk']:.2f}  "
                  f"{'ok' if c['ok'] else 'OUT OF TOLERANCE'}", flush=True)
    json.dump({"arm": a.arm, "cells": res}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def cmd_run(a):
    cal = json.load(open(a.cal))["cells"]
    rows = []
    shifts = [("level", d) for d in a.deltas]
    for seed in sim.parse_seeds(a.seeds):
        for sh in shifts:
            x, tau, end = sim.make_stream(seed, a.arm, sh)
            for det in a.dets:
                for pol, W in cells():
                    h = cal[f"{det}|{pol}|{W}"]["h"]
                    alarm, m, s = run_det(x, det, pol, W, h)
                    rec = sim.event_metrics(alarm, m, s, tau, end, sh, None)
                    rec.update(seed=seed, shift=f"level{sh[1]}", det=det, pol=pol, W=W)
                    rows.append(rec)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "extra_events.csv"), index=False)
    pd.set_option("display.width", 220)
    for det in a.dets:
        d = df[df["det"] == det]
        fam = sorted(d["shift"].unique())
        t = sim.shortened_table(d, fam)
        t.insert(0, "det", det)
        t.to_csv(os.path.join(a.out, f"extra_cov_{det}.csv"), index=False)
        print(f"\n== {det}: alert coverage, policy / static (one-sided paired Wilcoxon, Holm over {len(t)}) ==")
        print(t.round(4).to_string(index=False))
        det_tab = d.groupby(["shift", "pol", "W"])[["detect", "blind", "coverage"]].mean().round(3)
        print(f"\n{det}: P(detect), P(blind), coverage")
        print(det_tab.to_string())
    print("\nwrote", a.out)


def run_tests():
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("[ok]   " if cond else "[FAIL] ") + name + (f"  ({info})" if info else ""))

    # 1) the cusum branch here must equal sim.run for the three policies it implements
    x, tau, end = sim.make_stream(3, "btclike", ("level", 1.0))
    for pol, W in (("static", 0), ("rolling", 250), ("periodic", 250)):
        a1, *_ = sim.run(x, pol, W, 0, sim.K_LEVEL, 12.0, sim.INF, 0)
        a2, *_ = run_det(x, "cusum", pol, W, 12.0)
        chk(f"cusum equals sim.run ({pol})", np.array_equal(a1, a2))
    # 2) ewma and ma20 behave: a +1 sd shift is flagged by the static baseline
    xi, taui, _ = sim.make_stream(3, "iid", ("level", 1.0))
    for det, h in (("ewma", 0.6), ("ma20", 3.0)):
        a, *_ = run_det(xi, det, "static", 0, h)
        tau = taui
        chk(f"{det}: more alarms after the shift than before", a[tau:tau + 500].sum() > a[sim.WARM:tau].sum() * 500 / max(1, tau - sim.WARM))
    # 3) calibration hits the target on a small set
    xs = sim.cal_streams("btclike", 999, 12)
    c = calibrate("ewma", "rolling", 250, xs, xs)
    chk("ewma calibration reaches ~1 per 1,000", 0.7 <= c["rate_cal"] <= 1.3, f"rate {c['rate_cal']:.2f}, h {c['h']:.3f}")
    c = calibrate("ma20", "static", 0, xs, xs)
    chk("ma20 calibration reaches ~1 per 1,000", 0.7 <= c["rate_cal"] <= 1.3, f"rate {c['rate_cal']:.2f}, h {c['h']:.3f}")
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    for name in ("calibrate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--arm", default="btclike")
        p.add_argument("--dets", nargs="*", default=["ewma", "ma20"])
        if name == "calibrate":
            p.add_argument("--out", default="cal_extra.json")
        else:
            p.add_argument("--cal", required=True)
            p.add_argument("--seeds", default="1-30")
            p.add_argument("--deltas", type=float, nargs="*", default=[1.0, 0.75])
            p.add_argument("--out", default="extra_out")
    a = ap.parse_args()
    if a.cmd == "test":
        sys.exit(0 if run_tests() else 1)
    {"calibrate": cmd_calibrate, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()