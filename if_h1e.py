"""
if_h1e.py -- H1e (Isolation Forest): paired one-sided Wilcoxon on alert coverage, adaptive vs frozen forest,
Holm over the 6 comparisons (3 training windows x 2 shift sizes) against the matched static forest; the unmatched
reference (static01) is a sensitivity analysis with its own Holm over 6. Also the non-inferiority readout:
P(detect) of the adaptive forest not lower than the frozen forest's by more than 0.10.

    python if_h1e.py if_out/if_events.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main(path):
    ev = pd.read_csv(path)
    for ref, label in (("static", "matched (primary)"), ("static01", "unmatched (sensitivity)")):
        rows = []
        for sh in sorted(ev["shift"].unique()):
            for tr in (500, 1000, 1500):
                a = ev[(ev["shift"] == sh) & (ev["variant"] == f"adaptive|{tr}")].set_index("seed")
                s = ev[(ev["shift"] == sh) & (ev["variant"] == f"{ref}|{tr}")].set_index("seed")
                c = a.index.intersection(s.index)
                a, s = a.loc[c], s.loc[c]
                try:
                    p = float(wilcoxon(a["coverage"], s["coverage"], alternative="less", zero_method="zsplit").pvalue)
                except Exception:
                    p = 1.0
                rows.append(dict(shift=sh, train=tr, n=len(c), cov_adaptive=a["coverage"].mean(),
                                 cov_static=s["coverage"].mean(), ratio=a["coverage"].mean() / s["coverage"].mean(),
                                 p=p, dP_detect=a["detect"].mean() - s["detect"].mean()))
        t = pd.DataFrame(rows)
        t["p_holm"] = holm(t["p"].values)
        t["H1e_cell"] = (t["ratio"] <= 0.5) & (t["p_holm"] < 0.05)
        t["detect_noninferior"] = t["dP_detect"] >= -0.10
        pd.set_option("display.width", 200)
        print(f"\nH1e, {label}, Holm over {len(t)}")
        print(t.round(4).to_string(index=False))
        print(f"H1e [{label}]: {'PASS' if t['H1e_cell'].all() else 'FAIL'} ({int(t['H1e_cell'].sum())}/{len(t)} cells; "
              f"detection non-inferior in {int(t['detect_noninferior'].sum())}/{len(t)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "if_out/if_events.csv")