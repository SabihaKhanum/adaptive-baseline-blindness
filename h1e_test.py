import pandas as pd
from scipy.stats import wilcoxon
ev = pd.read_csv("if_out/if_events.csv")
for sh in sorted(ev["shift"].unique()):
    for tr in (500, 1000, 1500):
        for ref in ("static", "static01"):
            a = ev[(ev["shift"] == sh) & (ev.variant == f"adaptive|{tr}")].set_index("seed")["coverage"]
            s = ev[(ev["shift"] == sh) & (ev.variant == f"{ref}|{tr}")].set_index("seed")["coverage"]
            c = a.index.intersection(s.index)
            a, s = a[c], s[c]
            print(sh, tr, ref, round(a.mean() / s.mean(), 2),
                  round(wilcoxon(a, s, alternative="less", zero_method="zsplit").pvalue, 4))