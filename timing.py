import time, numpy as np
from sim import *

x, on = gen_stream(np.random.default_rng(1), delta=1.0)
for name in ["static", "rolling", "periodic", "dspot", "gated", "triggered"]:
    spec = PolicySpec(name, 250)
    t0 = time.time()
    run_stream(x, spec, CUSUM(5.8))
    print(f"{name:10s} {time.time() - t0:6.2f} s per stream")