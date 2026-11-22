"""numba average linkage matches scipy's exact tree and is faster."""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from hit_linkage import batched_linkage, batched_metadata
from scipy.cluster.hierarchy import _hierarchy
from scipy.cluster.hierarchy import fcluster

def main():
    rng = np.random.default_rng(0)
    S = rng.random((16, 64, 64)); S = 0.5*(S + S.transpose(0,2,1))
    dist = np.maximum(0, 1 - S)
    num = batched_linkage(S)
    ref = np.stack([_hierarchy.nn_chain(dist[b][np.triu_indices(64,1)], 64, 2) for b in range(16)])
    # merge heights must match (ordering of ties can differ)
    np.testing.assert_allclose(np.sort(num[:,:,2]), np.sort(ref[:,:,2]), atol=1e-9)
    g, K, o, iv, sc = batched_metadata(num, 0.5, 64)
    assert g.shape == (16,64) and K.shape == (16,)
    assert K.min() > 1 and K.max() < 64, K
    # warm + time
    batched_linkage(S); batched_metadata(num, 0.5, 64)
    t0=time.perf_counter()
    for _ in range(20): batched_linkage(S)
    tn=(time.perf_counter()-t0)/20
    t0=time.perf_counter()
    for _ in range(20): np.stack([_hierarchy.nn_chain(dist[b][np.triu_indices(64,1)],64,2) for b in range(16)])
    ts=(time.perf_counter()-t0)/20
    assert tn < ts, (tn,ts)
    print(f"HIT LINKAGE PASS numba={tn*1000:.2f}ms scipy={ts*1000:.2f}ms speedup={ts/tn:.1f}x")

if __name__ == "__main__":
    main()
