#!/usr/bin/env python3
"""numba average-linkage, ported from Nicholas0228/HiT-Released hnet/linkage.py.

BSD-3-Clause original (SciPy nn_chain), adapted for our local reference.
This replaces scipy's Python-level nn_chain call with a JIT-compiled kernel,
so the routing cost can be compared fairly with the released implementation.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True)
def average_linkage(similarity):
    n = similarity.shape[0]
    distances = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            distances[i, j] = max(0., 1. - np.float64(similarity[i, j]))
    size = np.ones(n, dtype=np.int64)
    chain = np.empty(n, dtype=np.int64)
    chain_length = 0
    result = np.empty((n - 1, 4), dtype=np.float64)
    for step in range(n - 1):
        if chain_length == 0:
            for i in range(n):
                if size[i] > 0:
                    chain[0] = i
                    break
            chain_length = 1
        y = 0
        while True:
            x = chain[chain_length - 1]
            nearest = np.inf
            if chain_length > 1:
                y = chain[chain_length - 2]
                nearest = distances[x, y]
            for i in range(n):
                if size[i] and i != x and distances[x, i] < nearest:
                    nearest = distances[x, i]
                    y = i
            if chain_length > 1 and y == chain[chain_length - 2]:
                break
            chain[chain_length] = y
            chain_length += 1
        chain_length -= 2
        if x > y:
            x, y = y, x
        nx, ny = size[x], size[y]
        result[step, 0] = x
        result[step, 1] = y
        result[step, 2] = nearest
        result[step, 3] = nx + ny
        size[x], size[y] = 0, nx + ny
        for i in range(n):
            if size[i] and i != y:
                updated = (nx * distances[i, x] + ny * distances[i, y]) / (nx + ny)
                distances[i, y] = updated
                distances[y, i] = updated
    order = np.argsort(result[:, 2], kind="mergesort")
    result = result[order]
    parent = np.arange(2 * n - 1, dtype=np.int64)
    counts = np.ones(2 * n - 1, dtype=np.int64)
    for step in range(n - 1):
        x, y = int(result[step, 0]), int(result[step, 1])
        while parent[x] != x:
            x = parent[x]
        while parent[y] != y:
            y = parent[y]
        result[step, 0], result[step, 1] = min(x, y), max(x, y)
        counts[n + step] = counts[x] + counts[y]
        result[step, 3] = counts[n + step]
        parent[x], parent[y] = n + step, n + step
    return result


@njit(cache=True, nogil=True, parallel=True)
def batched_linkage(similarities):
    B, n, _ = similarities.shape
    out = np.empty((B, n - 1, 4), dtype=np.float64)
    for b in prange(B):
        out[b] = average_linkage(similarities[b])
    return out


@njit(cache=True, nogil=True)
def tree_metadata(children, scores, threshold, n):
    parent = np.arange(2 * n - 1, dtype=np.int64)
    accepted = 0
    for m in range(n - 1):
        if scores[m] >= threshold:
            parent[children[m, 0]] = n + m
            parent[children[m, 1]] = n + m
            accepted += 1
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = i
        while parent[r] != r:
            r = parent[r]
        roots[i] = r
    lookup = np.full(2 * n - 1, -1, dtype=np.int64)
    groups = np.empty(n, dtype=np.int64)
    K = 0
    for i in range(n):
        if lookup[roots[i]] < 0:
            lookup[roots[i]] = K
            K += 1
        groups[i] = lookup[roots[i]]
    order = np.empty(n, dtype=np.int64)
    start = np.zeros(2 * n - 1, dtype=np.int64)
    end = np.zeros(2 * n - 1, dtype=np.int64)
    stack = np.empty(2 * n, dtype=np.int64)
    stack[0] = 2 * n - 2
    top, cursor = 1, 0
    while top:
        top -= 1
        node = stack[top]
        if node < n:
            start[node] = cursor; end[node] = cursor + 1
            order[cursor] = node; cursor += 1
        else:
            stack[top] = children[node - n, 1]
            stack[top + 1] = children[node - n, 0]
            top += 2
    for m in range(n - 1):
        l, r = children[m, 0], children[m, 1]
        start[n + m] = min(start[l], start[r])
        end[n + m] = max(end[l], end[r])
    return groups, K, order, start[:n-1], end[:n-1], start[n:2*n-1], end[n:2*n-1]


@njit(cache=True, nogil=True, parallel=True)
def batched_metadata(trees, threshold, n):
    B = trees.shape[0]
    groups = np.empty((B, n), dtype=np.int64)
    counts = np.empty(B, dtype=np.int64)
    order = np.empty((B, n), dtype=np.int64)
    iv = np.empty((B, max(n-1, 0), 4), dtype=np.int64)
    scores = 1. - trees[:, :, 2]
    for b in prange(B):
        g, K, o, s0, s1, s2, s3 = tree_metadata(
            trees[b, :, :2].astype(np.int64), scores[b], threshold, n)
        groups[b] = g; counts[b] = K; order[b] = o
        if n > 1:
            iv[b, :, 0] = s0; iv[b, :, 1] = s1; iv[b, :, 2] = s2; iv[b, :, 3] = s3
    return groups, counts, order, iv, scores
