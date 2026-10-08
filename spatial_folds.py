"""
spatial_folds.py - A deterministic group k-fold, so spatial cross-validation gives the SAME folds on every computer.

WHY: scikit-learn's GroupKFold has to decide which of several equally sized regions goes into which fold. That
tie-break differs between library and numpy versions, so the same data gave Senegal a spatial R2 of 0.783 on one
machine and 0.771 on another. Results in a paper must not depend on the software version.

RULE (fixed): sort the regions by number of clusters, largest first; break ties by region name (alphabetical);
give each region to the fold that currently holds the fewest clusters; if folds tie, use the lowest fold number.
The folds depend only on region names and sizes, never on row order, library version or computer.

USAGE:
    from spatial_folds import StableGroupKFold
    cross_val_score(model, X, y, cv=StableGroupKFold(5), groups=regions, scoring="r2")
"""

import numpy as np


class StableGroupKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def fold_of_group(self, groups):
        groups = np.asarray(groups)
        names, counts = np.unique(groups, return_counts=True)
        if len(names) < self.n_splits:
            raise ValueError(f"Need at least {self.n_splits} regions for {self.n_splits} folds, got {len(names)}.")
        order = sorted(range(len(names)), key=lambda i: (-int(counts[i]), str(names[i])))
        load = np.zeros(self.n_splits)
        fold_of = {}
        for i in order:
            k = int(np.argmin(load))
            load[k] += counts[i]
            fold_of[names[i]] = k
        return fold_of

    def split(self, X, y=None, groups=None):
        groups = np.asarray(groups)
        fold_of = self.fold_of_group(groups)
        fold = np.array([fold_of[g] for g in groups])
        for k in range(self.n_splits):
            yield np.where(fold != k)[0], np.where(fold == k)[0]
