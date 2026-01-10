from typing import Iterable


def set_jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    set_a = set(a)
    set_b = set(b)

    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    return intersection / union
