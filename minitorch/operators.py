"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# Implementation of a prelude of elementary functions.

def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y

def id(x: float) -> float:
    """$f(x) = x$"""
    return x

def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y

def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x

def lt(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is less than y else 0.0"""
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is equal to y else 0.0"""
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    """$f(x) =$ x if x is greater than y else y"""
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    """$f(x) = |x - y| < 1e-2$"""
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)

def relu(x: float) -> float:
    return max(0.0, x)

EPS = 1e-6

def log(x: float) -> float:
    return math.log(x + EPS)

def exp(x: float) -> float:
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    return d / (x + EPS)

def inv(x: float) -> float:
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    return -d / (x ** 2)

def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0

# Higher-order functions

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def apply_map(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return apply_map

def negList(ls: Iterable[float]) -> Iterable[float]:
    return list(map(neg)(ls))

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def apply_zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return apply_zipWith

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)

def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    def apply_reduce(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    return apply_reduce

def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)

def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)
