"""Collection of the core mathematical operators used throughout the code base."""

from collections.abc import Callable, Iterable
import math

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply x by y"""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Add x and y"""
    return x + y


def neg(x: float) -> float:
    """Negates x"""
    return -1 * x


def lt(x: float, y: float) -> float:
    """Checks if x is less than y"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check equality of x and y"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Max of x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if inputs are close"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Computes standard sigmoid function"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Computes standard relu"""
    return 0.0 if x <= 0 else x


def log(x: float) -> float:
    """Computes natural log of x"""
    return math.log(x)


def exp(x: float) -> float:
    """Computes e**x"""
    return math.exp(x)


def log_back(x: float, c: float) -> float:
    """Computes derivative of c*log(x) w.r.t. x"""
    return c * 1.0 / x


def inv(x: float) -> float:
    """Computes inverse of x"""
    return 1.0 / x


def inv_back(x: float, c: float) -> float:
    """Computes derivative of c * x**-1 w.r.t. x"""
    return -1 * c * x**-2


def relu_back(x: float, c: float) -> float:
    """Computes derivative of c*relu(x) w.r.t. x"""
    return 0.0 if x <= 0 else c


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable) -> Callable[[Iterable], list]:
    """Higher-order function that applies a given function to each element of
    an iterable.
    """

    def _map(xs: Iterable) -> list:
        return [fn(x) for x in xs]

    return _map


def zipWith(fn: Callable) -> Callable[[Iterable, Iterable], list]:
    """Higher-order function that combines elements from two iterables using
    a given function
    """

    def _zipWith(xs: Iterable, ys: Iterable) -> list:
        return [fn(x, y) for x, y in zip(xs, ys)]

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using
    a given function. Returns 0.0 if the iterable is empty.
    """

    def _reduce(xs: Iterable[float]) -> float:
        val = start
        for x in xs:
            val = fn(val, x)
        return val

    return _reduce


def negList(xs: Iterable[float]) -> list:
    """Negate a list using map"""
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> list:
    """Add two lists together using zipWith"""
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list using reduce"""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Take the product of a list using reduce"""
    return reduce(mul, 1.0)(xs)
