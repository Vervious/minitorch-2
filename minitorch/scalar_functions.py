from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of scalar values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Adds two numbers."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for add."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Applies the log function."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for log."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiplies two numbers."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for mul."""
        a, b = ctx.saved_values
        # d(a*b)/da * dZ/d(a*b), d(a*b)/db * dZ/d(a*b) respectively
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Inverts a number."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inv."""
        (a,) = ctx.saved_values
        # d(1/a)/da * dZ/d(1/a) = -1 a^{-2} * dZ/d(1/a)
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negates a float."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for neg."""
        # d(-a)/da * dZ/d(-a) = -1 * dZ/d(-a)
        return -1 * d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1/(1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Applies the sigmoid function."""
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid."""
        (out,) = ctx.saved_values
        # let x = 1+e**-a
        # d(Sigmoid(a))/da * dZ/d(sigmoid(a))
        # = d(1/x)/da * dZ/d(1/x)
        # = dx/da * d(1/x)/dx *  dZ/d(1/x)
        # = dx/da * -1 x**-2 * dZ/d(1/x)
        # = -1*e**-a * -1 x**-2 * dZ/d(1/x)
        ## return operators.exp(-a) * x**-2 * d_output

        # Use something more standard with better numerical precision, perhaps
        # Note e**-a = (1 - S(a)) / S(a)
        # = S(a)**2 * e**-a = S(a) * (1 - S(a))
        return out * (1 - out) * d_output


class ReLU(ScalarFunction):
    """Relu function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Applies the relu function."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for relu."""
        (a,) = ctx.saved_values
        # d(Relu(a))/da * dZ/d(Relu(a)) = [0 if a < 0, 1 else] * dZ/d(-a)
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Applies the exp function."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exp."""
        (a,) = ctx.saved_values
        # d(exp(a))/da * dZ/d(exp(a)) = e^a * dZ/d(exp(a))
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1.0 if x < y else 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Checks if a is less than b."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for lt."""
        # d(lt(a, b))/da * dZ/d(lt(a, b)), d(lt(a, b))/db * dZ/d(lt(a, b))
        # for fixed b (resp a), a step function (0 or 1)
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality function $f(x, y) = 1.0 if x == y else 0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Checks if a is equal to b."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for eq."""
        # d(eq(a, b))/da * dZ/d(eq(a, b)), d(eq(a, b))/db * dZ/d(eq(a, b))
        # Not sure how to implement backwards for a step function, no derivative
        return 0.0, 0.0
