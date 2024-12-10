from typing import Tuple

from .tensor import Tensor
from .tensor_functions import rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # NOTE: implemented for task 4.3
    broken = input.contiguous().view(
        batch, channel, height // kh, kh, width // kw, kw
    )  # thank you copilot
    fixed = broken.permute(0, 1, 2, 4, 3, 5)  # makes non-contiguous
    return (
        fixed.contiguous().view(batch, channel, height // kh, width // kw, kw * kh),
        height // kh,
        width // kw,
    )


# NOTE: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    """Apply a 2D average pooling over an input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel_size: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # NOTE: implemented for task 4.3
    batch, channel, _, _ = input.shape
    tile_input, new_height, new_width = tile(input, kernel_size)

    return (
        tile_input.mean(dim=4).contiguous().view(batch, channel, new_height, new_width)
    )


def max(input: Tensor, dim: int | None = None) -> Tensor:
    """Apply max reduction over all elements of the tensor.

    Args:
    ----
        input: tensor to reduce
        dim: Optional dimension to reduce on

    Returns:
    -------
        Tensor of size 1 with the maximum value of the input tensor.

    """
    # NOTE: implemented for task 4.4

    if dim is not None:
        newDims = []
        for d in range(len(input.shape)):
            if d == dim:
                continue
            newDims.append(input.shape[d])
        return input.max(dim=dim).view(*newDims)
    return input.max(dim=dim).view(1)


def softmax(input: Tensor, dim: int | None = None) -> Tensor:
    """Apply a softmax operator.

    Args:
    ----
        input: input tensor
        dim: Optional dimension to reduce on

    Returns:
    -------
        Tensor of the same shape as the input tensor.

    """
    # NOTE: implemented for task 4.4
    exponentiated = input.exp()
    total = exponentiated.sum(dim=dim)
    return exponentiated / total


def logsoftmax(input: Tensor, dim: int | None = None) -> Tensor:
    """Apply a log softmax operator.

    Args:
    ----
        input: input tensor
        dim: Optional dimension to reduce on

    Returns:
    -------
        Tensor of the same shape as the input tensor.

    """
    # NOTE: implemented for task 4.4
    c = input.max(dim=dim)
    exponentiated = (input - c).exp()
    total = exponentiated.sum(dim=dim)
    return input - (c + total.log())


def maxpool2d(input: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    """Apply a 2D max pooling over an input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel_size: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # NOTE: implemented for task 4.4
    batch, channel, _, _ = input.shape
    tile_input, new_height, new_width = tile(input, kernel_size)

    # note that contiguous just makes a copy, which has backwards defined
    return (
        tile_input.max(dim=4).contiguous().view(batch, channel, new_height, new_width)
    )


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply a dropout operator.

    Args:
    ----
        input: input tensor
        p: probability of dropping
        ignore: ignore the dropout (shortcut for disabling)

    Returns:
    -------
        Tensor of the same shape as the input tensor.

    """
    # NOTE: implemented for task 4.4
    if ignore:
        return input
    mask = rand(input.shape) > p
    return input * mask
