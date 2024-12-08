# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to create device functions."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator to create device functions."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """See `tensor_ops.py`"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # # TODO: this fast path might be slower than the slow path
        # shared_flag_fastpath = cuda.shared.array(1, dtype=numba.uint8)
        # if i == 0:
        #     shared_flag_fastpath[0] = 0
        # cuda.syncthreads()

        # if out_shape.size == in_shape.size:
        #     if i < out_shape.size:
        #         if out_shape[i] == in_shape[i] and out_strides[i] == in_strides[i]:
        #             shared_flag_fastpath[0] = 1
        #     cuda.syncthreads()

        #     if i < out_size:
        #         if shared_flag_fastpath[0] != 0:
        #             out[i] = fn(in_storage[i])
        #             return

        # # TODO: local fast path, but again might be slower
        # flag : numba.uint8 = 0
        # if out_shape.size == in_shape.size:
        #     for j in range(out_shape.size):
        #         # or is this a lot of global reads?
        #         if out_shape[j] == in_shape[j] and out_strides[j] == in_strides[j]:
        #             flag = 1
        #             break
        #     if flag != 0:
        #         if i < out_size:
        #             out[i] = fn(in_storage[i])
        #         return

        # Implement for Task 3.3.
        # basically the same as the cpu version
        if i < out_size:
            to_index(i, out_shape, out_index)

            broadcast_index(out_index, out_shape, in_shape, in_index)

            in_ordinal = index_to_position(in_index, in_strides)
            out_ordinal = index_to_position(out_index, out_strides)
            out[out_ordinal] = fn(in_storage[in_ordinal])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            # iterate over all possible out_indices
            to_index(i, out_shape, out_index)

            # get corresponding indices for a and b
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            a_ordinal = index_to_position(a_index, a_strides)
            b_ordinal = index_to_position(b_index, b_strides)
            out_ordinal = index_to_position(out_index, out_strides)
            out[out_ordinal] = fn(a_storage[a_ordinal], b_storage[b_ordinal])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if i < size:
        cache[pos] = a[i]
    elif i - pos < size:
        cache[pos] = 0.0

    cuda.syncthreads()

    # divide and conquer reduction
    # note block_dim = 2^5
    nxt_leaf_count = BLOCK_DIM // 2
    while nxt_leaf_count > 0:
        if pos < nxt_leaf_count:
            cache[pos] += cache[pos + int(nxt_leaf_count)]
        nxt_leaf_count //= 2
        cuda.syncthreads()

    if i < size and pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Sum practice function."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Note that the number of blocks = out_size

        reduce_size = a_shape[reduce_dim]

        # We can only sum so many things at a time
        # will need to move the shared memory
        # have each block work on a single out_pos
        # TODO Note that out_shape[reduce_dim] can be > 1 if reduce_size > 1024
        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)

            # TODO check if works for out_index[reduce_dim] > 1
            # TODO probably not, but maybe we don't need to fix it yet
            a_ordinal_base = index_to_position(out_index, a_strides)

            # copy the things we want to sum into shared memory
            # note that we know that reduce_size <= BLOCK_DIM
            if pos < reduce_size:
                cache[pos] = a_storage[a_ordinal_base + pos * a_strides[reduce_dim]]
            else:
                cache[pos] = reduce_value
            cuda.syncthreads()

            # Now, sum over shared memory as appropriate
            # divide and conquer reduction
            # note reduce_size may be odd... what do we do?
            # we dealt with it earlier by setting to reduce_value
            nxt_leaf_count = BLOCK_DIM // 2
            while nxt_leaf_count > 0:
                if pos < nxt_leaf_count:
                    cache[pos] = fn(cache[pos], cache[pos + nxt_leaf_count])
                nxt_leaf_count //= 2
                cuda.syncthreads()

            if pos == 0:
                out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    # BLOCK_DIM = 32
    # Implement for Task 3.3.
    # note size < 32 by assumption, so no need for tiling

    # TODO: check if there is more than one block
    # it seems that we only use one block...
    # block_idx = cuda.blockIdx.x
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # first, copy both a and b to shared memory
    # initialize the memory
    cacheA = cuda.shared.array((32, 32), numba.float64)
    cacheB = cuda.shared.array((32, 32), numba.float64)

    # copy in parallel
    if i < size and j < size:
        cacheA[i, j] = a[i * size + j]
        cacheB[i, j] = b[i * size + j]
    # synchronize threads
    cuda.syncthreads()

    # now, compute the dot product at each index
    # where each index is a thread
    if i < size and j < size:
        val = 0.0
        for k in range(size):
            val += cacheA[i, k] * cacheB[k, j]
        # write to global memory in parallel
        out[i * size + j] = val


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Matrix multiply practice function."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Sum-dimension size, for intuition
    sum_size = a_shape[-1]  # equal to b_shape[-2]

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    # TODO we assume that shape has length 3, i.e. only one batch dimension, fix this?

    # tile movement
    width_in_blocks = -(sum_size // -BLOCK_DIM)
    val = 0.0
    for K in range(width_in_blocks):
        # combine tile at (i, K) and (K, j)

        # for a, i index never changes, but j increments by BLOCK_DIM
        a_copy_i = i
        a_copy_j = K * BLOCK_DIM + pj

        # for b, j index never changes, but i increments by BLOCK_DIM
        b_copy_i = K * BLOCK_DIM + pi
        b_copy_j = j

        # now, copy both a and b tiles to shared memory, taking care iwth idices
        if a_copy_i < a_shape[-2] and a_copy_j < a_shape[-1]:
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride
                + a_copy_i * a_strides[-2]
                + a_copy_j * a_strides[-1]
            ]
        else:
            a_shared[pi, pj] = 0.0
        if b_copy_i < b_shape[-2] and b_copy_j < b_shape[-1]:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride
                + b_copy_i * b_strides[-2]
                + b_copy_j * b_strides[-1]
            ]
        else:
            b_shared[pi, pj] = 0.0
        # synchronize threads
        cuda.syncthreads()

        # now, compute the dot product at each index
        if i < out_shape[-2] and j < out_shape[-1]:
            for k in range(BLOCK_DIM):
                val += a_shared[pi, k] * b_shared[k, pj]
        # synchronize to not accidentally overwrite shared data
        cuda.syncthreads()

    # final global write
    ordin = batch * out_batch_stride + i * out_strides[-2] + j * out_strides[-1]
    # make sure its in bounds, else we get weird bugs
    if ordin < out_size and i < out_shape[-2] and j < out_shape[-1]:
        out[ordin] = val


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
