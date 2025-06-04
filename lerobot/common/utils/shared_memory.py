from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
import numbers
import time
from atomics import atomicview, MemoryOrder, UINT
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from queue import (Empty, Full)
from typing import Dict, List, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt


SharedMemoryLike = Union[str, SharedMemory]  # shared memory or name of shared memory
SharedT = TypeVar("SharedT", bound=np.generic)


@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedAtomicCounter:
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 size: int = 8  # 64bit int
                 ):
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0)  # initialize

    @property
    def buf(self):
        return self.shm.buf[:self.size]

    def load(self) -> int:
        with atomicview(buffer=self.buf, atype=UINT) as a:
            value = a.load(order=MemoryOrder.ACQUIRE)
        return value

    def store(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.store(value, order=MemoryOrder.RELEASE)

    def add(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.add(value, order=MemoryOrder.ACQ_REL)


class SharedNDArray(Generic[SharedT]):
    shm: SharedMemory
    # shape: Tuple[int, ...]  # is a property
    dtype: np.dtype
    lock: Optional[multiprocessing.synchronize.Lock]

    def __init__(
            self, shm: SharedMemoryLike, shape: Tuple[int, ...], dtype: npt.DTypeLike
    ):
        """Initialize a SharedNDArray object from existing shared memory, object shape, and dtype.
        To initialize a SharedNDArray object from a memory manager and data or shape, use the `from_array()
        or `from_shape()` classmethods.
        Parameters
        ----------
        shm
            `multiprocessing.shared_memory.SharedMemory` object or name for connecting to an existing block
            of shared memory (using SharedMemory constructor)
        shape
            Shape of the NumPy array to be represented in the shared memory
        dtype
            Data type for the NumPy array to be represented in shared memory. Any valid argument for
            `np.dtype` may be used as it will be converted to an actual `dtype` object.
        lock : bool, optional
            If True, create a multiprocessing.Lock object accessible with the `.lock` attribute, by default
            False.  If passing the `SharedNDArray` as an argument to a `multiprocessing.Pool` function this
            should not be used -- see this comment to a Stack Overflow question about `multiprocessing.Lock`:
            https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes#comment72803059_25558333
        Raises
        ------
        ValueError
            The SharedMemory size (number of bytes) does not match the product of the shape and dtype
            itemsize.
        """
        if isinstance(shm, str):
            shm = SharedMemory(name=shm, create=False)
        dtype = np.dtype(dtype)  # Try to convert to dtype
        assert shm.size >= (dtype.itemsize * np.prod(shape))
        self.shm = shm
        self.dtype = dtype
        self._shape: Tuple[int, ...] = shape

    def __repr__(self):
        # Like numpy's ndarray repr
        cls_name = self.__class__.__name__
        nspaces = len(cls_name) + 1
        array_repr = str(self.get())
        array_repr = array_repr.replace("\n", "\n" + " " * nspaces)
        return f"{cls_name}({array_repr}, dtype={self.dtype})"

    @classmethod
    def create_from_array(
            cls, mem_mgr: SharedMemoryManager, arr: npt.NDArray[SharedT]
    ) -> SharedNDArray[SharedT]:
        """Create a SharedNDArray from a SharedMemoryManager and an existing numpy array.
        Parameters
        ----------
        mem_mgr
            Running `multiprocessing.managers.SharedMemoryManager` instance from which to create the
            SharedMemory for the SharedNDArray
        arr
            NumPy `ndarray` object to copy into the created SharedNDArray upon initialization.
        """
        # Simply use from_shape() to create the SharedNDArray and copy the data into it.
        shared_arr = cls.create_from_shape(mem_mgr, arr.shape, arr.dtype)
        shared_arr.get()[:] = arr[:]
        return shared_arr

    @classmethod
    def create_from_shape(
            cls, mem_mgr: SharedMemoryManager, shape: Tuple, dtype: npt.DTypeLike) -> SharedNDArray:
        """Create a SharedNDArray directly from a SharedMemoryManager
        Parameters
        ----------
        mem_mgr
            SharedMemoryManager instance that has been started
        shape
            Shape of the array
        dtype
            Data type for the NumPy array to be represented in shared memory. Any valid argument for
            `np.dtype` may be used as it will be converted to an actual `dtype` object.
        """
        dtype = np.dtype(dtype)  # Convert to dtype if possible
        shm = mem_mgr.SharedMemory(np.prod(shape) * dtype.itemsize)
        return cls(shm=shm, shape=shape, dtype=dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def get(self) -> npt.NDArray[SharedT]:
        """Get a numpy array with access to the shared memory"""
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def __del__(self):
        self.shm.close()


class SharedMemoryQueue:
    """
    A Lock-Free FIFO Shared Memory Data Structure.
    Stores a sequence of dict of numpy arrays.
    """

    class CallbackGuard:
        def __init__(self, callback, data):
            self.callback = callback
            self.data = data

        def __enter__(self):
            return self.data

        def __exit__(self, type, value, traceback):
            self.callback()
            del self.data
            del self.callback

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 array_specs: List[ArraySpec],
                 buffer_size: int
                 ):

        # create atomic counter
        write_counter = SharedAtomicCounter(shm_manager)
        read_counter = SharedAtomicCounter(shm_manager)

        # allocate shared memory
        shared_arrays = dict()
        for spec in array_specs:
            key = spec.name
            assert key not in shared_arrays
            array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype)
            shared_arrays[key] = array

        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.write_counter = write_counter
        self.read_counter = read_counter
        self.shared_arrays = shared_arrays

    @classmethod
    def create_from_examples(cls,
                             shm_manager: SharedMemoryManager,
                             examples: Dict[str, Union[np.ndarray, numbers.Number]],
                             buffer_size: int
                             ):
        specs = list()
        for key, value in examples.items():
            shape = None
            dtype = None
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                assert dtype != np.dtype('O')
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f'Unsupported type {type(value)}')

            spec = ArraySpec(
                name=key,
                shape=shape,
                dtype=dtype
            )
            specs.append(spec)

        obj = cls(
            shm_manager=shm_manager,
            array_specs=specs,
            buffer_size=buffer_size
        )
        return obj

    def qsize(self):
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        return n_data

    def empty(self):
        n_data = self.qsize()
        return n_data <= 0

    def clear(self):
        self.read_counter.store(self.write_counter.load())

    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]]):
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()

        next_idx = write_count % self.buffer_size

        # write to shared memory
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        # update idx
        self.write_counter.add(1)

    def get(self, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        if out is None:
            out = self._allocate_empty()

        next_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[next_idx])

        # update idx
        self.read_counter.add(1)
        return out

    def get_next_view(self) -> Dict[str, np.ndarray]:
        """
        Get reference to the next element to write
        for zero-copy writing
        """
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()

        next_idx = write_count % self.buffer_size
        out = dict()
        for key, value in self.shared_arrays.items():
            arr = value.get()
            out[key] = arr[next_idx]

        return out

    def put_next_view(self, data: Dict[str, Union[np.ndarray, numbers.Number]]):
        """
        Used in conjunction with get_next_view
        for zero-copy writing
        """
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()

        next_idx = write_count % self.buffer_size
        # write to shared memory
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                # assumed already written to the array
                pass
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        # update idx
        self.write_counter.add(1)

    def get_view(self) -> CallbackGuard:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        next_idx = read_count % self.buffer_size
        data = dict()
        for key, value in self.shared_arrays.items():
            arr = value.get()
            data[key] = arr[next_idx]

        return self.CallbackGuard(
            callback=lambda: self.read_counter.add(1),
            data=data)

    def get_k(self, k, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()
        assert k <= n_data

        out = self._get_k_impl(k, read_count, out=out)
        self.read_counter.add(k)
        return out

    def get_all(self, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        out = self._get_k_impl(n_data, read_count, out=out)
        self.read_counter.add(n_data)
        return out

    def _get_k_impl(self, k, read_count, out=None) -> Dict[str, np.ndarray]:
        if out is None:
            out = self._allocate_empty(k)

        curr_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            start = curr_idx
            end = min(start + k, self.buffer_size)
            target_start = 0
            target_end = (end - start)
            target[target_start: target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                # wrap around
                start = 0
                end = start + remainder
                target_start = target_end
                target_end = k
                target[target_start: target_end] = arr[start:end]

        return out

    def _allocate_empty(self, k=None):
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(
                shape=shape, dtype=spec.dtype)
        return result


class SharedMemoryRingBuffer:
    """
    A Lock-Free FILO Shared Memory Data Structure.
    Stores a sequence of dict of numpy arrays.
    """

    def __init__(self,
            shm_manager: SharedMemoryManager,
            array_specs: List[ArraySpec],
            get_max_k: int,
            get_time_budget: float,
            put_desired_frequency: float,
            safety_margin: float=1.5
        ):
        """
        shm_manager: Manages the life cycle of share memories
            across processes. Remember to run .start() before passing.
        array_specs: Name, shape and type of arrays for a single time step.
        get_max_k: The maxmum number of items can be queried at once.
        get_time_budget: The maxmum amount of time spent copying data from
            shared memory to local memory. Increase this number for larger arrays.
        put_desired_frequency: The maximum frequency that .put() can be called.
            This influces the buffer size.
        """

        # create atomic counter
        counter = SharedAtomicCounter(shm_manager)

        # compute buffer size
        # At any given moment, the past get_max_k items should never
        # be touched (to be read freely). Assuming the reading is reading
        # these k items, which takes maximum of get_time_budget seconds,
        # we need enough empty slots to make sure put_desired_frequency Hz
        # of put can be sustaied.
        buffer_size = int(np.ceil(
            put_desired_frequency * get_time_budget
            * safety_margin)) + get_max_k

        # allocate shared memory
        shared_arrays = dict()
        for spec in array_specs:
            key = spec.name
            assert key not in shared_arrays
            array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype)
            shared_arrays[key] = array

        # allocate timestamp array
        timestamp_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager,
            shape=(buffer_size,),
            dtype=np.float64)
        timestamp_array.get()[:] = -np.inf

        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.counter = counter
        self.shared_arrays = shared_arrays
        self.timestamp_array = timestamp_array
        self.get_time_budget = get_time_budget
        self.get_max_k = get_max_k
        self.put_desired_frequency = put_desired_frequency


    @property
    def count(self):
        return self.counter.load()

    @classmethod
    def create_from_examples(cls,
            shm_manager: SharedMemoryManager,
            examples: Dict[str, Union[np.ndarray, numbers.Number]],
            get_max_k: int=32,
            get_time_budget: float=0.01,
            put_desired_frequency: float=60
            ):
        specs = list()
        for key, value in examples.items():
            shape = None
            dtype = None
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                assert dtype != np.dtype('O')
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f'Unsupported type {type(value)}')

            spec = ArraySpec(
                name=key,
                shape=shape,
                dtype=dtype
            )
            specs.append(spec)

        obj = cls(
            shm_manager=shm_manager,
            array_specs=specs,
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_desired_frequency
            )
        return obj

    def clear(self):
        self.counter.store(0)

    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]], wait: bool=True):
        count = self.counter.load()
        next_idx = count % self.buffer_size
        # Make sure the next self.get_max_k elements in the ring buffer have at least
        # self.get_time_budget seconds untouched after written, so that
        # get_last_k can safely read k elements from any count location.
        # Sanity check: when get_max_k == 1, the element pointed by next_idx
        # should be rewritten at minimum self.get_time_budget seconds later.
        timestamp_lookahead_idx = (next_idx + self.get_max_k - 1) % self.buffer_size
        old_timestamp = self.timestamp_array.get()[timestamp_lookahead_idx]
        t = time.monotonic()
        if (t - old_timestamp) < self.get_time_budget:
            deltat = t - old_timestamp
            if wait:
                # sleep the remaining time to be safe
                time.sleep(self.get_time_budget - deltat)
            else:
                # throw an error
                past_iters = self.buffer_size - self.get_max_k
                hz = past_iters / deltat
                raise TimeoutError(
                    'Put executed too fast {}items/{:.4f}s ~= {}Hz'.format(
                        past_iters, deltat,hz))

        # write to shared memory
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        # update timestamp
        self.timestamp_array.get()[next_idx] = time.monotonic()
        self.counter.add(1)

    def _allocate_empty(self, k=None):
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(
                shape=shape, dtype=spec.dtype)
        return result

    def get(self, out=None) -> Dict[str, np.ndarray]:
        if out is None:
            out = self._allocate_empty()
        start_time = time.monotonic()
        count = self.counter.load()
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[curr_idx])
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f'Get time out {dt} vs {self.get_time_budget}')
        return out

    def get_last_k(self, k:int, out=None) -> Dict[str, np.ndarray]:
        assert k <= self.get_max_k
        if out is None:
            out = self._allocate_empty(k)
        start_time = time.monotonic()
        count = self.counter.load()
        assert k <= count
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            end = curr_idx + 1
            start = max(0, end - k)
            target_end = k
            target_start = target_end - (end - start)
            target[target_start: target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                # wrap around
                end = self.buffer_size
                start = end - remainder
                target_start = 0
                target_end = end - start
                target[target_start: target_end] = arr[start:end]
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f'Get time out {dt} vs {self.get_time_budget}')
        return out

    def get_all(self) -> Dict[str, np.ndarray]:
        k = min(self.count, self.get_max_k)
        return self.get_last_k(k=k)
