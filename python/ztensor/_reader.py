"""Python Reader wrapping the Rust _Reader primitive."""

import numpy as np

from ._types import Tensor, TensorMetadata


class Reader:
    """Reader for tensor files (zTensor, SafeTensors, PyTorch, GGUF, NPZ, ONNX, HDF5).

    Automatically detects the file format based on file extension.
    Supports dict-like access: ``reader["tensor_name"]`` returns a NumPy array.
    """

    def __init__(self, path):
        from ._ztensor import _Reader

        self._inner = _Reader(path)

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def keys(self):
        """Returns a list of all tensor names."""
        return self._inner.keys()

    @property
    def format(self):
        """The detected file format."""
        return self._inner.format

    def __len__(self):
        return len(self._inner)

    def __contains__(self, name):
        return name in self._inner

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(self, name):
        """Returns metadata for a tensor (str → TensorMetadata, list → list[TensorMetadata])."""
        if isinstance(name, str):
            return self._build_metadata(name)
        elif isinstance(name, list):
            return [self._build_metadata(n) for n in name]
        else:
            raise TypeError("name must be str or list[str]")

    def _build_metadata(self, name):
        meta = self._inner._get_metadata(name)
        # Determine primary component dtype
        fmt = meta["format"]
        comps = meta["components"]
        if fmt == "dense":
            primary = comps.get("data")
        elif fmt in ("sparse_csr", "sparse_coo"):
            primary = comps.get("values")
        else:
            primary = next(iter(comps.values()), None) if comps else None
        dtype_str = primary["dtype"] if primary else "unknown"
        type_str = primary.get("type") if primary else None
        return TensorMetadata(
            name=name,
            dtype=dtype_str,
            shape=meta["shape"],
            format=fmt,
            type=type_str,
            components=comps,
            attributes=meta.get("attributes"),
        )

    # ------------------------------------------------------------------
    # read_numpy — dense tensors as NumPy arrays
    # ------------------------------------------------------------------

    def read_numpy(self, name, *, copy=False):
        """Read dense tensor(s) as NumPy arrays.

        Args:
            name: str → np.ndarray, list[str] → dict[str, np.ndarray]
            copy: If False (default), zero-copy mmap arrays when possible.
        """
        if isinstance(name, str):
            return self._read_numpy_single(name, copy)
        elif isinstance(name, list):
            return self._read_numpy_many(name, copy)
        else:
            raise TypeError("name must be str or list[str]")

    def _read_numpy_single(self, name, copy):
        buf, dtype_str, shape = self._inner._read_raw(name)
        return self._buf_to_numpy(buf, dtype_str, shape, copy)

    def _read_numpy_many(self, names, copy):
        # Collect metadata for offset sorting
        infos = []
        for n in names:
            meta = self._inner._get_metadata(n)
            comp = meta["components"].get("data")
            offset = comp["offset"] if comp else 0
            infos.append((n, offset))
        infos.sort(key=lambda x: x[1])

        result = {}
        for n, _ in infos:
            result[n] = self._read_numpy_single(n, copy)
        return result

    @staticmethod
    def _buf_to_numpy(buf, dtype_str, shape, copy):
        if dtype_str == "bfloat16":
            try:
                import ml_dtypes  # noqa: F401
            except ImportError:
                raise RuntimeError("bfloat16 tensors require the 'ml_dtypes' package")
            arr = buf.view(np.dtype("uint16"))
            bf16_dtype = np.dtype("bfloat16")
            arr = arr.view(bf16_dtype)
        else:
            arr = buf.view(np.dtype(dtype_str))

        if shape:
            arr = arr.reshape(shape)

        if copy:
            arr = arr.copy()
        return arr

    # ------------------------------------------------------------------
    # read_torch — dense tensors as torch.Tensors
    # ------------------------------------------------------------------

    def read_torch(self, name, *, copy=False, device="cpu"):
        """Read dense tensor(s) as torch.Tensor.

        Args:
            name: str → torch.Tensor, list[str] → dict[str, torch.Tensor]
            copy: If False (default), zero-copy via mmap + torch.frombuffer.
            device: Target device ("cpu", "cuda:0", etc.).
        """
        import torch

        if isinstance(name, str):
            return self._read_torch_single(name, copy, device, torch)
        elif isinstance(name, list):
            return self._read_torch_many(name, copy, device, torch)
        else:
            raise TypeError("name must be str or list[str]")

    def _read_torch_single(self, name, copy, device, torch):
        buf, dtype_str, shape = self._inner._read_raw(name)
        return self._buf_to_torch(buf, dtype_str, shape, copy, device, torch)

    def _read_torch_many(self, names, copy, device, torch):
        # Collect metadata for offset sorting
        infos = []
        for n in names:
            meta = self._inner._get_metadata(n)
            comp = meta["components"].get("data")
            offset = comp["offset"] if comp else 0
            infos.append((n, offset))
        infos.sort(key=lambda x: x[1])

        result = {}
        for n, _ in infos:
            result[n] = self._read_torch_single(n, copy, device, torch)
        return result

    _DTYPE_TO_TORCH = {
        "float64": "float64",
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "int64": "int64",
        "int32": "int32",
        "int16": "int16",
        "int8": "int8",
        "uint8": "uint8",
        "bool": "bool",
    }

    @staticmethod
    def _buf_to_torch(buf, dtype_str, shape, copy, device, torch):
        torch_attr = Reader._DTYPE_TO_TORCH.get(dtype_str)
        if torch_attr is None:
            raise RuntimeError(
                f"Unsupported dtype for torch: {dtype_str}. Use read_numpy() instead."
            )
        torch_dtype = getattr(torch, torch_attr)
        tensor = torch.frombuffer(buf, dtype=torch_dtype).reshape(shape)
        if copy:
            tensor = tensor.clone()
        if device != "cpu":
            tensor = tensor.to(device)
        return tensor

    # ------------------------------------------------------------------
    # read_into — copy tensor data into pre-allocated destinations
    # ------------------------------------------------------------------

    def read_into(self, name, dst=None):
        """Read tensor data directly into a pre-allocated array or tensor.

        Args:
            name: str for single tensor, or dict[str, dst] for batch (offset-sorted).
            dst: numpy.ndarray or torch.Tensor (any device). Must match tensor shape/dtype.
                 Required when name is a str.
        """
        if isinstance(name, dict):
            return self._read_into_many(name)
        if dst is None:
            raise TypeError("dst is required when name is a str")
        buf, dtype_str, shape = self._inner._read_raw(name)
        self._copy_into(buf, dtype_str, shape, dst)

    def _read_into_many(self, targets):
        infos = []
        for n in targets:
            meta = self._inner._get_metadata(n)
            comp = meta["components"].get("data")
            offset = comp["offset"] if comp else 0
            infos.append((n, offset))
        infos.sort(key=lambda x: x[1])

        for n, _ in infos:
            buf, dtype_str, shape = self._inner._read_raw(n)
            self._copy_into(buf, dtype_str, shape, targets[n])

    @staticmethod
    def _copy_into(buf, dtype_str, shape, dst):
        if isinstance(dst, np.ndarray):
            src = Reader._buf_to_numpy(buf, dtype_str, shape, copy=False)
            np.copyto(dst, src)
        else:
            import torch

            src = Reader._buf_to_torch(buf, dtype_str, shape, False, "cpu", torch)
            dst.copy_(src)

    # ------------------------------------------------------------------
    # read — full Tensor objects (all components, any format)
    # ------------------------------------------------------------------

    def read(self, name, *, copy=False):
        """Read tensor(s) as ztensor.Tensor objects.

        Args:
            name: str → single Tensor, list[str] → dict[str, Tensor]
            copy: If False (default), zero-copy mmap arrays when possible.
        """
        if isinstance(name, str):
            return self._read_tensor_single(name, copy)
        elif isinstance(name, list):
            return self._read_tensor_many(name, copy)
        else:
            raise TypeError("name must be str or list[str]")

    def _read_tensor_single(self, name, copy):
        meta = self._inner._get_metadata(name)
        fmt = meta["format"]
        shape = meta["shape"]
        comps_meta = meta["components"]

        # Determine primary dtype
        if fmt == "dense":
            primary = comps_meta.get("data")
        elif fmt in ("sparse_csr", "sparse_coo"):
            primary = comps_meta.get("values")
        else:
            primary = next(iter(comps_meta.values()), None) if comps_meta else None
        dtype_str = primary["dtype"] if primary else "unknown"
        type_str = primary.get("type") if primary else None

        # Read all components
        components = {}
        for comp_name, comp_info in comps_meta.items():
            buf, cdtype = self._inner._read_component_raw(name, comp_name)
            # For "data" component of dense tensors, reshape to tensor shape
            is_data_dense = comp_name == "data" and fmt == "dense"
            comp_shape = shape if is_data_dense else []
            arr = self._buf_to_numpy(buf, cdtype, comp_shape, copy)
            components[comp_name] = arr

        return Tensor(
            components,
            shape=shape,
            dtype=dtype_str,
            format=fmt,
            type=type_str,
            attributes=meta.get("attributes"),
        )

    def _read_tensor_many(self, names, copy):
        # Sort by minimum component offset
        infos = []
        for n in names:
            meta = self._inner._get_metadata(n)
            min_offset = min(
                (c["offset"] for c in meta["components"].values()),
                default=0,
            )
            infos.append((n, min_offset))
        infos.sort(key=lambda x: x[1])

        result = {}
        for n, _ in infos:
            result[n] = self._read_tensor_single(n, copy)
        return result

    # ------------------------------------------------------------------
    # read_tensors — batch read as numpy dict (legacy alias)
    # ------------------------------------------------------------------

    def read_tensors(self, names, *, copy=False):
        """Read multiple tensors as a dict of numpy arrays."""
        return self._read_numpy_many(names, copy)

    # ------------------------------------------------------------------
    # Dict-like access and iteration
    # ------------------------------------------------------------------

    def __getitem__(self, name):
        """Dict-like access: reader["name"] returns a numpy array (zero-copy, dense only)."""
        return self._read_numpy_single(name, False)

    def __iter__(self):
        return iter(self.keys())

    # ------------------------------------------------------------------
    # Context manager and repr
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __repr__(self):
        count = len(self)
        return f"<ztensor.Reader [{self.format}, {count} tensor(s)]>"
