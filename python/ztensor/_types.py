"""Pure Python Tensor and TensorMetadata classes."""


class TensorMetadata:
    """Metadata for a single tensor in a zTensor file."""

    __slots__ = ("name", "dtype", "type", "shape", "format", "components", "attributes")

    def __init__(
        self, name, dtype, shape, format, *, type=None, components=None, attributes=None
    ):
        self.name = name
        self.dtype = dtype
        self.type = type
        self.shape = shape
        self.format = format
        self.components = components
        self.attributes = attributes

    def __repr__(self):
        if self.type is not None:
            return (
                f"<TensorMetadata name='{self.name}' shape={list(self.shape)} "
                f"dtype='{self.dtype}' type='{self.type}'>"
            )
        return (
            f"<TensorMetadata name='{self.name}' shape={list(self.shape)} "
            f"dtype='{self.dtype}'>"
        )


class Tensor:
    """A tensor object representing a zt Object.

    Wraps one or more component arrays with metadata (shape, dtype, format).
    Dense tensors have a single "data" component; sparse and quantized tensors
    have multiple components (e.g., "values", "indices", "indptr").

    Example:
        t = ztensor.Tensor({"data": np_array})
        t = ztensor.Tensor({"values": v, "indices": i, "indptr": p},
                            shape=[4096, 4096], dtype="float32", format="sparse_csr")
    """

    __slots__ = ("shape", "dtype", "type", "format", "components", "attributes")

    def __init__(
        self,
        components,
        *,
        shape=None,
        dtype=None,
        format="dense",
        type=None,
        attributes=None,
    ):
        if not isinstance(components, dict):
            raise TypeError("components must be a dict")

        # Determine primary component name for dtype inference
        if format == "dense":
            primary = "data"
        elif format in ("sparse_csr", "sparse_coo"):
            primary = "values"
        else:
            primary = next(iter(components), "data")

        # Infer shape
        if shape is not None:
            self.shape = list(shape)
        elif format == "dense":
            if "data" not in components:
                raise ValueError(
                    "Dense tensor requires 'data' component or explicit shape"
                )
            self.shape = list(components["data"].shape)
        else:
            raise ValueError("Non-dense tensors require explicit shape")

        # Infer dtype
        if dtype is not None:
            self.dtype = str(dtype)
        else:
            if primary not in components:
                raise ValueError(f"Cannot infer dtype: missing '{primary}' component")
            self.dtype = str(components[primary].dtype)

        self.type = type
        self.format = format
        self.components = dict(components)
        self.attributes = dict(attributes) if attributes is not None else None

    def numpy(self):
        """Convert to a NumPy array (dense tensors only)."""
        if self.format != "dense":
            raise RuntimeError(
                f"numpy() only supported for dense tensors, got '{self.format}'"
            )
        if "data" not in self.components:
            raise RuntimeError("Dense tensor missing 'data' component")
        return self.components["data"]

    def torch(self, *, device="cpu"):
        """Convert to a torch.Tensor (dense tensors only)."""
        if self.format != "dense":
            raise RuntimeError(
                f"torch() only supported for dense tensors, got '{self.format}'"
            )
        import torch

        np_arr = self.numpy()
        tensor = torch.from_numpy(np_arr)
        if device != "cpu":
            tensor = tensor.to(device)
        return tensor

    def __repr__(self):
        return f"<ztensor.Tensor shape={self.shape} dtype='{self.dtype}' format='{self.format}'>"
