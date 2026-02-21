"""Python Writer wrapping the Rust _Writer primitive."""


class Writer:
    """Writer for creating zTensor files.

    Usage::

        writer = ztensor.Writer("output.zt")
        writer.write_numpy("weights", numpy_array)
        writer.finish()

    Or as a context manager::

        with ztensor.Writer("output.zt") as w:
            w.write_numpy("weights", numpy_array)
    """

    def __init__(self, path):
        from ._ztensor import _Writer

        self._inner = _Writer(path)
        self._finished = False

    @classmethod
    def append(cls, path):
        """Open an existing .zt file for appending new tensors."""
        from ._ztensor import _Writer

        obj = cls.__new__(cls)
        obj._inner = _Writer.append(path)
        obj._finished = False
        return obj

    def write(self, name, data, *, compress=None, checksum="none"):
        """Write a ztensor.Tensor to the file.

        Args:
            name: Tensor name.
            data: A ztensor.Tensor object.
            compress: False/0 for raw, True for default zstd, int for specific level.
            checksum: "none", "crc32c", or "sha256".
        """
        self._inner._write_object(name, data, compress=compress, checksum=checksum)

    def write_numpy(self, name, data, *, compress=None, checksum="none"):
        """Write a dense tensor from a NumPy array.

        Args:
            name: Tensor name.
            data: NumPy array (must be contiguous).
            compress: False/0 for raw, True for default zstd, int for specific level.
            checksum: "none", "crc32c", or "sha256".
        """
        self._inner._write_numpy(name, data, compress=compress, checksum=checksum)

    def write_torch(self, name, data, *, compress=None, checksum="none"):
        """Write a dense tensor from a torch.Tensor.

        Args:
            name: Tensor name.
            data: torch.Tensor (will be moved to CPU and made contiguous if needed).
            compress: False/0 for raw, True for default zstd, int for specific level.
            checksum: "none", "crc32c", or "sha256".
        """
        self._inner._write_torch(name, data, compress=compress, checksum=checksum)

    def add(self, name, data, compress=None, checksum="none"):
        """Add a dense tensor from a NumPy array (alias for write_numpy)."""
        self._inner._write_numpy(name, data, compress=compress, checksum=checksum)

    def finish(self):
        """Finish the file (writes manifest and footer). Returns file size."""
        self._finished = True
        return self._inner._finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._finished:
            if exc_type is None:
                self.finish()
            else:
                self._finished = True
                self._inner._discard()
        return False

    def __repr__(self):
        if not self._finished:
            return "<ztensor.Writer (open)>"
        return "<ztensor.Writer (finished)>"
