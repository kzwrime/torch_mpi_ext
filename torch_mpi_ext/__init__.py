import torch
from pathlib import Path
from . import _C, ops


def is_available() -> bool:
    return _C is not None


def get_include() -> str:
    """
    Return the directory containing installed torch_mpi_ext public C/C++ headers.
    """
    return str(Path(__file__).resolve().parent / "include")


def get_library_path() -> str:
    """
    Return the installed torch_mpi_ext extension library path.
    """
    if not is_available():
        raise RuntimeError("torch_mpi_ext is not available.")
    return str(Path(_C.__file__).resolve())
