import hashlib 
import numpy as np 
from collections.abc import Sequence

def prep( numpy_array: np.ndarray) -> np.ndarray:
    """
    return the given array with float32 type and make it organized in the memory
    
    Args
        numpy_array (np.ndarray): input numpy array 
        
    Returns
        np.ndarray: araray with float32 type and organized in the memory
    """
    return np.ascontiguousarray(numpy_array , dtype=np.float32)


def check_embeddings( embedding: np.ndarray, name : str  , expected_rows: int | None = None)   -> np.ndarray:
    """
    check the embedding is arrray type , has floa32 , no of dimentions is 2 , it is contiguous and has expected number of rows if provided
    to be ready for the FAISS index creation and search.

    Args
        embedding (np.ndarray): input embedding array 
        name (str): name of the embedding array 
        expected_rows (int, optional): expected number of rows in the embedding array. Defaults to None.

    Returns:
        np.ndarray: the input embedding array if all checks pass
    """
    assert isinstance(embedding , np.ndarray) , f"{name} should be a numpy array, got {type(embedding)}"
    assert embedding.dtype == np.float32, f"{name} should be float32, got {embedding.dtype}"
    assert embedding.ndim == 2 , f"{name} should be 2D array, got {embedding.ndim}D"
    assert embedding.flags['C_CONTIGUOUS'], f"{name} should be contiguous in memory"
    if expected_rows is not None:
        assert embedding.shape[0] == expected_rows, f"{name} should have {expected_rows} rows, got {embedding.shape[0]}"
    return embedding


def fingerprint(texts: Sequence[str]) -> str:
    """Return a short hash of exactly the texts given, in order.

    This is the content key for the embedding cache. The cache filename
    carries this value, so any change to the texts changes the filename and
    the cache misses - which is what should happen.

    The b"\\x00" separator is not optional: without it, ["ab", "c"] and
    ["a", "bc"] both hash the bytes "abc" and collide.
    """
    h = hashlib.sha1()
    for t in texts:
        h.update(str(t).encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:10]


