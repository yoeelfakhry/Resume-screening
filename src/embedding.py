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


if __name__ == "__main__":
    ok = 0

    def check(label, condition):
        global ok
        assert condition, f"FAILED: {label}"
        ok += 1
        print(f"  ok  {label}")

    good = np.zeros((3, 4), dtype="float32")
    # ============ الفحوصات تتحط من هنا ============
    check("prep converts float64 to float32",
        prep(np.zeros((3, 4))).dtype == np.float32)

    try:
        check_embeddings(np.zeros((5, 8)), "test")
    except AssertionError:
        check("check_embeddings rejects float64", True)
    else:
        check("check_embeddings rejects float64", False)

    check("prep makes a transpose contiguous",
          prep(np.zeros((4, 3)).T).flags["C_CONTIGUOUS"])

    check("prep is a no-op when the array is already correct",
          prep(good) is good)

    check("check_embeddings accepts a good array and returns it",
          check_embeddings(good, "test", 3) is good)

    check("same texts give the same hash",
          fingerprint(["a", "b"]) == fingerprint(["a", "b"]))

    check("one character changes the hash",
          fingerprint(["a", "b"]) != fingerprint(["a", "c"]))

    check("order matters",
          fingerprint(["a", "b"]) != fingerprint(["b", "a"]))

    check("the separator prevents a collision",
          fingerprint(["ab", "c"]) != fingerprint(["a", "bc"]))

    try:
        check_embeddings(np.zeros(8, dtype="float32"), "test")
    except AssertionError:
        check("check_embeddings rejects a 1-D array", True)
    else:
        check("check_embeddings rejects a 1-D array", False)

    try:
        check_embeddings(np.zeros((8, 5), dtype="float32").T, "test")
    except AssertionError:
        check("check_embeddings rejects a non-contiguous array", True)
    else:
        check("check_embeddings rejects a non-contiguous array", False)

    try:
        check_embeddings(good, "test", 99)
    except AssertionError:
        check("check_embeddings rejects a wrong row count", True)
    else:
        check("check_embeddings rejects a wrong row count", False)

    print(f"\n{ok} checks passed")
