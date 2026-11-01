import hashlib 
import numpy as np 
from collections.abc import Sequence
from sentence_transformers import SentenceTransformer
from config import CACHE_DIR, BATCH_SIZE, EMBED_DIM, MODEL_NAME, MODEL_MAX_TOKENS
import faiss


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


def load_encoder() -> SentenceTransformer:
    """ load the model, check the dimention is correct, and the max sequence length is correct and return the encoder object 
    
    Args:
        None
    Returns:
        SentenceTransformer: encoder object
    """
    model = SentenceTransformer(MODEL_NAME)
    assert model.get_embedding_dimension() == EMBED_DIM, \
    f"model dimension is {model.get_embedding_dimension()}, expected {EMBED_DIM}"
    assert model.max_seq_length == MODEL_MAX_TOKENS, \
    f"model max sequence length is {model.max_seq_length}, expected {MODEL_MAX_TOKENS}"
    model._model_tag = MODEL_NAME.split("/")[-1]
    return model


def build_index(embeddings: np.ndarray , name: str = "index") -> faiss.IndexFlatIP:
    """
    Build a FAISS index for the given embeddings.
    
    Args
        embeddings (np.ndarray): input embeddings array
        name (str): name of the index for logging purposes. Defaults to "index"
    Returns
        faiss.IndexFlatIP: FAISS index object
    """
    embs = prep(embeddings).copy()
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    assert index.ntotal == len(embeddings), f"{name} index has {index.ntotal} vectors, expected {len(embeddings)}"
    print(f"{name}: {index.ntotal} vectors, dim={embs.shape[1]}")
    return index


def encode_texts(texts: Sequence[str] , encoder: SentenceTransformer , cache_key : str | None = None , 
                 batch_size: int = BATCH_SIZE , show_progress: bool = True ,) -> np.ndarray:
    """Embed texts, reusing a saved cache only when it genuinely matches.
    Caching is opt-in: pass cache_key to enable it, leave it None to skip.
    A single query in search.py does not deserve a cache file; the 2,482
    resume corpus does.

    The filename carries three things that each change the result - the
    caller's key, the model, and a fingerprint of the texts. A naive cache
    keyed on (name, row count) would hand back vectors of the OLD text
    after a cleaning change, with no error at all.

    Args:
        texts: the texts to embed.
        encoder: a model from load_encoder().
        cache_key: a short name such as "head", or None to disable caching.
        batch_size: passed through to the model.
        show_progress: show the model's progress bar.

    Returns:
        A float32, C-contiguous array of shape (len(texts), EMBED_DIM).
    """
    # Make sure the texts are a sequence of strings 
    texts = [str(t) for t in texts]

    path=0

    if cache_key is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_tag = getattr(encoder, "_model_tag", type(encoder).__name__)
        path = CACHE_DIR / f"{cache_key}__{model_tag}__{fingerprint(texts)}.npy"

    if path is not None and path.exists():
        embs = prep(np.load(path))
        print(f"cache hit:  {path.name}  {embs.shape}")
    else:
        embs = prep(encoder.encode(texts, convert_to_numpy=True,
                                   batch_size=batch_size,
                                   show_progress_bar=show_progress))

        if path is not None:
            np.save(path, embs)
            print(f"cache save: {path.name}  {embs.shape}")

    # Both paths land here, so a stale cache file with the wrong shape is
    # caught too - not only a fresh encoding.
    assert embs.shape[0] == len(texts), \
        f"embeddings have {embs.shape[0]} rows, expected {len(texts)}"
    assert embs.shape[1] == EMBED_DIM, \
        f"embeddings have {embs.shape[1]} columns, expected {EMBED_DIM}"
    return embs
    

if __name__ == "__main__":
    ok = 0

    def check(label, condition):
        global ok
        assert condition, f"FAILED: {label}"
        ok += 1
        print(f"  ok  {label}")

    good = np.zeros((3, 4), dtype="float32")
    # ============ Checks is here ============
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
    
    orig = prep(np.array([[3., 4.], [1., 0.], [0., 2.]]))
    before = orig.copy()

    idx = build_index(orig, "test_index")

    check("build_index does not modify its input",
          np.array_equal(orig, before))

    check("every stored vector has length 1",
          np.allclose([np.linalg.norm(idx.reconstruct(i)) for i in range(3)], 1.0))

    idx2 = build_index(orig, "test_index")
    check("rebuilding does not accumulate", idx2.ntotal == 3)

    print(f"\n{ok} checks passed")
