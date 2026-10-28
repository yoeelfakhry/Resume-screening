"""
artifacts_io.py - Writes and reads the four build artifacts.

Why the writer and the reader live in ONE file:

build_index.py writes the artifacts, search.py reads them. If each of them
named the files itself, the two could disagree - and the bad case is not a
FileNotFoundError, it is an older file sitting under the other name and
loading without complaint. Putting save and load side by side means the
same four names, the same order, and the same checks are used by both.

The artifacts are a SET, not four independent files. Position i in the index
means row i of the DataFrame; if one is rebuilt and the other is not, every
result is wrong with no error raised. They are therefore always written
together and always verified together.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import faiss
import numpy as np
import pandas as pd

import config
from config import (
    ARTIFACT_DIR,
    EMBED_DIM,
    EMBEDS_FILE,
    INDEX_FILE,
    META_FILE,
    MODEL_NAME,
    RESUME_TEXT_COL,
    RESUMES_FILE,
)
# Reused, not rewritten: retrieval.py already hashes the exact texts that
# went into the embeddings.
from embedding import fingerprint


def save_artifacts(index, resumes_df: pd.DataFrame, embeddings: np.ndarray) -> dict:
    """Write index, DataFrame, raw embeddings and meta.json together.

    Returns the meta dict that was written, so the caller can print it.
    """
    assert index.ntotal == len(resumes_df), (
        f"refusing to save: index has {index.ntotal} vectors but the DataFrame "
        f"has {len(resumes_df)} rows - these artifacts would be unusable"
    )
    assert len(embeddings) == len(resumes_df), (
        f"refusing to save: {len(embeddings)} embeddings for "
        f"{len(resumes_df)} rows"
    )
    assert index.d == EMBED_DIM, (
        f"index dimension {index.d} != config embed_dim {EMBED_DIM} - "
        f"a different model was probably loaded"
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    resumes_df.to_parquet(RESUMES_FILE, index=False)
    np.save(EMBEDS_FILE, embeddings)

    # meta.json is what turns "a .faiss file of unknown origin" into
    # something identifiable. Without it you cannot tell which model, which
    # settings, or which version of the data produced the index.
    meta = {
        **config.as_dict(),
        "n_resumes": int(len(resumes_df)),
        "index_ntotal": int(index.ntotal),
        "index_dim": int(index.d),
        # The fingerprint is the part that catches a stale build: the data
        # changed, the row count did not, and nobody re-ran build_index.
        "corpus_fingerprint": fingerprint(resumes_df[RESUME_TEXT_COL]),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"saved to {ARTIFACT_DIR}/")
    for path in (INDEX_FILE, RESUMES_FILE, EMBEDS_FILE, META_FILE):
        print(f"  {path.name:24s} {path.stat().st_size / 1e6:7.2f} MB")
    return meta


def load_artifacts(check_corpus: pd.DataFrame | None = None):
    """Load the artifact set and verify it is internally consistent.

    Returns (index, resumes_df, meta).

    Pass check_corpus to also verify the artifacts match a freshly loaded
    DataFrame - that is how a stale index gets caught instead of silently
    answering queries from old data.
    """
    for path in (INDEX_FILE, RESUMES_FILE, META_FILE):
        assert path.exists(), f"missing artifact: {path} - run build_index.py first"

    index = faiss.read_index(str(INDEX_FILE))
    resumes_df = pd.read_parquet(RESUMES_FILE)
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))

    # 1. The index and the DataFrame must describe the same corpus, because
    #    position i in one means row i in the other.
    assert index.ntotal == len(resumes_df), (
        f"artifacts are out of sync: index={index.ntotal}, "
        f"dataframe={len(resumes_df)} - rebuild them"
    )

    # 2. The query model must be the model that built the index. A different
    #    model with the SAME dimension loads fine and returns nonsense.
    assert meta["model_name"] == MODEL_NAME, (
        f"model mismatch: artifacts built with {meta['model_name']!r}, "
        f"config says {MODEL_NAME!r} - rebuild or fix config.yaml"
    )

    # 3. Dimension, which also catches a config edit after the build.
    assert index.d == EMBED_DIM, (
        f"index dimension {index.d} != config embed_dim {EMBED_DIM}"
    )

    # 4. Optional and strongest: does the data on disk still hash to what was
    #    indexed? Catches "the CSV changed and nobody rebuilt".
    if check_corpus is not None:
        current = fingerprint(check_corpus[RESUME_TEXT_COL])
        assert current == meta["corpus_fingerprint"], (
            f"stale artifacts: data fingerprint {current} != "
            f"indexed fingerprint {meta['corpus_fingerprint']} - "
            f"the data changed since the last build, re-run build_index.py"
        )

    return index, resumes_df, meta


if __name__ == "__main__":
    # Self-test on fake data: no model download, no real corpus.
    import shutil
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    config.ARTIFACT_DIR = tmp
    for name in ("INDEX_FILE", "RESUMES_FILE", "EMBEDS_FILE", "META_FILE"):
        globals()[name] = tmp / getattr(config, name).name
    ARTIFACT_DIR = tmp

    rng = np.random.default_rng(0)
    embs = np.ascontiguousarray(rng.normal(size=(50, EMBED_DIM)), dtype="float32")
    df = pd.DataFrame({
        RESUME_TEXT_COL: [f"resume number {i}" for i in range(50)],
        "Category": ["IT"] * 50,
    })

    from embedding import build_index
    idx = build_index(embs, name="test")

    print()
    save_artifacts(idx, df, embs)

    print()
    idx2, df2, meta = load_artifacts(check_corpus=df)
    assert idx2.ntotal == 50
    assert df2[RESUME_TEXT_COL].equals(df[RESUME_TEXT_COL])
    print("load ok, fingerprint verified")

    # The stale-data check must fire when the corpus changes.
    df_changed = df.copy()
    df_changed.loc[0, RESUME_TEXT_COL] = "resume number 0 EDITED"
    try:
        load_artifacts(check_corpus=df_changed)
    except AssertionError as e:
        print(f"stale check fired: {str(e).splitlines()[0][:70]}...")
    else:
        raise SystemExit("FAIL: stale corpus was not detected")

    # A size mismatch must be refused at save time.
    try:
        save_artifacts(idx, df.head(10), embs)
    except AssertionError as e:
        print(f"size check fired:  {str(e).splitlines()[0][:70]}...")
    else:
        raise SystemExit("FAIL: size mismatch was accepted")

    shutil.rmtree(tmp)
    print("\nall self-tests passed")
