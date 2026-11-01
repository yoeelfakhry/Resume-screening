"""
Build the search artifacts from the raw resume CSV.
This is the indexing phase of the pipeline. It runs once and takes a few
minutes; search.py then answers queries from its output in milliseconds.

    data/Resume.csv
        -> clean and deduplicate         (data.load_resumes)
        -> verify against the reference  (regression guard)
        -> embed                         (embedding.encode_texts)
        -> build a FAISS index           (embedding.build_index)
        -> artifacts/

Reads:
    data/Resume.csv                  the raw corpus
    data/resumes_reference.parquet   the expected cleaning output

Writes (all four, or none):
    artifacts/head_index.faiss       the searchable index
    artifacts/resumes.parquet        the table the index positions refer to
    artifacts/head_embs.npy          the un-normalised embeddings
    artifacts/meta.json              model, settings and corpus fingerprint

Run:
    python src/build_index.py

The first run embeds 2,482 resumes and takes minutes. Later runs reuse the
embedding cache and finish in seconds.
"""

import pandas as pd 
from artifacts_io import save_artifacts
from config import RESUMES_REFERENCE , RESUME_TEXT_COL , RESUME_CAT_COL
from data import load_resumes 
from embedding import build_index as build_faiss_index , check_embeddings , encode_texts , load_encoder 


def main():
    # Load the resumes data 
    data = load_resumes()

    # Compare the loaded data with the refrence data to ensure consistency
    ref = pd.read_parquet(RESUMES_REFERENCE)
    assert len(data) == len(ref), f"loaded data has {len(data)} rows, expected {len(ref)}"
    assert data[RESUME_TEXT_COL].equals(ref[RESUME_TEXT_COL]), "cleaning drifted from the reference"
    assert data[RESUME_CAT_COL].equals(ref[RESUME_CAT_COL]), "categories drifted from the reference"

    # Load the encoder model 
    encoder = load_encoder()

    # Encode the resumes text to get embeddings 
    embeddings = encode_texts(data[RESUME_TEXT_COL].tolist(), encoder , cache_key="head")

    # check the embeddings shape and dimension
    check_embeddings(embeddings , name="head_embs"  , expected_rows=len(data))

    # Build the FAISS index using the embeddings
    index = build_faiss_index(embeddings, name="head_index")

    # Save the artifacts (index, resumes DataFrame, embeddings) to disk
    meta = save_artifacts(index, data, embeddings)

    return meta



if __name__ == "__main__":
    meta = main()
    print("Artifacts saved with the following metadata:")
    for key, value in meta.items():
        print(f"{key}: {value}")








