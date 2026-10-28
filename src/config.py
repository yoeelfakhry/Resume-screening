"""
config.py - Loads config.yaml, validates it, exposes typed constants.

The rest of the pipeline imports from HERE, never from the YAML:

    from config import MODEL_NAME, TOP_K, INDEX_FILE

Why a loader instead of yaml.safe_load() at each call site:

    cfg.get("chunk_overlap", 50)

A misspelled key does not raise. The line above returns 50 whether the key
is spelled right, spelled wrong, or missing - so an edit to the YAML is
silently ignored and the run still looks fine. The checks below turn every
such typo into an error at import time.

Scope: this file validates that the CONFIG is well-formed. It does not check
that files exist on disk - artifacts/ not existing yet is the normal state
before build_index.py has run. Run `python config.py` for a disk report.
"""

from pathlib import Path

import yaml

# Everything is anchored to the project root, which is found from THIS
# file's location - never from the current working directory.
#
# Why: a relative path like Path("data") is resolved against whatever folder
# you happened to run the command from. `python src/build_index.py` from the
# root would find data/, and the same script run from inside src/ would not.
# Anchoring to __file__ makes the pipeline behave the same from anywhere.
#
#   <project root>/
#     configs/config.yaml
#     src/config.py     <- __file__
#     data/  artifacts/  cache/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with CONFIG_PATH.open(encoding="utf-8") as fh:
    _cfg = yaml.safe_load(fh)

assert isinstance(_cfg, dict), f"{CONFIG_PATH.name}: expected a mapping at the top level"

# =============================================================================
# Validation
# =============================================================================
# Every key the pipeline expects, listed once. A missing key fails here,
# loudly, instead of becoming a None three modules away.
_REQUIRED = {
    "seed": None,
    "model": ["name", "embed_dim", "max_tokens", "batch_size"],
    "paths": ["data_dir", "artifact_dir", "cache_dir"],
    "data_files": ["resumes_csv", "resumes_reference", "jobs_eval",
                   "relevant_map", "judgments"],
    "artifacts": ["index", "resumes", "embeddings", "meta"],
    "columns": ["resume_text", "resume_category", "job_title", "job_text"],
    "retrieval": ["top_k"],
    "evaluation": ["min_jd_words", "n_jds"],
}

for _section, _keys in _REQUIRED.items():
    assert _section in _cfg, f"{CONFIG_PATH.name}: missing section '{_section}'"
    if _keys is None:
        continue
    for _key in _keys:
        assert _key in _cfg[_section], \
            f"{CONFIG_PATH.name}: missing key '{_section}.{_key}'"
    # A typo creates a NEW key instead of overwriting one, so any unexpected
    # key is almost always a misspelling of an expected one.
    _extra = set(_cfg[_section]) - set(_keys)
    assert not _extra, \
        f"{CONFIG_PATH.name}: unknown key(s) in '{_section}': {sorted(_extra)}"

_unknown = set(_cfg) - set(_REQUIRED)
assert not _unknown, f"{CONFIG_PATH.name}: unknown section(s): {sorted(_unknown)}"

# =============================================================================
# Typed constants
# =============================================================================
# int() and Path() here, not at the call sites. YAML will happily hand you
# the string "10" for top_k, and a string survives a long way before it breaks.

SEED = int(_cfg["seed"])

MODEL_NAME = str(_cfg["model"]["name"])
EMBED_DIM = int(_cfg["model"]["embed_dim"])
MODEL_MAX_TOKENS = int(_cfg["model"]["max_tokens"])
BATCH_SIZE = int(_cfg["model"]["batch_size"])

# Absolute, so every script resolves them identically no matter where it
# was launched from. The YAML keeps the short relative names.
DATA_DIR = PROJECT_ROOT / _cfg["paths"]["data_dir"]
ARTIFACT_DIR = PROJECT_ROOT / _cfg["paths"]["artifact_dir"]
CACHE_DIR = PROJECT_ROOT / _cfg["paths"]["cache_dir"]

# Production input.
RESUMES_CSV = DATA_DIR / _cfg["data_files"]["resumes_csv"]

# Outside the production path: regression fixture and evaluation inputs.
RESUMES_REFERENCE = DATA_DIR / _cfg["data_files"]["resumes_reference"]
JOBS_EVAL = DATA_DIR / _cfg["data_files"]["jobs_eval"]
RELEVANT_MAP_FILE = DATA_DIR / _cfg["data_files"]["relevant_map"]
JUDGMENTS_FILE = DATA_DIR / _cfg["data_files"]["judgments"]

INDEX_FILE = ARTIFACT_DIR / _cfg["artifacts"]["index"]
RESUMES_FILE = ARTIFACT_DIR / _cfg["artifacts"]["resumes"]
EMBEDS_FILE = ARTIFACT_DIR / _cfg["artifacts"]["embeddings"]
META_FILE = ARTIFACT_DIR / _cfg["artifacts"]["meta"]

RESUME_TEXT_COL = str(_cfg["columns"]["resume_text"])
RESUME_CAT_COL = str(_cfg["columns"]["resume_category"])
JOB_TITLE_COL = str(_cfg["columns"]["job_title"])
JOB_TEXT_COL = str(_cfg["columns"]["job_text"])

TOP_K = int(_cfg["retrieval"]["top_k"])
MIN_JD_WORDS = int(_cfg["evaluation"]["min_jd_words"])
N_JDS = int(_cfg["evaluation"]["n_jds"])

# Values that are wrong in a way the types cannot catch.
assert SEED >= 0, f"seed must be >= 0, got {SEED}"
assert TOP_K > 0, f"retrieval.top_k must be > 0, got {TOP_K}"
assert EMBED_DIM > 0, f"model.embed_dim must be > 0, got {EMBED_DIM}"
assert BATCH_SIZE > 0, f"model.batch_size must be > 0, got {BATCH_SIZE}"
assert MIN_JD_WORDS > 0, f"evaluation.min_jd_words must be > 0, got {MIN_JD_WORDS}"
assert N_JDS > 0, f"evaluation.n_jds must be > 0, got {N_JDS}"

# =============================================================================
# Dependency direction
# =============================================================================
# This module imports NOTHING from the project. It is the bottom layer:
# retrieval.py, artifacts_io.py, data.py, build_index.py and search.py all
# import from here, never the other way round.
#
# That is what removed the duplicate constants retrieval.py used to carry,
# and with them the thirty lines of drift assertions that guarded the
# duplication. One definition needs no guard.


def as_dict() -> dict:
    """The resolved settings, for writing into meta.json beside the artifacts.

    This is the real payoff of a config file: the exact settings that produced
    a set of artifacts get saved WITH them. A stale index becomes something
    you can identify instead of something you guess at.
    """
    return {
        "seed": SEED,
        "model_name": MODEL_NAME,
        "embed_dim": EMBED_DIM,
        "max_tokens": MODEL_MAX_TOKENS,
        "batch_size": BATCH_SIZE,
        "top_k": TOP_K,
        "min_jd_words": MIN_JD_WORDS,
        "n_jds": N_JDS,
    }


if __name__ == "__main__":
    print(f"config      {CONFIG_PATH}")
    print(f"model       {MODEL_NAME}")
    print(f"            dim={EMBED_DIM}  max_tokens={MODEL_MAX_TOKENS}  batch={BATCH_SIZE}")
    print(f"top_k       {TOP_K}        n_jds  {N_JDS}")
    print(f"seed        {SEED}        min_jd_len  {MIN_JD_WORDS} words")

    print("\nproduction input")
    print(f"  {'found  ' if RESUMES_CSV.exists() else 'MISSING'}  {RESUMES_CSV}")

    print("\nregression fixture and evaluation inputs")
    for _p in (RESUMES_REFERENCE, JOBS_EVAL, RELEVANT_MAP_FILE, JUDGMENTS_FILE):
        print(f"  {'found  ' if _p.exists() else 'MISSING'}  {_p}")

    print("\nfolders")
    for _d in (DATA_DIR, ARTIFACT_DIR, CACHE_DIR):
        print(f"  {'found  ' if _d.exists() else 'absent '}  {_d}/")

    print("\nvalidation passed")
