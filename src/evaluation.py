
import json
import pandas as pd
from config import RELEVANT_MAP_FILE , RESUME_CAT_COL , JOB_TITLE_COL , JOB_TEXT_COL , MIN_JD_WORDS , SEED , N_JDS

def hits_at_k(ranked_categories : list , correct_categories : set , k : int ) -> int:
    """
    Calculate the number of corrected predictions in the top k ranked categories.

    Args:
        ranked_categories (list): A list of categories ranked by relevance.
        correct_categories (set): A set of correct categories to check against the ranked list.
        k (int): The number of top items to consider for hits calculation.

    Returns:
        int: The number of correct predictions in the top k ranked categories.
    """
    return sum(c in correct_categories for c in list(ranked_categories)[:k])



def precision_at_k(ranked_categories : list , correct_categories : set , k : int ) -> float:
    """
    Calculate the precision at k for a given ranked list of categories and the correct category.

    Args:
        ranked_categories (list): A list of categories ranked by relevance.
        correct_categories (set): A set of correct categories to check against the ranked list.
        k (int): The number of top items to consider for precision calculation.

    Returns:
        float: The precision at k, which is the number of correct predictions in the top k ranked categories divided by k.
    """
    return hits_at_k(ranked_categories, correct_categories, k) / k


def reciprocal_rank(ranked_categories : list , correct_categories : set ) -> float:
    """
    Calculate the order of the first correct prediction in the ranked list of categories.

    Args:
        ranked_categories (list): A list of categories ranked by relevance.
        correct_categories (set): A set of correct categories to check against the ranked list.

    Returns:
        float: The reciprocal rank, which is 1 divided by the rank position of the first correct prediction.
            If the correct categories are not found, returns 0.
    """
    for i, category in enumerate(ranked_categories):
        if category in correct_categories:
            return 1 / (i + 1)
    return 0

def load_relevant_map(path = RELEVANT_MAP_FILE) -> dict[str , set]:
    """
    Read the json file and convert the list into a set for each key in the dictionary.
    This is useful for efficient lookups when checking if a category is relevant.

    Args:
        path (str): The path to the relevant map file. Defaults to RELEVANT_MAP_FILE.

    Returns:
        dict[str, set]: A mapping of job title to the set of acceptable resume
            categories. The values are sets, not lists, so that `in` compares
            whole category names instead of doing substring matching.
    """
    with open(path, "r") as f:
        relevant_map = json.load(f)
    # Convert the list of relevant categories into a set for each key in the dictionary
    relevant_map = {k: set(v) for k, v in relevant_map.items()}
    return relevant_map


def validate_labels(relevant_map : dict[str , set] , resumes_df : pd.DataFrame , jobs_df = None) -> None:
    """
    Validate the labels in the relevant_map against the categories.

    Args:
        relevant_map (dict): A dictionary mapping job titles to sets of relevant categories.
        resumes_df (pd.DataFrame): A DataFrame containing resumes with job titles and categories.
        jobs_df (pd.DataFrame, optional): A DataFrame containing job postings with job titles. Defaults to None.
    Returns:
        None
    """
    # Every category in the map must exist in the resume corpus

    known = set(resumes_df[RESUME_CAT_COL].unique())
    for title , cats in relevant_map.items():
        assert isinstance(cats, set), (
            f"{title}: must be a set, not {type(cats).__name__} - with a "
            f"string, `in` would do silent substring matching")
        missing = cats - known
        assert not missing, (
            f"{title}: categories not found in the corpus -> {sorted(missing)}")
    # If jobs_df is provided, validate job titles
    if jobs_df is not None:
        known_titles = set(jobs_df[JOB_TITLE_COL].unique())
        for title in relevant_map:
            if title not in known_titles:
                print(f"Warning: {title!r} in relevant_map is not present in jobs_df")

    print(f"Labels ok ({len(relevant_map)} titles)")


def sample_job_descriptions(jobs_df : pd.DataFrame , title : str , n = N_JDS , min_words = MIN_JD_WORDS , seed = SEED) -> list[str]:
    """
    Pick a reproducible sample of job descriptions for one job title.

    Postings shorter than min_words are dropped first: a 7-word posting
    carries no signal, so scoring it would measure the input, not the model.
    The seed makes the sample identical on every run, which is what lets a
    change in the final score be read as a change in the system.

    Args:
        jobs_df (pd.DataFrame): A DataFrame containing job postings with job titles and descriptions.
        title (str): The job title to sample descriptions for.
        n (int, optional): The number of job descriptions to sample. Defaults to N_JDS.
        min_words (int, optional): The minimum number of words required in a job description. Defaults to MIN_JD_WORDS.
        seed (int, optional): The random seed for reproducibility. Defaults to SEED.

    Returns:
        list[str]: The sampled descriptions. Fewer than n if the title has
            fewer long-enough postings, and empty if the title is unknown or
            every posting is too short. Never raises for a missing title.
    """
    jobs = jobs_df[jobs_df[JOB_TITLE_COL] == title]

    jobs = jobs[jobs[JOB_TEXT_COL].str.split().str.len() >= min_words]

    jobs = jobs.sample(n=min(n, len(jobs)), random_state=seed)

    jobs = jobs[JOB_TEXT_COL].tolist()

    return jobs



if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    ok = 0

    def check(label, condition):
        global ok
        assert condition, f"FAILED: {label}"
        ok += 1
        print(f"  ok  {label}")

    def raises(fn, *args, **kwargs) -> bool:
        """Return True if calling fn(...) raises AssertionError."""
        try:
            fn(*args, **kwargs)
        except AssertionError:
            return True
        return False

    # one shared example you can count by hand
    ranked = ["HR", "IT", "AGRICULTURE", "IT", "HR",
            "CONSULTANT", "IT", "ENGINEERING", "HR", "HR"]
    correct = {"IT"}
    # correct results sit at ranks 2, 4, 7


    check("hits@10 counts all three", hits_at_k(ranked, correct, 10) == 3)

    check("hits@5 counts two", hits_at_k(ranked, correct, 5) == 2)

    # check precision@k
    check("precision@10 is 0.3", precision_at_k(ranked, correct, 10) == 0.3)

    check("precision@5 is 0.4", precision_at_k(ranked, correct, 5) == 0.4)

    # check reciprocal rank
    check("reciprocal rank is 1/2", reciprocal_rank(ranked, correct) == 1/2)

    # check no correct predictions
    check("hits is 0 when nothing matches", hits_at_k(ranked, {"SALES"}, 10) == 0)
    check("precision is 0 when nothing matches", precision_at_k(ranked, {"SALES"}, 10) == 0.0)

    fin = ["BANKING", "FINANCE", "HR", "BANKING", "CHEF",
    "ACCOUNTANT", "FINANCE", "HR", "HR", "AVIATION"]
    accepted = {"BANKING", "FINANCE"}
    # correct results sit at ranks 1, 2, 4, 7

    check("hits@10 with a two-category set", hits_at_k(fin, accepted, 10) == 4)
    check("RR is 1.0 when the first result is correct", reciprocal_rank(fin, accepted) == 1.0)

    check("RR is 0 when nothing matches", reciprocal_rank(ranked, {"SALES"}) == 0)

    try:
        n = hits_at_k(["IT", "HR", "IT"], {"IT"}, 10)
        check("a short list does not raise, and counts what is there", n == 2)
    except IndexError:
        check("a short list does not raise, and counts what is there", False)

    # check load_relevant_map on a temporary file, so the checks do not depend
    # on the real relevant_map.json and break every time it is edited
    with tempfile.TemporaryDirectory() as tmp:
        tmp_json = Path(tmp) / "relevant_map.json"
        tmp_json.write_text(json.dumps({
            "Software Engineer": ["INFORMATION-TECHNOLOGY", "ENGINEERING"],
            "HR Manager": ["HR", "HR"],
        }))

        loaded = load_relevant_map(tmp_json)

        check("every value is a set, not a list",
              all(isinstance(v, set) for v in loaded.values()))
        check("the titles and categories survive the round trip",
              loaded["Software Engineer"] == {"INFORMATION-TECHNOLOGY", "ENGINEERING"})
        check("a repeated category collapses to one",
              loaded["HR Manager"] == {"HR"})

    resumes = pd.DataFrame({RESUME_CAT_COL:
    ["INFORMATION-TECHNOLOGY", "HR", "BANKING", "FINANCE", "ACCOUNTANT", "CHEF"] * 4})
    jobs = pd.DataFrame({JOB_TITLE_COL: ["Software Engineer"] * 3 + ["HR Manager"] * 2})

    good = {"Software Engineer": {"INFORMATION-TECHNOLOGY"}, "HR Manager": {"HR"}}

    check("a valid map passes",
          not raises(validate_labels, good, resumes, jobs))
    check("a misspelled category is rejected",
          raises(validate_labels, {"Dev": {"INFORMATION-TECHNOLOGI"}}, resumes))
    check("a category that is not in the corpus is rejected",
          raises(validate_labels, {"Dev": {"HR", "PYTHON"}}, resumes))
    check("a string instead of a set is rejected",
          raises(validate_labels, {"Dev": "ACCOUNTANT"}, resumes))
    check("a list instead of a set is rejected",
          raises(validate_labels, {"Dev": ["ACCOUNTANT"]}, resumes))
    check("a title with no postings warns but does not raise",
          not raises(validate_labels, {"Truck Driver": {"CHEF"}}, resumes, jobs))

    def posting(tag, words):
            """
            Build a fake posting whose first word is tag and whose length is known.
            """
            return f"{tag} " + "word " * (words - 1)

    # HR postings are long on purpose: if the title filter were broken they
    # would survive the word filter and show up in the result
    postings = pd.DataFrame({
        JOB_TITLE_COL: ["Software Engineer"] * 4 + ["HR Manager"] * 2 + ["Software Engineer"],
        JOB_TEXT_COL: [posting("SWE-A", 200), posting("SWE-B", 200),
                       posting("SWE-C", 200), posting("SWE-SHORT", 10),
                       posting("HR-A", 200), posting("HR-B", 200),
                       None],
    })
    # 3 usable Software Engineer postings: SWE-A, SWE-B, SWE-C

    check("it returns at most n",
          len(sample_job_descriptions(postings, "Software Engineer", n=2)) == 2)

    picked = sample_job_descriptions(postings, "Software Engineer", n=10)
    check("only the requested title comes back",
          all(t.split()[0].startswith("SWE") for t in picked))

    check("every posting returned is long enough",
          all(len(t.split()) >= MIN_JD_WORDS for t in picked))

    a = sample_job_descriptions(postings, "Software Engineer", n=2, seed=42)
    b = sample_job_descriptions(postings, "Software Engineer", n=2, seed=42)
    check("the same seed gives the same sample", a == b)

    check("an unknown title returns an empty list",
          sample_job_descriptions(postings, "Truck Driver", n=5) == [])

    check("asking for more than exists returns what exists",
          len(sample_job_descriptions(postings, "Software Engineer", n=10)) == 3)

    check("a missing description is skipped, not crashed on",
          "nan" not in [str(t).strip() for t in picked])


    print(f"\n{ok} checks passed")