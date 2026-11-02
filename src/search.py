import faiss 
import numpy as np

from artifacts_io import load_artifacts
from config import TOP_K , RESUME_TEXT_COL , RESUME_CAT_COL , EMBEDS_FILE
from embedding import load_encoder , encode_texts


class ResumeSearcher():
    """
    A class to search resumes based on a job description using FAISS index and embeddings.    
    """
    
    def __init__(self):
        # Load the artifacts and the encoder model once 
        self.index, self.resumes , self.meta = load_artifacts()
        self.encoder = load_encoder()

    def search(self, job_description: str , top_k: int = TOP_K):   
        """
        search for the top_k resumes that are most similar to the given job description

        Args:
            job_description (str): The job description text to search against the resumes.
            top_k (int): The number of top similar resumes to return. Defaults to TOP_K.
        
        Returns:
            pd.DataFrame: A DataFrame containing the top_k resumes and their similarity scores.
        """

        # Encode the job description to get its embedding 
        if not isinstance(job_description, str):
            raise TypeError(f"job_description must be a string, got {type(job_description).__name__}")
        if not job_description.strip():
            raise ValueError("job_description is empty or whitespace only")
        job_embedding = encode_texts([job_description] , self.encoder)

        # Normalize the job embedding to ensure it is compatible with the FAISS index
        faiss.normalize_L2(job_embedding)

        # Search the fiass index for the top_k most similar resumes
        distances, indices = self.index.search(job_embedding, top_k)

        # Retrieve the top_k resumes and their similarity scores
        cols = [RESUME_TEXT_COL, RESUME_CAT_COL]
        top_resumes = self.resumes.iloc[indices[0]][cols].copy()
        top_resumes['similarity_score'] = distances[0]

        return top_resumes


if __name__ == "__main__":

    ok = 0
    def check(label, condition):
        global ok
        assert condition, f"FAILED: {label}"
        ok += 1
        print(f"  ok  {label}")

    jd = ("We are hiring a senior software engineer with strong Python, "
            "Django and REST API experience to build backend services.")

    s = ResumeSearcher()
    out = s.search(jd)
    scores = out["similarity_score"].values

    # 1 - shape
    check(f"returns exactly {TOP_K} rows", len(out) == TOP_K)

    # 2 - a cosine cannot leave [-1, 1]
    check("every score is within [-1, 1]",
            scores.max() <= 1.0 and scores.min() >= -1.0)

    # 3 - every adjacent pair, not just the first
    check("scores are in descending order", (scores[:-1] >= scores[1:]).all())

    # 4 - empty input is rejected
    for bad in ("", "   "):
        try:
            s.search(bad)
        except ValueError:
            check(f"empty input {bad!r} raises ValueError", True)
        else:
            check(f"empty input {bad!r} raises ValueError", False)

    # 5 - the class matches a direct faiss query (no class involved)
    q_raw = encode_texts([jd], s.encoder, show_progress=False)
    q = q_raw.copy()
    faiss.normalize_L2(q)
    _, direct_pos = s.index.search(q, TOP_K)
    check("positions match a direct faiss search",
            (out.index.values == direct_pos[0]).all())

    # 6 - the top score equals the cosine worked out by hand
    doc = np.load(EMBEDS_FILE)[int(out.index.values[0])]
    by_hand = float(np.dot(q_raw[0], doc) /
                    (np.linalg.norm(q_raw[0]) * np.linalg.norm(doc)))
    check(f"top score matches the hand cosine "
            f"({scores[0]:.6f} vs {by_hand:.6f})",
            np.isclose(by_hand, scores[0], atol=1e-5))

    print(f"\n{ok} checks passed")
    print()
    print(out[[RESUME_CAT_COL, "similarity_score"]].to_string())
