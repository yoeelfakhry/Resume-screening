import pandas as pd
import re 
from config import RESUMES_CSV, RESUMES_REFERENCE, JOBS_EVAL, RESUME_TEXT_COL, RESUME_CAT_COL, JOB_TITLE_COL, JOB_TEXT_COL

EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
PHONE_PATTERN = re.compile(r'(?<!\d)(\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)')
LINKEDIN_PATTERN = re.compile(r'linkedin\.com/in/[a-zA-Z0-9-]+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

def mask_pii (text: str) -> str:
    """
    mask personally identifiable information (PII) in a given text using regex patterns 
    
    Args:
        text (str): input text to mask PII
    
    Returns:
        str: text with PII masked
    """
    if not isinstance(text, str):
        return text
    text = EMAIL_PATTERN.sub('[EMAIL]', text)
    text = LINKEDIN_PATTERN.sub('[LINKEDIN]', text)
    text = URL_PATTERN.sub('[URL]', text)
    text = PHONE_PATTERN.sub('[PHONE]', text) 
    return text

def load_resumes(path : str = RESUMES_CSV)-> pd.DataFrame:
    """
    load resumes from a given path and clean it by dropping duplicates, null values and resetting the index.
    
    Args:
        path (str): path to the resumes file 
    
    Returns:
        pd.DataFrame: loaded resumes DataFrame
    """
    df = pd.read_csv(path)
    resumes_df = df.copy()
    resumes_df = resumes_df.drop_duplicates(subset=['Resume_str'])
    resumes_df = resumes_df.dropna(subset=['Resume_str'])
    resumes_df[RESUME_TEXT_COL] = resumes_df['Resume_str'].apply(mask_pii)
    resumes_df =resumes_df.reset_index(drop=True)

    assert resumes_df[RESUME_TEXT_COL].isna().sum() == 0, "null text survived"
    assert resumes_df["Resume_str"].duplicated().sum() == 0, "duplicates survived"
    assert len(resumes_df.columns) == 5, f"expected 5 columns, got {len(resumes_df.columns)}"
    assert resumes_df.index.equals(pd.RangeIndex(len(resumes_df))), "index has gaps"

    return resumes_df


def load_jobs(path : str = JOBS_EVAL )-> pd.DataFrame:
    """
    load jobs from a given path and it is already cleaned.
    
    Args:
        path (str): path to the jobs file
    Returns:
        pd.DataFrame: loaded jobs DataFrame
    """
    df = pd.read_parquet(path)
    assert JOB_TITLE_COL in df.columns and JOB_TEXT_COL in df.columns, \
    f"jobs file is missing expected columns, got {list(df.columns)}"
    return df

if __name__ == "__main__":
    ref = pd.read_parquet(RESUMES_REFERENCE)
    new = load_resumes(RESUMES_CSV)
    print(len(new), len(ref))
    print(new[RESUME_TEXT_COL].equals(ref[RESUME_TEXT_COL]))
    print(new[RESUME_CAT_COL].equals(ref[RESUME_CAT_COL]))


