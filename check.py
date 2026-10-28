import sys; sys.path.insert(0, "src")
import pandas as pd
from config import RESUMES_REFERENCE, RESUME_TEXT_COL, RESUME_CAT_COL
from data import load_resumes

ref = pd.read_parquet(RESUMES_REFERENCE)
new = load_resumes()

print("columns new:", list(new.columns))
print("columns ref:", list(ref.columns))

# Same rows, in the same order? The ID column answers that.
if "ID" in new.columns and "ID" in ref.columns:
    print("\nIDs identical and in the same order:", (new["ID"].values == ref["ID"].values).all())

for col in (RESUME_TEXT_COL, RESUME_CAT_COL):
    print(f"\n--- {col} ---")
    print("  dtype new / ref :", new[col].dtype, "/", ref[col].dtype)
    print("  equals (strict) :", new[col].equals(ref[col]))
    same = (new[col].values == ref[col].values)
    print("  values equal    :", same.all())
    print("  mismatched rows :", (~same).sum())
    if not same.all():
        i = int((~same).argmax())
        print(f"  first mismatch at row {i}")
        print("    NEW:", repr(str(new[col].iloc[i])[:200]))
        print("    REF:", repr(str(ref[col].iloc[i])[:200]))
