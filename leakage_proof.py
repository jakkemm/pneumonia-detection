import re
from itertools import combinations
from pathlib import Path

_IM_RE = re.compile(r"IM-(\d+)")
_PERSON_RE = re.compile(r"^(person\d+)_")

DATA_DIR = Path.cwd() / "data" / "chest_xray"
SPLITS = ("train", "val", "test")


def patient_id(path):
    """
    person99_bacteria_474   -> "person99"
    NORMAL2-IM-0381-0001    -> "IM-0381"
    IM-0011-0001-0002       -> "IM-0011"
    IM-0115-0001            -> "IM-0115"
    """
    stem = Path(path).stem

    m = _PERSON_RE.match(stem)
    if m:
        return m.group(1)
    m = _IM_RE.search(stem)
    if m:
        return f"IM-{m.group(1)}"
    raise ValueError(f"Cannot extract patient ID from {path}")

def leakage_proof():
    dirs = [DATA_DIR / p for p in SPLITS]
    patients = {}
    
    for d in dirs:
        indiv_patients = [patient_id(p) for p in d.glob("**/*.jpeg")]
        patients[d.stem] = set(indiv_patients)
    
    for left, right in combinations(SPLITS, 2):
        overlap = patients[left] & patients[right]
        
        if overlap:
            print(f"Found {len(overlap)} overlapping patients between {left} and {right}.")

if __name__ == "__main__":
    leakage_proof()
