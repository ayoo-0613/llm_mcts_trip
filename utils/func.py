from __future__ import annotations

import os
import re
from typing import List, Tuple


def extract_before_parenthesis(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    # drop trailing cost / extra segments: "Name, City; Cost: 12" -> "Name, City"
    s = s.split(";")[0].strip()
    return s


def get_valid_name_city(text: str) -> Tuple[str, str]:
    """
    Parse strings like:
      - "Name, City"
      - "Name, City; Cost: 12"
      - "Name, City; Cost: 321 for 3 nights"
    Return (name, city).
    """
    if text is None:
        return "", ""
    base = extract_before_parenthesis(text)
    if not base:
        return "", ""
    if "," not in base:
        return base.strip(), ""
    # split on last comma to be safer for names containing commas
    name, city = base.rsplit(",", 1)
    return name.strip(), city.strip()


def extract_numbers_from_filenames(path: str) -> List[int]:
    base = os.path.basename(path or "")
    nums = re.findall(r"\d+", base)
    out: List[int] = []
    for n in nums:
        try:
            out.append(int(n))
        except Exception:
            continue
    return out
