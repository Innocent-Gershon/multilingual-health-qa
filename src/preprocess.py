import re
import pandas as pd

SUBSET_TO_LANG = {
    "Aka_Gha": "Akan",
    "Amh_Eth": "Amharic",
    "Lug_Uga": "Luganda",
    "Swa_Ken": "Swahili",
    "Eng_Uga": "English",
    "Eng_Gha": "English",
    "Eng_Eth": "English",
    "Eng_Ken": "English",
}


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_prompt(question: str, subset: str) -> str:
    lang = SUBSET_TO_LANG.get(subset, "English")
    return f"answer health question in {lang}: {clean_text(question)}"


def load_and_prepare(path: str, is_test: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    # drop the one empty row if present
    df = df[df["input"].notna() & (df["input"].str.strip() != "")]
    if not is_test:
        df = df[df["output"].notna() & (df["output"].str.strip() != "")]
        df["output"] = df["output"].apply(clean_text)
    df["prompt"] = df.apply(lambda r: build_prompt(r["input"], r["subset"]), axis=1)
    df = df.reset_index(drop=True)
    return df
