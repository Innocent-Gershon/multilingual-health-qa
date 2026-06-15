import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm

from preprocess import load_and_prepare


def load_model(checkpoint_dir: str, base_model: str = None):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    except Exception:
        base = base_model or "google/mt5-small"
        base_model_obj = AutoModelForSeq2SeqLM.from_pretrained(base)
        model = PeftModel.from_pretrained(base_model_obj, checkpoint_dir)
    model.eval()
    return tokenizer, model


def generate_answers(
    df: pd.DataFrame,
    tokenizer,
    model,
    max_input: int = 128,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    batch_size: int = 16,
    no_repeat_ngram_size: int = 3,
) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    predictions = []

    for i in tqdm(range(0, len(df), batch_size), desc="Generating"):
        batch = df["prompt"].iloc[i : i + batch_size].tolist()
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_input,
            truncation=True,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


def make_submission(
    test_path: str,
    checkpoint_dir: str,
    output_path: str = "submissions/submission.csv",
    base_model: str = None,
):
    tokenizer, model = load_model(checkpoint_dir, base_model)
    test_df = load_and_prepare(test_path, is_test=True)
    preds = generate_answers(test_df, tokenizer, model)
    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "TargetRLF1": preds,
        "TargetR1F1": preds,
        "TargetLLM": preds,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} — {len(submission)} rows")
    return submission


if __name__ == "__main__":
    make_submission(
        test_path="Test.csv",
        checkpoint_dir="outputs/checkpoint",
        output_path="submissions/submission.csv",
    )
