from rouge_score import rouge_scorer


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1_scores, rl_scores = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
    return {
        "rouge1_f1": round(sum(r1_scores) / len(r1_scores), 4),
        "rougeL_f1": round(sum(rl_scores) / len(rl_scores), 4),
    }


def evaluate_by_subset(predictions: list[str], references: list[str], subsets: list[str]) -> dict:
    from collections import defaultdict
    grouped = defaultdict(lambda: {"preds": [], "refs": []})
    for pred, ref, sub in zip(predictions, references, subsets):
        grouped[sub]["preds"].append(pred)
        grouped[sub]["refs"].append(ref)
    results = {}
    for sub, data in grouped.items():
        results[sub] = compute_rouge(data["preds"], data["refs"])
    return results
