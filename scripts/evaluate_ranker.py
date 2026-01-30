import argparse
import pandas as pd
from scipy.stats import kendalltau, spearmanr  # :contentReference[oaicite:2]{index=2}

from src.f1outcome.models.ranker import load as load_model, FEATURES

def eval_race(race_df: pd.DataFrame, model) -> dict:
    race_df = race_df.dropna(subset=["finishPosition"]).copy()
    if len(race_df) < 5:
        return {}

    X = race_df[FEATURES].fillna(-1)
    race_df["pred_score"] = model.predict(X)

    # True ranking: smaller finishPosition is better
    race_df = race_df.sort_values("finishPosition")
    true_order = race_df["driverId"].tolist()

    # Pred ranking: higher pred_score is better
    pred_order = race_df.sort_values("pred_score", ascending=False)["driverId"].tolist()

    # Convert to ranks for correlation
    true_rank = {d: i for i, d in enumerate(true_order)}
    pred_rank = {d: i for i, d in enumerate(pred_order)}
    common = [d for d in true_order if d in pred_rank]

    t = [true_rank[d] for d in common]
    p = [pred_rank[d] for d in common]

    kt = kendalltau(t, p).correlation
    sp = spearmanr(t, p).correlation

    top5_true = set(true_order[:5])
    top5_pred = set(pred_order[:5])
    top10_true = set(true_order[:10])
    top10_pred = set(pred_order[:10])

    return {
        "kendall_tau": float(kt) if kt is not None else None,
        "spearman": float(sp) if sp is not None else None,
        "top5_overlap": len(top5_true & top5_pred) / 5.0,
        "top10_overlap": len(top10_true & top10_pred) / 10.0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="artifacts/ranker.joblib")
    ap.add_argument("--season", type=int, default=2023)
    ap.add_argument("--min_round", type=int, default=None)
    ap.add_argument("--max_round", type=int, default=None)
    args = ap.parse_args()

    # ✅ 1) LOAD FIRST
    df = pd.read_parquet(args.data)

    # ✅ 2) THEN FILTER
    if args.season is not None:
        df = df[df["season"] == args.season].copy()

    if args.min_round is not None:
        df = df[df["round"] >= args.min_round].copy()

    if args.max_round is not None:
        df = df[df["round"] <= args.max_round].copy()

    model = load_model(args.model)

    metrics = []
    for (season, rnd), race_df in df.groupby(["season", "round"]):
        m = eval_race(race_df, model)
        if m:
            m["season"] = int(season)
            m["round"] = int(rnd)
            metrics.append(m)

    out = pd.DataFrame(metrics).sort_values(["season", "round"])
    print(out.describe(include="number"))
    out.to_csv("artifacts/eval_report.csv", index=False)
    print("Saved -> artifacts/eval_report.csv")


if __name__ == "__main__":
    main()
