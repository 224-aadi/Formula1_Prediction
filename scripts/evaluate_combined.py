import argparse
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from src.f1outcome.models.ranker import load as load_ranker, FEATURES
from src.f1outcome.models.dnf import load as load_dnf

def eval_one_race(race_df, ranker, dnf_model, alpha, combine, p_dnf_capt) -> dict:
    race_df = race_df.dropna(subset=["finishPosition"]).copy()
    if len(race_df) < 5:
        return {}

    X = race_df[FEATURES].fillna(-1)

    race_df["score_rank"] = ranker.predict(X)  # raw ranking score; sort within race ([LightGBM docs]) :contentReference[oaicite:0]{index=0}
    race_df["p_dnf"] = dnf_model.predict_proba(X)[:, 1]

    if combine == "subtract":
        race_df["score_adj"] = race_df["score_rank"] - alpha * race_df["p_dnf"]
    elif combine == "subtract_cap":
        p = race_df["p_dnf"].clip(upper=p_dnf_capt)
        race_df["score_adj"] = race_df["score_rank"] - alpha * p
    elif combine == "multiply":
        race_df["score_adj"] = race_df["score_rank"] * (1.0 - race_df["p_dnf"])

    # True order (lower finishPosition is better)
    true_order = race_df.sort_values("finishPosition")["driverId"].tolist()
    # Pred order (higher score_adj is better)
    pred_order = race_df.sort_values("score_adj", ascending=False)["driverId"].tolist()

    true_rank = {d: i for i, d in enumerate(true_order)}
    pred_rank = {d: i for i, d in enumerate(pred_order)}
    common = [d for d in true_order if d in pred_rank]

    t = [true_rank[d] for d in common]
    p = [pred_rank[d] for d in common]

    kt = kendalltau(t, p).correlation
    sp = spearmanr(t, p).correlation

    top5 = len(set(true_order[:5]) & set(pred_order[:5])) / 5.0
    top10 = len(set(true_order[:10]) & set(pred_order[:10])) / 10.0

    return {
        "kendall_tau": float(kt) if kt is not None else None,
        "spearman": float(sp) if sp is not None else None,
        "top5_overlap": top5,
        "top10_overlap": top10,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ranker", required=True)
    ap.add_argument("--dnf", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0, 1, 2, 3, 4, 5, 6, 8, 10])
    ap.add_argument("--combine", choices=["subtract", "subtract_cap", "multiply"], default="subtract_cap")
    ap.add_argument("--p_dnf_cap", type=float, default=0.30)
    ap.add_argument("--select_by", choices=["kendall", "spearman", "top10", "top5"], default="kendall")

    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = df[df["season"] == args.season].copy()

    ranker = load_ranker(args.ranker)
    dnf_model = load_dnf(args.dnf)

    rows = []
    for alpha in args.alphas:
        metrics = []
        for (_, _), race_df in df.groupby(["season", "round"]):
            m = eval_one_race(race_df, ranker, dnf_model, alpha, args.combine, args.p_dnf_cap)
            if m:
                metrics.append(m)
        out = pd.DataFrame(metrics)
        rows.append({
            "alpha": alpha,
            "mean_kendall": out["kendall_tau"].mean(),
            "mean_spearman": out["spearman"].mean(),
            "mean_top5": out["top5_overlap"].mean(),
            "mean_top10": out["top10_overlap"].mean(),
        })

    key = {
    "kendall": "mean_kendall",
    "spearman": "mean_spearman",
    "top10": "mean_top10",
    "top5": "mean_top5",
    }[args.select_by]

    sweep = pd.DataFrame(rows).sort_values(key, ascending=False)

    print(sweep)
    best = sweep.iloc[0].to_dict()
    print("\nBest by", key, ":", best)
    sweep.to_csv("artifacts/alpha_sweep.csv", index=False)
    print("Saved -> artifacts/alpha_sweep.csv")

if __name__ == "__main__":
    main()
