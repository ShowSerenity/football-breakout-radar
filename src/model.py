"""Этап 2. Модель прироста рейтинга за 2 года + baseline + валидация по времени."""
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prepare import ATTRS

PAIRS = "data/pairs.csv"
CAND = "data/candidates_fifa23.csv"
BREAKOUT = 5      # "прорыв" = прирост overall >= 5 за 2 года
TOP_K = 100

NUM = [
    "age", "overall", "potential", "pot_gap", "log_value", "log_wage", "height_cm", "weight_kg",
    "weak_foot", "skill_moves", "international_reputation", "league_level",
    "d_overall_1y", "d_potential_1y",
] + ATTRS
CAT = ["pos_group", "pos_primary", "preferred_foot", "league_name"]
FEATURES = NUM + CAT

PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=42, verbose=-1)


def fit(train_df, alpha=None):
    Xtr = train_df[FEATURES]
    p = dict(PARAMS)
    if alpha is not None:
        p.update(objective="quantile", alpha=alpha)
    return LGBMRegressor(**p).fit(Xtr, train_df["target"])


def metrics(y, pred, name, test):
    order = np.argsort(-pred)[:TOP_K]
    return {
        "model": name,
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred),
        "Spearman": spearmanr(y, pred)[0],
        f"Top{TOP_K}_mean_actual_gain": y.values[order].mean(),
        f"Top{TOP_K}_breakout_hit_rate": (y.values[order] >= BREAKOUT).mean(),
    }


def baselines(train, test):
    out = {}
    out["B0: нулевой рост"] = np.zeros(len(test))
    out["B1: potential-overall (оценка игры)"] = test["pot_gap"].values.astype(float)
    # B2: средний рост по возрасту и группе позиций
    g = train.groupby(["age", "pos_group"])["target"].mean().rename("m").reset_index()
    out["B2: средний рост по возрасту×позиции"] = (
        test[["age", "pos_group"]].merge(g, how="left", on=["age", "pos_group"])["m"]
        .fillna(train["target"].mean()).values)
    # B3: линейная регрессия на 4 признаках
    cols = ["age", "overall", "potential", "pot_gap"]
    lr = LinearRegression().fit(train[cols], train["target"])
    out["B3: линейная регрессия (4 признака)"] = lr.predict(test[cols])
    return out


def main():
    pairs = pd.read_csv(PAIRS)
    cand = pd.read_csv(CAND)

    # единые категории
    allc = pd.concat([pairs[CAT], cand[CAT]])
    for c in CAT:
        cats = allc[c].astype("category").cat.categories
        pairs[c] = pd.Categorical(pairs[c], categories=cats)
        cand[c] = pd.Categorical(cand[c], categories=cats)

    # Разбиение по времени (walk-forward): обучающие пары используют только те цели,
    # которые были известны на момент прогноза (версия-источник N, цель N+2 <= дата прогноза)
    #   Dev:  train N in 15..16 (цели <=18), val N = 18 (цель FIFA 20)
    #   Test: train N in 15..19 (цели <=21), test N = 21 (цель FIFA 23)
    dev_tr, dev_va = pairs[pairs.fifa_version <= 16], pairs[pairs.fifa_version == 18]
    fin_tr, fin_te = pairs[pairs.fifa_version <= 19], pairs[pairs.fifa_version == 21]

    rows = []
    for tag, tr, te in [("dev", dev_tr, dev_va), ("test", fin_tr, fin_te)]:
        m = fit(tr)
        preds = baselines(tr, te)
        preds["LightGBM (наша модель)"] = m.predict(te[FEATURES])
        for name, p in preds.items():
            r = metrics(te["target"], np.asarray(p), name, te)
            r["split"] = tag
            r["n_train"], r["n_eval"] = len(tr), len(te)
            rows.append(r)
        if tag == "test":
            test_pred = te.copy()
            test_pred["pred_gain"] = preds["LightGBM (наша модель)"]
            test_pred["pred_gain_b1"] = preds["B1: potential-overall (оценка игры)"]
            lo, hi = fit(tr, 0.2), fit(tr, 0.8)
            test_pred["q20"], test_pred["q80"] = lo.predict(te[FEATURES]), hi.predict(te[FEATURES])
            cover = ((te["target"] >= test_pred["q20"]) & (te["target"] <= test_pred["q80"])).mean()
            print(f"Покрытие интервала [q20, q80] на тесте: {cover:.3f} (ожидается ~0.60)")
            imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
            imp.sort_values(ascending=False).head(20).to_csv("results/feature_importance.csv")
            # ошибка по лигам
            test_pred["abs_err"] = (test_pred["target"] - test_pred["pred_gain"]).abs()
            lg = test_pred.groupby("league_name", observed=True)["abs_err"].agg(["count", "mean"])
            lg[lg["count"] >= 30].sort_values("mean").to_csv("results/mae_by_league.csv")
            ps = test_pred.groupby("pos_group", observed=True)["abs_err"].agg(["count", "mean"])
            ps.to_csv("results/mae_by_position.csv")

    res = pd.DataFrame(rows)
    res.to_csv("results/metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))

    test_pred.to_csv("results/test_predictions_fifa21_to_23.csv", index=False)

    # Финальная модель: все пары, цель которых известна (N <= 21), прогноз для FIFA 23 -> FIFA 25
    final_tr = pairs[pairs.fifa_version <= 21]
    mean_m, lo, hi = fit(final_tr), fit(final_tr, 0.2), fit(final_tr, 0.8)
    cand["pred_gain"] = mean_m.predict(cand[FEATURES])
    cand["q20"], cand["q80"] = lo.predict(cand[FEATURES]), hi.predict(cand[FEATURES])
    cand["pred_overall_2y"] = cand["overall"] + cand["pred_gain"]
    cand.to_csv("results/candidates_with_forecast.csv", index=False)
    joblib.dump(mean_m, "results/model.joblib")
    json.dump({"features": FEATURES}, open("results/features.json", "w"))
    print("\nСохранено: results/metrics.csv, candidates_with_forecast.csv и др.")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
