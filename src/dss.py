"""Этап 3. DSS: оптимизатор состава под бюджет, бэктест и поиск "скрытых талантов"."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
POS = ["DEF", "MID", "ATT"]
WAGE_RATIO = 0.004    # допущение: недельный лимит зарплат = 0.4% от бюджета на трансферы
GEM_MIN_OVERALL = 62  # допущение: порог текущего рейтинга для "скрытых талантов"
GEM_Q20_WEIGHT = 0.5  # допущение: вес нижнего квантиля в gem_score
CORE = ("B", "C", "D", "E")


def clean_pool(df):
    d = df.dropna(subset=["value_eur", "wage_eur"]).copy()
    d = d[d["pos_group"].isin(POS)]
    return d.reset_index(drop=True)


def solve(pool, score, budget, wage_cap, slots, time_limit=30):
    """Целочисленная оптимизация: максимум суммы score при лимитах цены, зарплат и числе игроков по линиям."""
    n = len(pool)
    rows = [pool["value_eur"].values, pool["wage_eur"].values]
    lo, hi = [-np.inf, -np.inf], [budget, wage_cap]
    for p in POS:
        rows.append((pool["pos_group"] == p).astype(float).values)
        lo.append(slots[p])
        hi.append(slots[p])
    res = milp(c=-np.asarray(score, dtype=float),
               constraints=LinearConstraint(np.vstack(rows), lo, hi),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-4})
    if res.x is None:
        return None
    return pool[res.x > 0.5]


def greedy(pool, key, budget, wage_cap, slots):
    """Простой ранжированный выбор по key с резервом бюджета под оставшиеся слоты (эвристика, может не найти решение)."""
    chosen, spent, wage, left = [], 0.0, 0.0, dict(slots)
    minv = pool.groupby("pos_group")["value_eur"].min().to_dict()
    minw = pool.groupby("pos_group")["wage_eur"].min().to_dict()
    for idx, r in pool.sort_values(key, ascending=False).iterrows():
        p = r["pos_group"]
        if left[p] == 0:
            continue
        left[p] -= 1
        need_v = sum(minv[q] * n for q, n in left.items())
        need_w = sum(minw[q] * n for q, n in left.items())
        if spent + r["value_eur"] + need_v <= budget and wage + r["wage_eur"] + need_w <= wage_cap:
            chosen.append(idx)
            spent += r["value_eur"]
            wage += r["wage_eur"]
        else:
            left[p] += 1
        if sum(left.values()) == 0:
            break
    return pool.loc[chosen] if sum(left.values()) == 0 else None


def risk_gain(df, risk):
    """risk=0 -> ожидаемый прирост, risk=1 -> пессимистичный (q20)."""
    return (1 - risk) * df["pred_gain"] + risk * df["q20"]


def run_methods(pool, budget, wage_cap, slots, risk=0.5):
    out = {}
    out["A: топ по potential (простой рейтинг)"] = greedy(pool, "potential", budget, wage_cap, slots)
    out["B: оптимизатор + прогноз игры (potential-overall)"] = solve(
        pool, pool["overall"] + pool["pred_gain_b1"], budget, wage_cap, slots)
    out["C: оптимизатор + наша модель"] = solve(
        pool, pool["overall"] + pool["pred_gain"], budget, wage_cap, slots)
    out["D: оптимизатор + наша модель (с учётом риска)"] = solve(
        pool, pool["overall"] + risk_gain(pool, risk), budget, wage_cap, slots)
    out["E: оптимизатор + текущий overall (без прогноза)"] = solve(
        pool, pool["overall"], budget, wage_cap, slots)
    return out


def _compare(piv, a, b, label):
    d = piv[a] - piv[b]
    return {"сравнение": label, "доля побед": (d > 0).mean(),
            "ср. разница рейтинга": d.mean(), "n сценариев": len(d)}


def backtest():
    """Бэктест на данных, невидимых модели: решение на FIFA 21, результат - реальные рейтинги в FIFA 23."""
    pool = clean_pool(pd.read_csv(RES / "test_predictions_fifa21_to_23.csv"))
    configs = {"3-3-2": {"DEF": 3, "MID": 3, "ATT": 2}, "2-3-3": {"DEF": 2, "MID": 3, "ATT": 3}}
    rows, failed = [], []
    for cname, slots in configs.items():
        for base_budget in [20e6, 40e6, 80e6]:
            for mult in [1.0, 0.8]:
                b = base_budget * mult
                for name, sel in run_methods(pool, b, b * WAGE_RATIO, slots).items():
                    if sel is None:
                        failed.append({"config": cname, "budget_M": b / 1e6, "method": name})
                        continue
                    rows.append({"config": cname, "budget_M": b / 1e6, "method": name,
                                 "cost_M": sel["value_eur"].sum() / 1e6,
                                 "avg_overall_FIFA21": sel["overall"].mean(),
                                 "avg_overall_FIFA23_actual": sel["overall_future"].mean(),
                                 "avg_gain_actual": sel["target"].mean(),
                                 "share_declined": (sel["target"] <= 0).mean(),
                                 "worst_gain": sel["target"].min()})
    bt = pd.DataFrame(rows)
    bt["letter"] = bt["method"].str[0]
    bt.drop(columns="letter").to_csv(RES / "backtest.csv", index=False)
    key = ["config", "budget_M"]
    n_all = len(configs) * 3 * 2

    if failed:
        fl = pd.DataFrame(failed)
        print("Сценарии, где метод не нашёл допустимый состав:")
        print(fl.groupby("method").size().to_string(), "\n")

    core = bt[bt["letter"].isin(CORE)]
    core = core[core.groupby(key)["method"].transform("nunique") == len(CORE)]
    if core.empty:
        raise RuntimeError("Нет сценариев, где решены методы B-E")
    n_core = core[key].drop_duplicates().shape[0]
    print(f"Сценариев, где решены все методы B-E: {n_core} из {n_all}")

    metrics = ["cost_M", "avg_overall_FIFA21", "avg_overall_FIFA23_actual",
               "avg_gain_actual", "share_declined", "worst_gain"]
    summ = core.groupby("method")[metrics].mean()
    summ.to_csv(RES / "backtest_summary.csv")

    piv = core.pivot_table(index=key, columns="letter", values="avg_overall_FIFA23_actual")
    pair = pd.DataFrame([
        _compare(piv, "C", "E", "C vs E (вклад модели)"),
        _compare(piv, "C", "B", "C vs B (модель vs оценка игры)"),
        _compare(piv, "D", "C", "D vs C (эффект осторожности)"),
    ])
    pair.to_csv(RES / "backtest_pairwise.csv", index=False)

    a = bt[bt["letter"] == "A"].set_index(key)["avg_overall_FIFA23_actual"]
    overlap = a.index.intersection(piv.index)
    vs_a = pd.DataFrame([{
        "метод": L, "n сценариев": len(overlap),
        "ср. рейтинг FIFA23": piv.loc[overlap, L].mean(),
        "ср. рейтинг A": a.loc[overlap].mean(),
        "доля побед над A": (piv.loc[overlap, L] > a.loc[overlap]).mean(),
        "ср. разница": (piv.loc[overlap, L] - a.loc[overlap]).mean()} for L in CORE])
    vs_a.to_csv(RES / "backtest_vs_A.csv", index=False)
    return bt, summ, pair, vs_a


def _gem_score(df, base_table, fallback):
    d = df.merge(base_table, on=["age", "pos_group"], how="left")
    d["base_gain"] = d["base_gain"].fillna(fallback)
    d["gem_score"] = (d["pred_gain"] - d["base_gain"]) + GEM_Q20_WEIGHT * d["q20"]
    return d


def _gem_filter(d):
    cap = d["value_eur"].quantile(0.75)
    return d[(d["value_eur"] <= cap) & (d["pred_gain"] >= 3) & (d["age"] <= 21)
             & (d["overall"] >= GEM_MIN_OVERALL)]


def hidden_gems(top=25):
    """Молодые недорогие игроки, у которых прогноз роста выше типичного для возраста и позиции (с учётом q20)."""
    c = pd.read_csv(RES / "candidates_with_forecast.csv").dropna(subset=["value_eur"])
    pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    b2 = (pairs.groupby(["age", "pos_group"])["target"].mean()
          .rename("base_gain").reset_index())
    c = _gem_score(c, b2, pairs["target"].mean())
    g = _gem_filter(c)
    cols = ["short_name", "age", "pos_primary", "club_name", "league_name", "overall", "potential",
            "pred_gain", "base_gain", "q20", "q80", "gem_score", "value_eur", "wage_eur"]
    g = g.sort_values("gem_score", ascending=False).head(top)[cols]
    g.to_csv(RES / "hidden_gems.csv", index=False)
    return g


def gems_backtest(tops=(25, 50), n_boot=1000, seed=0):
    """Проверка gem_score на FIFA 21 -> FIFA 23 (base_gain только по парам N<=19, без утечки)."""
    t = pd.read_csv(RES / "test_predictions_fifa21_to_23.csv").dropna(subset=["value_eur"])
    pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    tr = pairs[pairs["fifa_version"] <= 19]
    b2 = tr.groupby(["age", "pos_group"])["target"].mean().rename("base_gain").reset_index()
    g = _gem_filter(_gem_score(t, b2, tr["target"].mean())).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows, diffs = [], []
    for top in tops:
        k = min(top, len(g))
        ours, pot = g.nlargest(k, "gem_score"), g.nlargest(k, "pot_gap")
        for name, s in [("наши gems (gem_score)", ours), ("топ по potential-overall", pot)]:
            rows.append({"top": k, "группа": name, "ср. реальный прирост": s["target"].mean(),
                         "доля прорывов (>=5)": (s["target"] >= 5).mean()})
        rows.append({"top": k, "группа": "весь отфильтрованный пул (= случайный выбор)",
                     "ср. реальный прирост": g["target"].mean(),
                     "доля прорывов (>=5)": (g["target"] >= 5).mean()})

        def diff(df):
            return (df.nlargest(k, "gem_score")["target"].mean()
                    - df.nlargest(k, "pot_gap")["target"].mean())
        boots = [diff(g.sample(len(g), replace=True, random_state=int(rng.integers(1_000_000_000))))
                 for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        diffs.append({"top": k, "gems минус pot_gap": diff(g), "CI95 low": lo, "CI95 high": hi})
    res, dif = pd.DataFrame(rows), pd.DataFrame(diffs)
    res.to_csv(RES / "gems_backtest.csv", index=False)
    dif.to_csv(RES / "gems_backtest_diff.csv", index=False)
    return res, dif


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    bt, summ, pair, vs_a = backtest()
    print("\nБЭКТЕСТ B-E (среднее по сценариям): решение на FIFA 21 -> реальный результат в FIFA 23")
    print(summ.round(3).to_string())
    print("\nПопарное сравнение:")
    print(pair.round(3).to_string(index=False))
    print("\nСравнение с простым рейтингом A (только сценарии, где A нашёл решение):")
    print(vs_a.round(3).to_string(index=False))
    print("\nПодробно по сценариям: results/backtest.csv")
    g = hidden_gems()
    print("\nСКРЫТЫЕ ТАЛАНТЫ (FIFA 23 -> прогноз FIFA 25), топ-10:")
    print(g.head(10).round(2).to_string(index=False))
    res, dif = gems_backtest()
    print("\nБЭКТЕСТ ПРАВИЛА gem_score (FIFA 21 -> FIFA 23):")
    print(res.round(3).to_string(index=False))
    print("\nРазница gems минус pot_gap с bootstrap-интервалом:")
    print(dif.round(3).to_string(index=False))