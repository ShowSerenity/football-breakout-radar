"""Проверка качества данных: пропуски, дубликаты, выбросы."""
from pathlib import Path

import pandas as pd

from prepare import ATTRS, MAX_AGE, MIN_OVERALL, load

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PAIRS = ROOT / "data" / "pairs.csv"


def main():
    pd.set_option("display.width", 200)
    RES.mkdir(exist_ok=True)
    df = load()
    print(f"Всего строк (игрок-версия): {len(df)} | версий: {df['fifa_version'].nunique()} "
          f"| уникальных игроков: {df['player_id'].nunique()}")

    dup_key = df.duplicated(["player_id", "fifa_version"]).sum()
    dup_row = df.duplicated().sum()
    print(f"\nДубликаты: по (игрок, версия) = {dup_key}, полных копий строк = {dup_row}")

    young = df[(df["pos_group"] != "GK") & (df["age"] <= MAX_AGE) & (df["overall"] >= MIN_OVERALL)]
    print(f"\nРабочая выборка (до {MAX_AGE} лет, рейтинг >= {MIN_OVERALL}, без вратарей): {len(young)} строк")

    miss = (young.isna().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values(ascending=False).rename("пропусков, %")
    print("\nПропуски в рабочей выборке (только колонки, где они есть):")
    print(miss.head(15).to_string() if len(miss) else "пропусков нет")
    miss.to_csv(RES / "data_quality_missing.csv")

    cols = ["age", "overall", "potential", "value_eur", "wage_eur", "height_cm", "weight_kg"]
    q = young[cols].describe(percentiles=[0.01, 0.5, 0.99]).T[["min", "1%", "50%", "99%", "max"]].round(1)
    print("\nРаспределения (мин, 1%, медиана, 99%, макс):")
    print(q.to_string())
    q.to_csv(RES / "data_quality_ranges.csv")

    checks = {
        "потенциал ниже рейтинга": int((young["potential"] < young["overall"]).sum()),
        "цена = 0 или пропуск": int((young["value_eur"].fillna(0) <= 0).sum()),
        "зарплата = 0 или пропуск": int((young["wage_eur"].fillna(0) <= 0).sum()),
        "навыки вне диапазона 0-100": int(((young[ATTRS] < 0) | (young[ATTRS] > 100)).sum().sum()),
    }
    for col in ["value_eur", "wage_eur"]:
        s = young[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        checks[f"выбросы по {col} (выше Q3 + 3*IQR), доля %"] = round((s > q3 + 3 * (q3 - q1)).mean() * 100, 1)
    print("\nПроверки:")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    if PAIRS.exists():
        pr = pd.read_csv(PAIRS)
        big = (pr["target"].abs() >= 12).sum()
        print(f"\nПары: {len(pr)} | с очень большим изменением рейтинга (|прирост| >= 12): {big} "
              f"({big / len(pr) * 100:.2f}%)")
        print(pr.loc[pr["target"].abs() >= 12, ["short_name", "fifa_version", "age", "overall", "overall_future", "target"]]
              .sort_values("target").head(8).to_string(index=False))


if __name__ == "__main__":
    main()