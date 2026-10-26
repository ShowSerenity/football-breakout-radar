"""Диагностика калибровки квантилей q20/q80 на тесте (FIFA 21 -> FIFA 23)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    t = pd.read_csv(RES / "test_predictions_fifa21_to_23.csv")
    t["above_q20"] = t["target"] >= t["q20"]   # ожидается ~0.80
    t["below_q80"] = t["target"] <= t["q80"]   # ожидается ~0.80
    t["in_interval"] = t["above_q20"] & t["below_q80"]  # ожидается ~0.60
    t["cross"] = t["q20"] > t["q80"]

    print(f"Всего: {len(t)} | пересечение квантилей (q20 > q80): {t['cross'].mean():.3f}")
    print(f"target >= q20: {t['above_q20'].mean():.3f} (ожидается 0.80)")
    print(f"target <= q80: {t['below_q80'].mean():.3f} (ожидается 0.80)")
    print(f"в интервале:   {t['in_interval'].mean():.3f} (ожидается 0.60)")

    groups = {
        "overall": pd.cut(t["overall"], [59, 65, 70, 75, 90]),
        "age": t["age"],
        "pos_group": t["pos_group"],
    }
    out = []
    for name, g in groups.items():
        r = t.groupby(g, observed=True).agg(
            n=("target", "size"),
            above_q20=("above_q20", "mean"),
            below_q80=("below_q80", "mean"),
            in_interval=("in_interval", "mean"),
            mean_target=("target", "mean"),
            mean_q20=("q20", "mean"),
        ).round(3)
        r.insert(0, "group", name)
        r.index = r.index.astype(str)
        out.append(r.reset_index().rename(columns={"index": "bin", name: "bin"}))
        print(f"\nПо {name}:")
        print(r.drop(columns="group").to_string())

    pd.concat(out).to_csv(RES / "quantile_calibration.csv", index=False)


if __name__ == "__main__":
    main()