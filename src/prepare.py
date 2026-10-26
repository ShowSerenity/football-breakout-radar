"""Этап 1. Подготовка данных: строим пары "игрок в версии N -> тот же игрок в версии N+2"."""
import numpy as np
import pandas as pd

RAW = "data/male_players_legacy.csv"
OUT_PAIRS = "data/pairs.csv"
OUT_CAND = "data/candidates_fifa23.csv"

HORIZON = 2          # горизонт прогноза в версиях FIFA (лет)
MAX_AGE = 23         # "молодой" игрок
MIN_OVERALL = 60     # отсекаем нерелевантных для скаутинга

ATTRS = [
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve",
    "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
    "power_stamina", "power_strength", "power_long_shots", "mentality_aggression",
    "mentality_interceptions", "mentality_positioning", "mentality_vision",
    "mentality_penalties", "mentality_composure", "defending_marking_awareness",
    "defending_standing_tackle", "defending_sliding_tackle",
]
BASE = [
    "player_id", "short_name", "fifa_version", "player_positions", "overall", "potential",
    "value_eur", "wage_eur", "age", "height_cm", "weight_kg", "league_name", "league_level",
    "club_name", "preferred_foot", "weak_foot", "skill_moves", "international_reputation",
]

DEF = {"CB", "LB", "RB", "LWB", "RWB"}
MID = {"CDM", "CM", "CAM", "LM", "RM"}


def pos_group(p):
    if p == "GK":
        return "GK"
    if p in DEF:
        return "DEF"
    if p in MID:
        return "MID"
    return "ATT"


def load():
    df = pd.read_csv(RAW, usecols=BASE + ATTRS, low_memory=False)
    df["pos_primary"] = df["player_positions"].str.split(",").str[0].str.strip()
    df["pos_group"] = df["pos_primary"].map(pos_group)
    df["pot_gap"] = df["potential"] - df["overall"]
    df["log_value"] = np.log1p(df["value_eur"])
    df["log_wage"] = np.log1p(df["wage_eur"])
    return df


def build():
    df = load()
    df = df[df["pos_group"] != "GK"].copy()   # вратари оцениваются по другим атрибутам -> вне модели
    pairs, cands = [], []
    for n in sorted(df["fifa_version"].unique()):
        a = df[df["fifa_version"] == n].copy()
        prev = df[df["fifa_version"] == n - 1][["player_id", "overall", "potential"]]
        prev = prev.rename(columns={"overall": "overall_prev", "potential": "potential_prev"})
        a = a.merge(prev, on="player_id", how="left")
        a["d_overall_1y"] = a["overall"] - a["overall_prev"]
        a["d_potential_1y"] = a["potential"] - a["potential_prev"]
        a = a[(a["age"] <= MAX_AGE) & (a["overall"] >= MIN_OVERALL)]
        fut = df[df["fifa_version"] == n + HORIZON][["player_id", "overall"]]
        fut = fut.rename(columns={"overall": "overall_future"})
        m = a.merge(fut, on="player_id", how="left")
        if n + HORIZON <= df["fifa_version"].max():
            m["survived"] = m["overall_future"].notna()
            pairs.append(m)
        if n == df["fifa_version"].max():
            cands.append(a)
    pairs = pd.concat(pairs, ignore_index=True)
    # доля исчезнувших из датасета игроков (смещение выжившего) - сохраняем для отчёта
    surv = pairs.groupby("fifa_version")["survived"].mean().rename("share_found_in_t_plus_2")
    surv.to_csv("data/survivorship.csv")
    pairs = pairs[pairs["survived"]].drop(columns="survived")
    pairs["target"] = pairs["overall_future"] - pairs["overall"]
    pairs.to_csv(OUT_PAIRS, index=False)
    pd.concat(cands).to_csv(OUT_CAND, index=False)
    print("pairs:", pairs.shape, "| candidates (latest version):", pd.concat(cands).shape)
    print("\nДоля игроков, найденных в версии N+2 (смещение выжившего):")
    print(surv.round(3).to_string())
    print("\nСтрок пар по версии-источнику:")
    print(pairs.groupby("fifa_version").size().to_string())
    print("\nСтатистика target (прирост overall за 2 года):")
    print(pairs["target"].describe().round(2).to_string())


if __name__ == "__main__":
    build()
