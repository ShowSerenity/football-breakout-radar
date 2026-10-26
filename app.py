"""Интерфейс для спортивного директора. Запуск: streamlit run app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dss import POS, RES, WAGE_RATIO, clean_pool, greedy, risk_gain, solve  # noqa: E402

st.set_page_config(page_title="Breakout Radar", layout="wide")
st.title("Breakout Radar: подбор молодых игроков под бюджет")


@st.cache_data
def load():
    return clean_pool(pd.read_csv(RES / "candidates_with_forecast.csv"))


pool_all = load()
tab1, tab2, tab3 = st.tabs(["Подбор состава", "Качество модели", "Скрытые таланты"])

with tab1:
    with st.sidebar:
        st.header("Параметры клуба")
        budget = st.slider("Бюджет на трансферы, млн €", 5, 300, 60, 5) * 1e6
        wage_cap = st.number_input("Лимит недельных зарплат, € (суммарно)",
                                   min_value=1000, value=int(budget * WAGE_RATIO), step=1000)
        d_n = st.number_input("Защитники", 0, 6, 3)
        m_n = st.number_input("Полузащитники", 0, 6, 3)
        a_n = st.number_input("Нападающие", 0, 6, 2)
        risk = st.slider("Осторожность (0 = ожидаемый рост, 1 = пессимистичный)", 0.0, 1.0, 0.5, 0.1)
        max_age = st.slider("Макс. возраст", 17, 23, 21)
        min_ovr = st.slider("Мин. текущий рейтинг", 55, 80, 62)
        leagues = st.multiselect("Лиги (пусто = все)", sorted(pool_all["league_name"].dropna().unique()))
    pool = pool_all[(pool_all["age"] <= max_age) & (pool_all["overall"] >= min_ovr)]
    if leagues:
        pool = pool[pool["league_name"].isin(leagues)]
    pool = pool.reset_index(drop=True)
    slots = {"DEF": int(d_n), "MID": int(m_n), "ATT": int(a_n)}

    def show(sel):
        t = sel.assign(pred_overall_2y=sel["overall"] + sel["pred_gain"])
        cols = ["short_name", "age", "pos_primary", "club_name", "league_name", "overall", "potential",
                "pred_gain", "q20", "q80", "pred_overall_2y", "value_eur", "wage_eur"]
        st.dataframe(t[cols].round(1), width="stretch", hide_index=True)

    if len(pool) == 0 or sum(slots.values()) == 0:
        st.warning("Нет кандидатов под выбранные фильтры.")
    else:
        sel = solve(pool, pool["overall"] + risk_gain(pool, risk), budget, wage_cap, slots)
        base = greedy(pool, "potential", budget, wage_cap, slots)
        if sel is None:
            st.error("Решения нет: ослабьте лимиты или фильтры.")
        else:
            st.subheader("Рекомендация оптимизатора")
            show(sel)
            c1, c2, c3 = st.columns(3)
            c1.metric("Потрачено, млн €", f"{sel['value_eur'].sum() / 1e6:.1f}")
            c2.metric("Прогноз ср. рейтинга через 2 года", f"{(sel['overall'] + sel['pred_gain']).mean():.1f}")
            c3.metric("Ср. прогноз прироста", f"{sel['pred_gain'].mean():.1f}")
            if base is not None:
                st.caption(f"Для сравнения - простой выбор по potential: прогноз ср. рейтинга через 2 года "
                           f"{(base['overall'] + base['pred_gain']).mean():.1f}, "
                           f"потрачено {base['value_eur'].sum() / 1e6:.1f} млн €.")
            else:
                st.caption("Простой выбор по potential не укладывается в эти лимиты.")
            st.subheader("Чувствительность к бюджету")
            rows = []
            for k in [0.6, 0.8, 1.0, 1.2]:
                s = solve(pool, pool["overall"] + risk_gain(pool, risk), budget * k, wage_cap * k, slots)
                rows.append({"бюджет, млн €": round(budget * k / 1e6, 1),
                             "ср. рейтинг через 2 года": None if s is None else round((s["overall"] + s["pred_gain"]).mean(), 2),
                             "игроки": "нет решения" if s is None else ", ".join(s["short_name"])})
            st.table(pd.DataFrame(rows))
        st.caption("Цены и зарплаты - игровые оценки (EA FC), а не реальные трансферные суммы. Прогноз - вероятностный.")

with tab2:
    for f, title in [("metrics.csv", "Метрики модели и baseline"),
                     ("backtest_summary.csv", "Бэктест оптимизатора B-E (FIFA 21 -> FIFA 23)"),
                     ("backtest_pairwise.csv", "Попарное сравнение методов"),
                     ("backtest_vs_A.csv", "Сравнение с простым рейтингом A (где A нашёл решение)"),
                     ("feature_importance.csv", "Важность признаков")]:
        p = RES / f
        if p.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(p).round(3), width="stretch", hide_index=True)

with tab3:
    p = RES / "hidden_gems.csv"
    if p.exists():
        st.write("Молодые недорогие игроки, у которых прогноз роста выше типичного для их возраста и позиции "
                 "(base_gain), с поправкой на нижний квантиль q20.")
        st.dataframe(pd.read_csv(p).round(2), width="stretch", hide_index=True)
        for f, title in [("gems_backtest.csv", "Проверка правила на FIFA 21 -> FIFA 23"),
                         ("gems_backtest_diff.csv", "Разница gem_score и потенциала игры (95% bootstrap CI)")]:
            q = RES / f
            if q.exists():
                st.subheader(title)
                st.dataframe(pd.read_csv(q).round(3), width="stretch", hide_index=True)
        st.caption("Правило отбирает игроков лучше среднего по пулу, но превосходство над простым топом по "
                   "potential-overall не доказано (см. доверительный интервал).")
    else:
        st.info("Запустите: python src/dss.py")