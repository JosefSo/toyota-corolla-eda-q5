#!/usr/bin/env python3
"""EDA для вопроса №5 по данным Toyota Corolla.

Скрипт намеренно не использует clustering, PCA, регрессию или другие модели.
Он создаёт только описательные таблицы, графики и статистические тесты.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats


# Файловый backend обеспечивает одинаковый результат локально и в Google Colab.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANALYSIS_COLUMNS = [
    "Price",
    "Age_08_04",
    "KM",
    "HP",
    "CC",
    "Weight",
    "Fuel_Type",
    "Automatic",
    "Quarterly_Tax",
    "Doors",
]

NUMERIC_COLUMNS = [
    "Price",
    "Age_08_04",
    "KM",
    "HP",
    "CC",
    "Weight",
    "Quarterly_Tax",
]

FUEL_ORDER = ["Petrol", "Diesel", "CNG"]
FUEL_COLORS = {"Petrol": "#2563EB", "Diesel": "#EA580C", "CNG": "#16A34A"}
SUSPECT_CC_THRESHOLD = 4000

# Порог возраста, выше которого пробег KM <= 1 физически невозможен и
# трактуется как замаскированный пропуск.
IMPLAUSIBLE_KM_AGE_MONTHS = 24

# Сегменты читаются из кросс-таблицы дискретных уровней Fuel_Type x CC x HP.
# Это описательная перекодировка частотной таблицы, а не алгоритм кластеризации.
# Ключи канонические и не зависят от языка; подписи — в SEGMENT_LABELS.
SEGMENT_ORDER = [
    "Petrol 1.3",
    "Petrol 1.4 VVT-i",
    "Petrol 1.6",
    "Petrol 1.8 T-Sport",
    "Diesel 2.0/1.9 old",
    "Diesel D4D new",
    "CNG 1.6",
]

SEGMENT_COLORS = {
    "Petrol 1.3": "#93C5FD",
    "Petrol 1.4 VVT-i": "#3B82F6",
    "Petrol 1.6": "#1D4ED8",
    "Petrol 1.8 T-Sport": "#7C3AED",
    "Diesel 2.0/1.9 old": "#FDBA74",
    "Diesel D4D new": "#EA580C",
    "CNG 1.6": "#16A34A",
}

SEGMENT_LABELS = {
    "ru": {
        "Petrol 1.3": "Petrol 1.3",
        "Petrol 1.4 VVT-i": "Petrol 1.4 VVT-i",
        "Petrol 1.6": "Petrol 1.6",
        "Petrol 1.8 T-Sport": "Petrol 1.8 T-Sport",
        "Diesel 2.0/1.9 old": "Diesel 2.0/1.9 (старое поколение)",
        "Diesel D4D new": "Diesel D4D (новое поколение)",
        "CNG 1.6": "CNG 1.6",
    },
    "en": {
        "Petrol 1.3": "Petrol 1.3",
        "Petrol 1.4 VVT-i": "Petrol 1.4 VVT-i",
        "Petrol 1.6": "Petrol 1.6",
        "Petrol 1.8 T-Sport": "Petrol 1.8 T-Sport",
        "Diesel 2.0/1.9 old": "Diesel 2.0/1.9 (older gen.)",
        "Diesel D4D new": "Diesel D4D (newer gen.)",
        "CNG 1.6": "CNG 1.6",
    },
    "he": {
        "Petrol 1.3": "Petrol 1.3",
        "Petrol 1.4 VVT-i": "Petrol 1.4 VVT-i",
        "Petrol 1.6": "Petrol 1.6",
        "Petrol 1.8 T-Sport": "Petrol 1.8 T-Sport",
        "Diesel 2.0/1.9 old": "Diesel 2.0/1.9 (דור קודם)",
        "Diesel D4D new": "Diesel D4D (דור חדש)",
        "CNG 1.6": "CNG 1.6",
    },
}

# Границы возрастных полос для стратифицированного сравнения цены.
AGE_BAND_EDGES = [0, 30, 45, 60, 70, 80]

# Язык подписей на графиках. Таблицы и ключи данных от него не зависят.
LANG = "ru"

TEXTS = {
    "ru": {
        "cars_count": "Количество автомобилей",
        "median_prefix": "медиана",
        "price": "Цена (Price), €",
        "age": "Возраст (Age_08_04), месяцы",
        "km": "Пробег (KM), км",
        "hp": "Мощность (HP), л. с.",
        "cc": "Объём двигателя (CC), см³",
        "weight": "Вес (Weight), кг",
        "fuel_type": "Тип топлива (Fuel_Type)",
        "fig01_suptitle": "Распределения ключевых признаков жизненного цикла",
        "trend_line": "общая медианная траектория",
        "fig02_title": "Цена резко снижается с возрастом автомобиля",
        "fig03_title": "Больший пробег связан с более низкой ценой",
        "fig04_left": "Объём двигателя и вес по типу топлива",
        "fig04_right": "Мощность и цена по типу топлива",
        "fig04_suptitle": "CC и HP принимают лишь несколько дискретных уровней",
        "jitter_cc": "Объём двигателя (CC), см³ — добавлен джиттер ±22 см³",
        "jitter_hp": "Мощность (HP), л. с. — добавлен джиттер ±1,6 л. с.",
        "excluded_cc": "Из масштаба исключено CC > {threshold}: {count} наблюдение",
        "fig05_suptitle": "Распределения различаются, но группы заметно перекрываются",
        "fig06_title": "Стандартизированные медианные профили типов топлива",
        "fig06_vars": ["Цена", "Возраст", "Пробег", "Мощность", "Объём", "Вес", "Налог"],
        "fig06_cbar": "Z-оценка медианы",
        "fig06_note": (
            "Профили рассчитаны описательно по заранее существующей категории Fuel_Type; "
            "алгоритмы кластеризации не применялись."
        ),
        "fig07_title": "Из {total} возможных сочетаний CC × HP заполнены только {filled}",
        "fig07_cbar": "Количество автомобилей (лог. шкала)",
        "fig07_note": (
            "Заполнено {filled} ячеек из {total}. Сегменты прочитаны из этой таблицы; "
            "алгоритмы кластеризации не применялись."
        ),
        "fig08_title": "Технические сегменты разделяют цену намного сильнее, чем тип топлива",
        "fig08_xlabel": "Технический сегмент (Fuel_Type × CC × HP)",
        "fig08_box": (
            "Kruskal–Wallis для цены:\n"
            "по сегментам  ε² = {segment}  (большой эффект)\n"
            "по Fuel_Type  ε² = {fuel}  (эффекта нет)"
        ),
        "fig09_age_band": "Возрастная полоса (Age_08_04), месяцы",
        "fig09_median_price": "Медианная цена (Price), €",
        "fig09_left": "Линии пересекаются: знак разницы меняется",
        "fig09_annotation": "молодые Diesel дороже",
        "fig09_diff": "Медиана Petrol − медиана Diesel, €",
        "fig09_right": "Разница медиан внутри полосы (синий: Diesel дешевле)",
        "fig09_suptitle": "Общий тест «цена не зависит от топлива» — результат маскировки возрастом",
        "p_small": "p < 0,001",
        "p_value": "p = {value}",
    },
    "en": {
        "cars_count": "Number of cars",
        "median_prefix": "median",
        "price": "Price, €",
        "age": "Age (Age_08_04), months",
        "km": "Mileage (KM), km",
        "hp": "Power (HP), hp",
        "cc": "Engine displacement (CC), cm³",
        "weight": "Weight, kg",
        "fuel_type": "Fuel type (Fuel_Type)",
        "fig01_suptitle": "Distributions of the key lifecycle variables",
        "trend_line": "overall median trajectory",
        "fig02_title": "Price falls sharply as the car gets older",
        "fig03_title": "Higher mileage goes with a lower price",
        "fig04_left": "Engine displacement and weight by fuel type",
        "fig04_right": "Power and price by fuel type",
        "fig04_suptitle": "CC and HP take only a handful of discrete levels",
        "jitter_cc": "Engine displacement (CC), cm³ — jitter of ±22 cm³ added",
        "jitter_hp": "Power (HP), hp — jitter of ±1.6 hp added",
        "excluded_cc": "Excluded from the axis range, CC > {threshold}: {count} observation",
        "fig05_suptitle": "Distributions differ, but the groups overlap substantially",
        "fig06_title": "Standardised median profiles of the fuel types",
        "fig06_vars": ["Price", "Age", "Mileage", "Power", "Displac.", "Weight", "Tax"],
        "fig06_cbar": "Z-score of the median",
        "fig06_note": (
            "Profiles are descriptive summaries of the pre-existing Fuel_Type category; "
            "no clustering algorithm was used."
        ),
        "fig07_title": "Only {filled} of {total} possible CC × HP combinations are populated",
        "fig07_cbar": "Number of cars (log scale)",
        "fig07_note": (
            "{filled} of {total} cells are populated. The segments were read off this table; "
            "no clustering algorithm was used."
        ),
        "fig08_title": "Technical segments separate price far better than fuel type does",
        "fig08_xlabel": "Technical segment (Fuel_Type × CC × HP)",
        "fig08_box": (
            "Kruskal–Wallis on price:\n"
            "by segment    ε² = {segment}  (large effect)\n"
            "by Fuel_Type  ε² = {fuel}  (no effect)"
        ),
        "fig09_age_band": "Age band (Age_08_04), months",
        "fig09_median_price": "Median price, €",
        "fig09_left": "The lines cross: the sign of the gap flips",
        "fig09_annotation": "young diesels cost more",
        "fig09_diff": "Median Petrol − median Diesel, €",
        "fig09_right": "Median gap within each band (blue: Diesel is cheaper)",
        "fig09_suptitle": "The overall «price does not depend on fuel» test is an age-masking artefact",
        "p_small": "p < 0.001",
        "p_value": "p = {value}",
    },
    "he": {
        "cars_count": "מספר רכבים",
        "median_prefix": "חציון",
        "price": "מחיר (Price), €",
        "age": "גיל (Age_08_04), חודשים",
        "km": "קילומטראז' (KM), ק\"מ",
        "hp": "הספק (HP), כ\"ס",
        "cc": "נפח מנוע (CC), סמ\"ק",
        "weight": "משקל (Weight), ק\"ג",
        "fuel_type": "סוג דלק (Fuel_Type)",
        "fig01_suptitle": "התפלגויות המשתנים המרכזיים של מחזור חיי הרכב",
        "trend_line": "מסלול חציוני כללי",
        "fig02_title": "המחיר יורד בחדות עם גיל הרכב",
        "fig03_title": "קילומטראז' גבוה יותר קשור למחיר נמוך יותר",
        "fig04_left": "נפח מנוע ומשקל לפי סוג דלק",
        "fig04_right": "הספק ומחיר לפי סוג דלק",
        "fig04_suptitle": "CC ו-HP מקבלים מספר מצומצם של ערכים בדידים",
        "jitter_cc": "נפח מנוע (CC), סמ\"ק — נוסף פיזור של 22±",
        "jitter_hp": "הספק (HP), כ\"ס — נוסף פיזור של 1.6±",
        "excluded_cc": "הוצא מטווח הצירים, CC > {threshold}: {count} תצפית",
        "fig05_suptitle": "ההתפלגויות שונות, אך הקבוצות חופפות במידה רבה",
        "fig06_title": "פרופילים חציוניים מתוקננים לפי סוג דלק",
        "fig06_vars": ["מחיר", "גיל", "ק\"מ", "הספק", "נפח", "משקל", "מס"],
        "fig06_cbar": "ציון Z של החציון",
        "fig06_note": (
            "הפרופילים מתארים את הקטגוריה הקיימת Fuel_Type; "
            "לא נעשה שימוש באלגוריתם clustering."
        ),
        "fig07_title": "רק {filled} מתוך {total} שילובי CC × HP מאוכלסים",
        "fig07_cbar": "מספר רכבים (סקאלה לוגריתמית)",
        "fig07_note": (
            "{filled} מתוך {total} תאים מאוכלסים. הסגמנטים נקראו מטבלה זו; "
            "לא נעשה שימוש באלגוריתם clustering."
        ),
        "fig08_title": "סגמנטים טכניים מפרידים את המחיר הרבה יותר טוב מסוג הדלק",
        "fig08_xlabel": "סגמנט טכני (Fuel_Type × CC × HP)",
        "fig08_box": (
            "Kruskal–Wallis על המחיר:\n"
            "לפי סגמנט: ε² = {segment} (אפקט גדול)\n"
            "לפי Fuel_Type: ε² = {fuel} (ללא אפקט)"
        ),
        "fig09_age_band": "קבוצת גיל (Age_08_04), חודשים",
        "fig09_median_price": "מחיר חציוני (Price), €",
        "fig09_left": "הקווים מצטלבים: סימן הפער מתהפך",
        "fig09_annotation": "דיזל צעיר יקר יותר",
        "fig09_diff": "חציון Petrol פחות חציון Diesel, €",
        "fig09_right": "פער החציונים בכל קבוצת גיל (כחול: Diesel זול יותר)",
        "fig09_suptitle": "המבחן הכללי לפיו המחיר אינו תלוי בדלק הוא תוצר של מיסוך לפי גיל",
        "p_small": "p < 0.001",
        "p_value": "p = {value}",
    },
}


def _lang_is_rtl() -> bool:
    return LANG == "he"


def shape_rtl(text: str) -> str:
    """Визуальный порядок символов для иврита.

    Matplotlib не умеет bidi: без этой обработки ивритская строка
    выводится задом наперёд.
    """
    if LANG != "he":
        return text
    from bidi.algorithm import get_display

    return "\n".join(get_display(line) for line in text.split("\n"))


def T(key: str, **kwargs):
    """Подпись на текущем языке графиков."""
    value = TEXTS[LANG][key]
    if isinstance(value, list):
        return [shape_rtl(item) for item in value]
    if kwargs:
        value = value.format(**kwargs)
    return shape_rtl(value)


def num(value: float, digits: int = 3) -> str:
    """Число в типографике текущего языка (десятичный разделитель)."""
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if LANG == "ru" else text


def thousands(value: float) -> str:
    """Целое с пробелом в роли разделителя тысяч (одинаково для обоих языков)."""
    return f"{value:,.0f}".replace(",", " ")


def segment_label(key: str) -> str:
    return shape_rtl(SEGMENT_LABELS[LANG][key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Toyota_Corolla_cars.xlsx"),
        help="Путь к исходному Excel-файлу.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Каталог для таблиц, графиков и итогового JSON.",
    )
    parser.add_argument(
        "--lang",
        choices=sorted(TEXTS),
        default="ru",
        help="Язык подписей на графиках (на расчёты и таблицы не влияет).",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.titlesize": 15,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def load_and_validate(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден исходный файл: {path.resolve()}")

    full = pd.read_excel(path, sheet_name="data")
    missing_columns = sorted(set(ANALYSIS_COLUMNS) - set(full.columns))
    if missing_columns:
        raise ValueError(f"В листе data отсутствуют столбцы: {missing_columns}")

    cars = full[ANALYSIS_COLUMNS].copy()
    expected_fuels = set(FUEL_ORDER)
    actual_fuels = set(cars["Fuel_Type"].dropna().unique())
    unexpected_fuels = sorted(actual_fuels - expected_fuels)

    range_checks = {
        "Price > 0": bool((cars["Price"] > 0).all()),
        "0 <= Age_08_04 <= 120": bool(cars["Age_08_04"].between(0, 120).all()),
        "KM >= 0": bool((cars["KM"] >= 0).all()),
        "0 < HP <= 300": bool(cars["HP"].between(1, 300).all()),
        "CC > 0": bool((cars["CC"] > 0).all()),
        "500 <= Weight <= 3000": bool(cars["Weight"].between(500, 3000).all()),
        "Automatic in {0, 1}": bool(set(cars["Automatic"].unique()).issubset({0, 1})),
        "2 <= Doors <= 5": bool(cars["Doors"].between(2, 5).all()),
        "Quarterly_Tax >= 0": bool((cars["Quarterly_Tax"] >= 0).all()),
        "Fuel_Type expected": not unexpected_fuels,
    }

    suspect_cc = cars[cars["CC"] > SUSPECT_CC_THRESHOLD].copy()

    # Модель автомобиля подтверждает, что CC=16000 — опечатка в «1600».
    suspect_cc_models = full.loc[suspect_cc.index, "Model"].tolist() if "Model" in full.columns else []

    # KM <= 1 у автомобиля старше двух лет физически невозможен: это
    # замаскированный пропуск, который проверка «KM >= 0» не ловит.
    implausible_km = cars[
        (cars["KM"] <= 1) & (cars["Age_08_04"] >= IMPLAUSIBLE_KM_AGE_MONTHS)
    ].copy()

    # Id уникален по построению, поэтому дубликаты ищем без него.
    payload = full.drop(columns=[column for column in ["Id"] if column in full.columns])
    payload_no_model = payload.drop(columns=[column for column in ["Model"] if column in payload.columns])

    # Age_08_04 должен в точности воспроизводиться из даты выпуска.
    if {"Mfg_Year", "Mfg_Month"}.issubset(full.columns):
        derived_age = (2004 - full["Mfg_Year"]) * 12 + (8 - full["Mfg_Month"]) + 1
        age_mismatches = int((derived_age != full["Age_08_04"]).sum())
    else:
        age_mismatches = -1

    quality_rows = [
        ("rows", len(cars), "OK" if len(cars) == 1436 else "CHECK"),
        ("analysis_columns", len(cars.columns), "OK"),
        ("missing_values", int(cars.isna().sum().sum()), "OK" if not cars.isna().any().any() else "CHECK"),
        (
            "full_row_duplicates_without_Id",
            int(payload.duplicated().sum()),
            "OK" if not payload.duplicated().any() else "CHECK",
        ),
        (
            "duplicates_without_Id_and_Model",
            int(payload_no_model.duplicated().sum()),
            "INFO",
        ),
        (
            "duplicates_on_analysis_columns",
            int(cars.duplicated().sum()),
            "INFO",
        ),
        (
            "Age_08_04_vs_Mfg_date_mismatches",
            age_mismatches,
            "OK" if age_mismatches == 0 else "CHECK",
        ),
        ("unexpected_fuel_categories", ", ".join(unexpected_fuels) or "none", "OK" if not unexpected_fuels else "CHECK"),
        ("suspect_CC_gt_4000", len(suspect_cc), "CHECK" if len(suspect_cc) else "OK"),
        (
            "suspect_CC_model_names",
            "; ".join(suspect_cc_models) or "none",
            "INFO",
        ),
        (
            f"implausible_KM_le_1_and_Age_ge_{IMPLAUSIBLE_KM_AGE_MONTHS}",
            len(implausible_km),
            "CHECK" if len(implausible_km) else "OK",
        ),
    ]
    quality_rows.extend(
        (f"range_check: {label}", value, "OK" if value else "CHECK")
        for label, value in range_checks.items()
    )
    quality = pd.DataFrame(quality_rows, columns=["Metric", "Value", "Status"])

    if len(cars) != 1436:
        raise ValueError(f"Ожидалось 1436 строки, получено {len(cars)}")
    if cars.isna().any().any():
        raise ValueError("В анализируемых столбцах обнаружены пропуски")
    if payload.duplicated().any():
        raise ValueError("В исходном листе обнаружены полные дубликаты строк")
    failed_ranges = [name for name, passed in range_checks.items() if not passed]
    if failed_ranges:
        raise ValueError(f"Не пройдены проверки диапазонов: {failed_ranges}")

    return cars, quality, suspect_cc, implausible_km


def descriptive_statistics(cars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_COLUMNS + ["Automatic", "Doors"]:
        values = cars[column]
        q1, median, q3 = values.quantile([0.25, 0.50, 0.75])
        rows.append(
            {
                "Variable": column,
                "N": values.count(),
                "Mean": values.mean(),
                "Std": values.std(ddof=1),
                "Min": values.min(),
                "Q1": q1,
                "Median": median,
                "Q3": q3,
                "Max": values.max(),
                "IQR": q3 - q1,
            }
        )
    return pd.DataFrame(rows).round(3)


def fuel_profiles(cars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    profile_variables = ["Price", "Age_08_04", "KM", "HP", "CC", "Weight", "Quarterly_Tax", "Doors"]
    for fuel in FUEL_ORDER:
        group = cars[cars["Fuel_Type"] == fuel]
        row: dict[str, float | int | str] = {
            "Fuel_Type": fuel,
            "N": len(group),
            "Share_pct": 100 * len(group) / len(cars),
            "Automatic_pct": 100 * group["Automatic"].mean(),
        }
        for variable in profile_variables:
            row[f"{variable}_Q1"] = group[variable].quantile(0.25)
            row[f"{variable}_Median"] = group[variable].median()
            row[f"{variable}_Q3"] = group[variable].quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def epsilon_squared(h_statistic: float, n: int, k: int = 0) -> float:
    """Классическая ε² для Kruskal–Wallis: H / (n - 1).

    Параметр k сохранён для совместимости вызовов и в формуле не участвует.
    """
    return max(0.0, h_statistic / (n - 1))


def rank_biserial_from_u(u_statistic: float, n1: int, n2: int) -> float:
    return 2 * u_statistic / (n1 * n2) - 1


def effect_label(value: float, kind: str) -> str:
    """Словесная оценка размера эффекта.

    Для ε² используются пороги Cohen (0,01 / 0,06 / 0,14).
    Для rank-biserial — конвенция Cohen для r (0,10 / 0,30 / 0,50);
    это распространённая, но не единственная шкала интерпретации.
    """
    value = abs(value)
    if kind == "epsilon_squared":
        if value < 0.01:
            return "пренебрежимо малый"
        if value < 0.06:
            return "малый"
        if value < 0.14:
            return "средний"
        return "большой"
    if value < 0.10:
        return "пренебрежимо малый"
    if value < 0.30:
        return "малый"
    if value < 0.50:
        return "средний"
    return "большой"


def spearman_results(cars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = ["Age_08_04", "KM", "HP", "CC", "Weight", "Quarterly_Tax"]
    rows = []
    for predictor in predictors:
        rho, p_value = stats.spearmanr(cars["Price"], cars[predictor])
        rows.append(
            {
                "Variable_1": "Price",
                "Variable_2": predictor,
                "Spearman_rho": rho,
                "p_raw": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"].to_numpy())
    result["Significant_0_05"] = result["p_holm"] < 0.05
    result["Abs_rho"] = result["Spearman_rho"].abs()
    result = result.sort_values("Abs_rho", ascending=False).reset_index(drop=True)

    matrix_variables = ["Price", "Age_08_04", "KM", "HP", "CC", "Weight", "Quarterly_Tax"]
    matrix = cars[matrix_variables].corr(method="spearman")
    return result, matrix


def kruskal_results(cars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = ["Price", "KM", "HP", "CC", "Weight", "Quarterly_Tax"]
    omnibus_rows = []
    for variable in variables:
        groups = [cars.loc[cars["Fuel_Type"] == fuel, variable].to_numpy() for fuel in FUEL_ORDER]
        h_statistic, p_value = stats.kruskal(*groups)
        eps2 = epsilon_squared(h_statistic, len(cars), len(groups))
        omnibus_rows.append(
            {
                "Variable": variable,
                "H": h_statistic,
                "df": len(groups) - 1,
                "p_raw": p_value,
                "epsilon_squared": eps2,
                "Effect_size": effect_label(eps2, "epsilon_squared"),
            }
        )

    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["p_holm"] = holm_adjust(omnibus["p_raw"].to_numpy())
    omnibus["Significant_0_05"] = omnibus["p_holm"] < 0.05
    omnibus = omnibus.sort_values("epsilon_squared", ascending=False).reset_index(drop=True)

    significant_variables = omnibus.loc[omnibus["Significant_0_05"], "Variable"].tolist()
    pairwise_rows = []
    pairs = [("Petrol", "Diesel"), ("Petrol", "CNG"), ("Diesel", "CNG")]
    for variable in significant_variables:
        for fuel_1, fuel_2 in pairs:
            values_1 = cars.loc[cars["Fuel_Type"] == fuel_1, variable].to_numpy()
            values_2 = cars.loc[cars["Fuel_Type"] == fuel_2, variable].to_numpy()
            u_statistic, p_value = stats.mannwhitneyu(
                values_1,
                values_2,
                alternative="two-sided",
                method="auto",
            )
            rbc = rank_biserial_from_u(u_statistic, len(values_1), len(values_2))
            pairwise_rows.append(
                {
                    "Variable": variable,
                    "Group_1": fuel_1,
                    "Group_2": fuel_2,
                    "N_1": len(values_1),
                    "N_2": len(values_2),
                    "Median_1": np.median(values_1),
                    "Median_2": np.median(values_2),
                    "U": u_statistic,
                    "p_raw": p_value,
                    "rank_biserial": rbc,
                    "abs_rank_biserial": abs(rbc),
                    "Effect_size": effect_label(rbc, "rank_biserial"),
                }
            )

    pairwise = pd.DataFrame(pairwise_rows)
    if not pairwise.empty:
        pairwise["p_holm"] = holm_adjust(pairwise["p_raw"].to_numpy())
        pairwise["Significant_0_05"] = pairwise["p_holm"] < 0.05
        pairwise = pairwise.sort_values(
            ["Variable", "p_holm", "abs_rank_biserial"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    return omnibus, pairwise


def assign_segment(row: pd.Series) -> str:
    """Сегмент автомобиля по дискретным уровням Fuel_Type / CC / HP.

    Правила прочитаны из кросс-таблицы частот (см. configuration_crosstab):
    и CC, и HP принимают лишь по десятку значений, образующих отчётливые
    технические конфигурации. Алгоритмы кластеризации не применяются.
    """
    fuel, cc, hp = row["Fuel_Type"], row["CC"], row["HP"]
    if fuel == "Diesel":
        return "Diesel D4D new" if hp >= 90 else "Diesel 2.0/1.9 old"
    if fuel == "CNG":
        return "CNG 1.6"
    if cc <= 1332:
        return "Petrol 1.3"
    if cc <= 1400:
        return "Petrol 1.4 VVT-i"
    if hp >= 150:
        return "Petrol 1.8 T-Sport"
    # Сюда же попадает единственная строка с опечаткой CC=16000:
    # поле Model подтверждает, что это 1.6.
    return "Petrol 1.6"


def with_segments(cars: pd.DataFrame) -> pd.DataFrame:
    return cars.assign(Segment=cars.apply(assign_segment, axis=1))


def configuration_crosstab(cars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Частотные таблицы, из которых выводятся сегменты."""
    crosstab = pd.crosstab(cars["CC"], cars["HP"])
    configurations = (
        cars.groupby(["Fuel_Type", "CC", "HP"])
        .agg(
            N=("Price", "size"),
            Price_Median=("Price", "median"),
            Age_Median=("Age_08_04", "median"),
            KM_Median=("KM", "median"),
            Weight_Median=("Weight", "median"),
            Quarterly_Tax_Median=("Quarterly_Tax", "median"),
        )
        .sort_values("N", ascending=False)
        .reset_index()
    )
    configurations["Share_pct"] = (100 * configurations["N"] / len(cars)).round(2)
    configurations["Cumulative_share_pct"] = configurations["Share_pct"].cumsum().round(2)
    return crosstab, configurations


def segment_profiles(cars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Профили сегментов и сравнение их разделяющей силы с Fuel_Type."""
    data = with_segments(cars)
    profile = (
        data.groupby("Segment")
        .agg(
            N=("Price", "size"),
            Price_Q1=("Price", lambda values: values.quantile(0.25)),
            Price_Median=("Price", "median"),
            Price_Q3=("Price", lambda values: values.quantile(0.75)),
            Age_Median=("Age_08_04", "median"),
            KM_Median=("KM", "median"),
            HP_Median=("HP", "median"),
            CC_Median=("CC", "median"),
            Weight_Median=("Weight", "median"),
            Quarterly_Tax_Median=("Quarterly_Tax", "median"),
            Automatic_pct=("Automatic", lambda values: 100 * values.mean()),
        )
        .reindex(SEGMENT_ORDER)
        .reset_index()
    )
    profile.insert(2, "Share_pct", (100 * profile["N"] / len(data)).round(2))

    rows = []
    for label, grouping, frame in [
        ("Fuel_Type", "Fuel_Type", data),
        ("Segment (Fuel x CC x HP)", "Segment", data),
    ]:
        groups = [group["Price"].to_numpy() for _, group in frame.groupby(grouping)]
        h_statistic, p_value = stats.kruskal(*groups)
        rows.append(
            {
                "Grouping": label,
                "k": len(groups),
                "H": h_statistic,
                "df": len(groups) - 1,
                "p_value": p_value,
                "epsilon_squared": epsilon_squared(h_statistic, len(frame)),
                "Effect_size": effect_label(epsilon_squared(h_statistic, len(frame)), "epsilon_squared"),
            }
        )

    # Контроль возраста: сегменты должны разделять цену и внутри узкой полосы.
    narrow = data[data["Age_08_04"].between(45, 70)]
    narrow_groups = [
        group["Price"].to_numpy()
        for _, group in narrow.groupby("Segment")
        if len(group) >= 5
    ]
    h_statistic, p_value = stats.kruskal(*narrow_groups)
    rows.append(
        {
            "Grouping": "Segment | возраст 45-70 мес",
            "k": len(narrow_groups),
            "H": h_statistic,
            "df": len(narrow_groups) - 1,
            "p_value": p_value,
            "epsilon_squared": epsilon_squared(h_statistic, len(narrow)),
            "Effect_size": effect_label(epsilon_squared(h_statistic, len(narrow)), "epsilon_squared"),
        }
    )
    return profile, pd.DataFrame(rows)


def stratified_price(cars: pd.DataFrame) -> pd.DataFrame:
    """Сравнение цены Petrol и Diesel внутри возрастных полос.

    Без стратификации возраст выступает конфаундером и маскирует
    противоположно направленные различия.
    """
    bands = pd.cut(cars["Age_08_04"], AGE_BAND_EDGES)
    rows = []
    for band, group in cars.assign(AgeBand=bands).groupby("AgeBand", observed=True):
        petrol = group.loc[group["Fuel_Type"] == "Petrol", "Price"].to_numpy()
        diesel = group.loc[group["Fuel_Type"] == "Diesel", "Price"].to_numpy()
        if len(diesel) < 5 or len(petrol) < 5:
            continue
        u_statistic, p_value = stats.mannwhitneyu(petrol, diesel, alternative="two-sided")
        rbc = rank_biserial_from_u(u_statistic, len(petrol), len(diesel))
        rows.append(
            {
                "Age_band_months": f"{int(band.left) + 1}-{int(band.right)}",
                "N_Petrol": len(petrol),
                "N_Diesel": len(diesel),
                "Median_Petrol": float(np.median(petrol)),
                "Median_Diesel": float(np.median(diesel)),
                "Median_difference": float(np.median(petrol) - np.median(diesel)),
                "U": u_statistic,
                "p_raw": p_value,
                "rank_biserial": rbc,
                "Effect_size": effect_label(rbc, "rank_biserial"),
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"].to_numpy())
    result["Significant_0_05"] = result["p_holm"] < 0.05
    result["Cheaper_group"] = np.where(
        result["Median_difference"] > 0, "Diesel дешевле", "Petrol дешевле"
    )
    return result


def diesel_generations(cars: pd.DataFrame) -> pd.DataFrame:
    """Разбивка Diesel по мощности: медиана скрывает два поколения."""
    diesel = cars[cars["Fuel_Type"] == "Diesel"]
    return (
        diesel.groupby("HP")
        .agg(
            N=("Price", "size"),
            Price_Median=("Price", "median"),
            Age_Median=("Age_08_04", "median"),
            KM_Median=("KM", "median"),
            Weight_Median=("Weight", "median"),
            Quarterly_Tax_Median=("Quarterly_Tax", "median"),
        )
        .reset_index()
    )


def secondary_factors(cars: pd.DataFrame) -> pd.DataFrame:
    """Проверка второстепенных категорий: Doors и Automatic.

    Doors даёт малый, но устойчивый эффект (сохраняется при контроле
    возраста), Automatic — пренебрежимо малый. Оба на порядок слабее
    технических сегментов, поэтому в основную группировку не входят.
    """
    rows = []
    for column in ["Doors", "Automatic"]:
        prices = [group["Price"].to_numpy() for _, group in cars.groupby(column)]
        h_statistic, p_value = stats.kruskal(*prices)
        # Тот же тест внутри узкой возрастной полосы: если эффект исчезает,
        # он был проявлением возраста, а не самостоятельным.
        narrow = cars[cars["Age_08_04"].between(45, 70)]
        narrow_prices = [
            group["Price"].to_numpy()
            for _, group in narrow.groupby(column)
            if len(group) >= 5
        ]
        h_narrow, p_narrow = stats.kruskal(*narrow_prices)
        for value, group in cars.groupby(column):
            rows.append(
                {
                    "Variable": column,
                    "Level": value,
                    "N": len(group),
                    "Price_Median": group["Price"].median(),
                    "Age_Median": group["Age_08_04"].median(),
                    "H_all": h_statistic,
                    "p_all": p_value,
                    "epsilon_squared_all": epsilon_squared(h_statistic, len(cars)),
                    "epsilon_squared_age_45_70": epsilon_squared(h_narrow, len(narrow)),
                    "p_age_45_70": p_narrow,
                }
            )
    return pd.DataFrame(rows)


def km_sensitivity(cars: pd.DataFrame, implausible_km: pd.DataFrame) -> pd.DataFrame:
    """Устойчивость связи «пробег — цена» к замаскированным пропускам KM."""
    rows = []
    for label, data in [
        ("all_data", cars),
        (f"without_implausible_KM ({len(implausible_km)} rows)", cars.drop(index=implausible_km.index)),
    ]:
        rho, p_value = stats.spearmanr(data["Price"], data["KM"])
        rows.append(
            {
                "Scenario": label,
                "N": len(data),
                "KM_min": data["KM"].min(),
                "KM_median": data["KM"].median(),
                "Spearman_rho_Price_KM": rho,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def cc_sensitivity(cars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, data in [
        ("all_data", cars),
        (f"without_CC_gt_{SUSPECT_CC_THRESHOLD}", cars[cars["CC"] <= SUSPECT_CC_THRESHOLD]),
    ]:
        groups = [data.loc[data["Fuel_Type"] == fuel, "CC"].to_numpy() for fuel in FUEL_ORDER]
        h_statistic, p_value = stats.kruskal(*groups)
        rows.append(
            {
                "Scenario": label,
                "N": len(data),
                "CC_max": data["CC"].max(),
                "CC_mean": data["CC"].mean(),
                "CC_median": data["CC"].median(),
                "Kruskal_H": h_statistic,
                "p_value": p_value,
                "epsilon_squared": epsilon_squared(h_statistic, len(data), len(groups)),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distributions(cars: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    specs = [
        ("Price", T("price"), "#2563EB", 28),
        ("Age_08_04", T("age"), "#7C3AED", 20),
        ("KM", T("km"), "#EA580C", 28),
    ]
    for ax, (variable, xlabel, color, bins) in zip(axes, specs):
        values = cars[variable]
        ax.hist(values, bins=bins, color=color, alpha=0.78, edgecolor="white")
        median = values.median()
        ax.axvline(median, color="#111827", linestyle="--", linewidth=1.8, label=f"{T('median_prefix')}: {thousands(median)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(T("cars_count"))
        ax.legend(frameon=False)
    fig.suptitle(T("fig01_suptitle"))
    fig.tight_layout()
    save_figure(fig, path)


def add_fuel_scatter(
    ax: plt.Axes,
    cars: pd.DataFrame,
    x: str,
    y: str,
    x_jitter: float = 0.0,
) -> None:
    """Точечная диаграмма по типам топлива.

    x_jitter > 0 добавляет равномерный сдвиг по оси X: CC и HP дискретны,
    и без сдвига сотни наблюдений сливаются в одну вертикальную линию.
    """
    rng = np.random.default_rng(20240819)
    for fuel in FUEL_ORDER:
        group = cars[cars["Fuel_Type"] == fuel]
        x_values = group[x].to_numpy(dtype=float)
        if x_jitter:
            x_values = x_values + rng.uniform(-x_jitter, x_jitter, size=len(x_values))
        ax.scatter(
            x_values,
            group[y],
            s=23 if fuel != "CNG" else 42,
            alpha=0.48 if fuel == "Petrol" else 0.72,
            color=FUEL_COLORS[fuel],
            edgecolor="none",
            label=f"{fuel} (n={len(group)})",
        )


def plot_price_age(cars: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    add_fuel_scatter(ax, cars, "Age_08_04", "Price")
    age_bins = np.arange(0, cars["Age_08_04"].max() + 11, 10)
    binned = cars.assign(Age_bin=pd.cut(cars["Age_08_04"], age_bins, include_lowest=True))
    trend = binned.groupby("Age_bin", observed=True).agg(Age=("Age_08_04", "median"), Price=("Price", "median"))
    ax.plot(trend["Age"], trend["Price"], color="#111827", linewidth=2.5, marker="o", label=T("trend_line"))
    ax.set_title(T("fig02_title"))
    ax.set_xlabel(T("age"))
    ax.set_ylabel(T("price"))
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    save_figure(fig, path)


def plot_price_km(cars: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    add_fuel_scatter(ax, cars, "KM", "Price")
    quantile_bins = pd.qcut(cars["KM"], q=10, duplicates="drop")
    trend = cars.assign(KM_bin=quantile_bins).groupby("KM_bin", observed=True).agg(KM=("KM", "median"), Price=("Price", "median"))
    ax.plot(trend["KM"], trend["Price"], color="#111827", linewidth=2.5, marker="o", label=T("trend_line"))
    ax.set_title(T("fig03_title"))
    ax.set_xlabel(T("km"))
    ax.set_ylabel(T("price"))
    ax.legend(frameon=True, facecolor="white")
    fig.tight_layout()
    save_figure(fig, path)


def plot_technical(cars: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    clean_cc = cars[cars["CC"] <= SUSPECT_CC_THRESHOLD]
    add_fuel_scatter(axes[0], clean_cc, "CC", "Weight", x_jitter=22)
    axes[0].set_title(T("fig04_left"))
    axes[0].set_xlabel(T("jitter_cc"))
    axes[0].set_ylabel(T("weight"))
    axes[0].text(
        0.02,
        0.98,
        T("excluded_cc", threshold=SUSPECT_CC_THRESHOLD, count=len(cars) - len(clean_cc)),
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
        color="#991B1B",
    )
    add_fuel_scatter(axes[1], cars, "HP", "Price", x_jitter=1.6)
    axes[1].set_title(T("fig04_right"))
    axes[1].set_xlabel(T("jitter_hp"))
    axes[1].set_ylabel(T("price"))
    for ax in axes:
        ax.legend(frameon=True, facecolor="white")
    fig.suptitle(T("fig04_suptitle"))
    fig.tight_layout()
    save_figure(fig, path)


def plot_fuel_boxplots(cars: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    specs = [
        ("Price", T("price")),
        ("KM", T("km")),
        ("Weight", T("weight")),
    ]
    labels = [f"{fuel}\n(n={(cars['Fuel_Type'] == fuel).sum()})" for fuel in FUEL_ORDER]
    for ax, (variable, ylabel) in zip(axes, specs):
        values = [cars.loc[cars["Fuel_Type"] == fuel, variable].to_numpy() for fuel in FUEL_ORDER]
        box = ax.boxplot(values, patch_artist=True, showfliers=True)
        ax.set_xticks(range(1, len(labels) + 1), labels=labels)
        for patch, fuel in zip(box["boxes"], FUEL_ORDER):
            patch.set_facecolor(FUEL_COLORS[fuel])
            patch.set_alpha(0.58)
        for median_line in box["medians"]:
            median_line.set_color("#111827")
            median_line.set_linewidth(2)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(T("fuel_type"))
    fig.suptitle(T("fig05_suptitle"))
    fig.tight_layout()
    save_figure(fig, path)


def plot_standardized_profiles(cars: pd.DataFrame, path: Path) -> pd.DataFrame:
    variables = ["Price", "Age_08_04", "KM", "HP", "CC", "Weight", "Quarterly_Tax"]
    medians = cars.groupby("Fuel_Type")[variables].median().reindex(FUEL_ORDER)
    standardized = (medians - cars[variables].mean()) / cars[variables].std(ddof=1)

    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    limit = float(max(2.0, np.nanmax(np.abs(standardized.to_numpy()))))
    image = ax.imshow(standardized.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(variables)), labels=T("fig06_vars"))
    ax.set_yticks(range(len(FUEL_ORDER)), labels=FUEL_ORDER)
    ax.set_title(T("fig06_title"))
    for row in range(standardized.shape[0]):
        for column in range(standardized.shape[1]):
            value = standardized.iloc[row, column]
            text_color = "white" if abs(value) > limit * 0.48 else "#111827"
            ax.text(column, row, f"{value:+.2f}", ha="center", va="center", color=text_color, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.035)
    colorbar.set_label(T("fig06_cbar"))
    fig.text(
        0.5,
        0.01,
        T("fig06_note"),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, path)
    return standardized


def plot_configuration_heatmap(cars: pd.DataFrame, path: Path) -> None:
    """Кросс-таблица CC x HP: прямое доказательство дискретных конфигураций."""
    clean = cars[cars["CC"] <= SUSPECT_CC_THRESHOLD]
    table = pd.crosstab(clean["CC"], clean["HP"])

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    masked = np.ma.masked_where(table.to_numpy() == 0, table.to_numpy())
    colormap = plt.get_cmap("Blues").copy()
    colormap.set_bad("#F3F4F6")
    image = ax.imshow(masked, cmap=colormap, aspect="auto", norm=matplotlib.colors.LogNorm(vmin=1, vmax=table.to_numpy().max()))

    ax.set_xticks(range(len(table.columns)), labels=table.columns)
    ax.set_yticks(range(len(table.index)), labels=table.index)
    ax.set_xlabel(T("hp"))
    ax.set_ylabel(T("cc"))
    ax.set_title(T("fig07_title", total=table.size, filled=int((table.to_numpy() > 0).sum())))
    ax.grid(False)

    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            count = table.iloc[row, column]
            if count == 0:
                continue
            ax.text(
                column,
                row,
                f"{count}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if count > 60 else "#111827",
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)
    colorbar.set_label(T("fig07_cbar"))
    fig.text(
        0.5,
        0.005,
        T("fig07_note", filled=int((table.to_numpy() > 0).sum()), total=table.size),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, path)


def plot_price_by_segment(cars: pd.DataFrame, path: Path) -> None:
    """Главный график ответа: цена по техническим сегментам."""
    data = with_segments(cars)
    order = (
        data.groupby("Segment")["Price"].median().sort_values().index.tolist()
    )
    values = [data.loc[data["Segment"] == segment, "Price"].to_numpy() for segment in order]
    # Медиана вынесена в подпись оси: внутри «ящика» она перекрывала линию медианы.
    labels = [
        f"{segment_label(segment)}\nn={len(values[position])} · "
        f"{T('median_prefix')} {thousands(np.median(values[position]))} €"
        for position, segment in enumerate(order)
    ]

    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    box = ax.boxplot(values, patch_artist=True, showfliers=True, widths=0.62)
    for patch, segment in zip(box["boxes"], order):
        patch.set_facecolor(SEGMENT_COLORS[segment])
        patch.set_alpha(0.75)
    for median_line in box["medians"]:
        median_line.set_color("#111827")
        median_line.set_linewidth(2.2)

    ax.set_xticks(range(1, len(labels) + 1), labels=labels, fontsize=9)
    ax.set_ylabel(T("price"))
    ax.set_xlabel(T("fig08_xlabel"))
    ax.set_title(T("fig08_title"))

    groups = [group["Price"].to_numpy() for _, group in data.groupby("Segment")]
    h_statistic, _ = stats.kruskal(*groups)
    fuel_groups = [group["Price"].to_numpy() for _, group in data.groupby("Fuel_Type")]
    h_fuel, _ = stats.kruskal(*fuel_groups)
    ax.text(
        0.015,
        0.97,
        T(
            "fig08_box",
            segment=num(epsilon_squared(h_statistic, len(data))),
            fuel=num(epsilon_squared(h_fuel, len(data))),
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "boxstyle": "round,pad=0.5"},
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_stratified_price(cars: pd.DataFrame, path: Path) -> None:
    """Разворот знака разницы цен Petrol/Diesel при контроле возраста."""
    bands = pd.cut(cars["Age_08_04"], AGE_BAND_EDGES)
    data = cars.assign(AgeBand=bands)
    band_labels = [f"{int(b.left) + 1}-{int(b.right)}" for b in data["AgeBand"].cat.categories]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    for fuel in ["Petrol", "Diesel"]:
        medians = (
            data[data["Fuel_Type"] == fuel]
            .groupby("AgeBand", observed=False)["Price"]
            .median()
        )
        counts = data[data["Fuel_Type"] == fuel].groupby("AgeBand", observed=False)["Price"].size()
        axes[0].plot(
            range(len(band_labels)),
            medians.to_numpy(),
            marker="o",
            linewidth=2.6,
            markersize=8,
            color=FUEL_COLORS[fuel],
            label=f"{fuel} (n={counts.sum()})",
        )
    axes[0].set_xticks(range(len(band_labels)), labels=band_labels)
    axes[0].set_xlabel(T("fig09_age_band"))
    axes[0].set_ylabel(T("fig09_median_price"))
    axes[0].set_title(T("fig09_left"))
    axes[0].legend(frameon=True, facecolor="white")
    axes[0].annotate(
        T("fig09_annotation"),
        xy=(0, 19600),
        xytext=(0.12, 0.86),
        textcoords="axes fraction",
        fontsize=9,
        color="#991B1B",
        arrowprops={"arrowstyle": "->", "color": "#991B1B"},
    )

    stratified = stratified_price(cars)
    difference = stratified["Median_difference"].to_numpy()
    colors = ["#EA580C" if value < 0 else "#2563EB" for value in difference]
    axes[1].bar(range(len(stratified)), difference, color=colors, alpha=0.85)
    axes[1].axhline(0, color="#111827", linewidth=1.2)
    axes[1].set_xticks(range(len(stratified)), labels=stratified["Age_band_months"])
    axes[1].set_xlabel(T("fig09_age_band"))
    axes[1].set_ylabel(T("fig09_diff"))
    axes[1].set_title(T("fig09_right"))
    # Запас по оси, иначе подпись под отрицательным столбцом наезжает на ось X.
    span = float(np.abs(difference).max())
    axes[1].set_ylim(-span * 1.32, span * 1.28)
    for position, (value, p_holm) in enumerate(zip(difference, stratified["p_holm"])):
        marker = T("p_small") if p_holm < 0.001 else T("p_value", value=num(p_holm))
        axes[1].text(
            position,
            value + (span * 0.05 if value >= 0 else -span * 0.11),
            marker,
            ha="center",
            fontsize=9,
        )

    fig.suptitle(T("fig09_suptitle"))
    fig.tight_layout()
    save_figure(fig, path)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def build_summary(
    cars: pd.DataFrame,
    suspect_cc: pd.DataFrame,
    implausible_km: pd.DataFrame,
    profiles: pd.DataFrame,
    correlations: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    cc_check: pd.DataFrame,
    km_check: pd.DataFrame,
    secondary: pd.DataFrame,
    segments: pd.DataFrame,
    segment_tests: pd.DataFrame,
    stratified: pd.DataFrame,
    diesel_split: pd.DataFrame,
    configurations: pd.DataFrame,
) -> dict:
    profile_records = profiles.set_index("Fuel_Type").to_dict(orient="index")
    correlation_records = correlations.set_index("Variable_2").to_dict(orient="index")
    omnibus_records = omnibus.set_index("Variable").to_dict(orient="index")
    strongest_technical = (
        cars[["HP", "CC", "Weight", "Quarterly_Tax"]]
        .corr(method="spearman")
        .where(lambda frame: ~np.eye(len(frame), dtype=bool))
        .abs()
        .stack()
        .sort_values(ascending=False)
    )
    strongest_pair = strongest_technical.index[0]
    strongest_value = cars[list(strongest_pair)].corr(method="spearman").iloc[0, 1]

    return json_ready(
        {
            "overview": {
                "rows": len(cars),
                "columns_used": len(ANALYSIS_COLUMNS),
                "missing_values": int(cars.isna().sum().sum()),
                "fuel_counts": cars["Fuel_Type"].value_counts().reindex(FUEL_ORDER).to_dict(),
                "suspect_cc_rows": len(suspect_cc),
                "suspect_cc_values": suspect_cc["CC"].tolist(),
                "implausible_km_rows": len(implausible_km),
            },
            "overall_medians": cars[NUMERIC_COLUMNS].median().round(2).to_dict(),
            "fuel_profiles": profile_records,
            "segment_profiles": segments.set_index("Segment").to_dict(orient="index"),
            "segment_tests": segment_tests.set_index("Grouping").to_dict(orient="index"),
            "top_configurations": configurations.head(6).to_dict(orient="records"),
            "top_configurations_coverage_pct": float(configurations.head(6)["Share_pct"].sum()),
            "stratified_price_petrol_vs_diesel": stratified.to_dict(orient="records"),
            "diesel_generations": diesel_split.to_dict(orient="records"),
            "price_correlations": correlation_records,
            "kruskal_tests": omnibus_records,
            "significant_pairwise_tests": pairwise[pairwise.get("Significant_0_05", False)].to_dict(orient="records") if not pairwise.empty else [],
            "cc_sensitivity": cc_check.to_dict(orient="records"),
            "km_sensitivity": km_check.to_dict(orient="records"),
            "secondary_factors": secondary.to_dict(orient="records"),
            "strongest_technical_correlation": {
                "variable_1": strongest_pair[0],
                "variable_2": strongest_pair[1],
                "rho": strongest_value,
            },
        }
    )


def run_analysis(input_path: Path, output_dir: Path, lang: str = "ru") -> dict:
    global LANG
    if lang not in TEXTS:
        raise ValueError(f"Неизвестный язык подписей: {lang}")
    LANG = lang
    configure_plotting()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cars, quality, suspect_cc, implausible_km = load_and_validate(input_path)
    descriptive = descriptive_statistics(cars)
    profiles = fuel_profiles(cars)
    correlations, correlation_matrix = spearman_results(cars)
    omnibus, pairwise = kruskal_results(cars)
    cc_check = cc_sensitivity(cars)
    km_check = km_sensitivity(cars, implausible_km)
    secondary = secondary_factors(cars)
    crosstab, configurations = configuration_crosstab(cars)
    segments, segment_tests = segment_profiles(cars)
    stratified = stratified_price(cars)
    diesel_split = diesel_generations(cars)

    quality.to_csv(tables_dir / "data_quality.csv", index=False)
    descriptive.to_csv(tables_dir / "descriptive_statistics.csv", index=False)
    profiles.to_csv(tables_dir / "fuel_profiles.csv", index=False)
    correlations.to_csv(tables_dir / "spearman_price_correlations.csv", index=False)
    correlation_matrix.round(4).to_csv(tables_dir / "spearman_correlation_matrix.csv")
    omnibus.to_csv(tables_dir / "kruskal_wallis_by_fuel.csv", index=False)
    pairwise.to_csv(tables_dir / "pairwise_mannwhitney.csv", index=False)
    cc_check.to_csv(tables_dir / "cc_sensitivity.csv", index=False)
    km_check.to_csv(tables_dir / "km_sensitivity.csv", index=False)
    secondary.to_csv(tables_dir / "secondary_factors.csv", index=False)
    suspect_cc.to_csv(tables_dir / "suspect_cc_rows.csv", index=False)
    implausible_km.to_csv(tables_dir / "implausible_km_rows.csv", index=False)
    crosstab.to_csv(tables_dir / "configuration_crosstab_cc_hp.csv")
    configurations.to_csv(tables_dir / "technical_configurations.csv", index=False)
    segments.to_csv(tables_dir / "segment_profiles.csv", index=False)
    segment_tests.to_csv(tables_dir / "segment_vs_fuel_price_tests.csv", index=False)
    stratified.to_csv(tables_dir / "stratified_price_by_age.csv", index=False)
    diesel_split.to_csv(tables_dir / "diesel_generations.csv", index=False)

    plot_distributions(cars, figures_dir / "01_distributions.png")
    plot_price_age(cars, figures_dir / "02_price_age_by_fuel.png")
    plot_price_km(cars, figures_dir / "03_price_km_by_fuel.png")
    plot_technical(cars, figures_dir / "04_technical_profiles.png")
    plot_fuel_boxplots(cars, figures_dir / "05_fuel_boxplots.png")
    standardized = plot_standardized_profiles(cars, figures_dir / "06_standardized_fuel_profiles.png")
    standardized.round(4).to_csv(tables_dir / "standardized_fuel_profiles.csv")
    plot_configuration_heatmap(cars, figures_dir / "07_configuration_heatmap.png")
    plot_price_by_segment(cars, figures_dir / "08_price_by_segment.png")
    plot_stratified_price(cars, figures_dir / "09_stratified_price_by_age.png")

    summary = build_summary(
        cars,
        suspect_cc,
        implausible_km,
        profiles,
        correlations,
        omnibus,
        pairwise,
        cc_check,
        km_check,
        secondary,
        segments,
        segment_tests,
        stratified,
        diesel_split,
        configurations,
    )
    with (output_dir / "results_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print("EDA выполнен успешно")
    print(f"Наблюдений: {len(cars)}")
    print(f"Графики: {figures_dir.resolve()}")
    print(f"Таблицы: {tables_dir.resolve()}")
    return summary


def main() -> None:
    args = parse_args()
    run_analysis(args.input, args.output, args.lang)


if __name__ == "__main__":
    main()
