"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.path)

    missing_values = df.isna().sum().to_dict()

    passengers_over_30 = int((df["Age"] > 30).sum())

    mean_age_by_pclass = {
        int(pclass): float(mean_age)
        for pclass, mean_age in df.groupby("Pclass")["Age"].mean().items()
    }

    survival_rate_by_pclass = {
        int(pclass): float(rate)
        for pclass, rate in df.groupby("Pclass")["Survived"].mean().items()
    }

    top_fares = df["Fare"].sort_values(ascending=False).head(5).tolist()

    return TitanicSummary(
        missing_values=missing_values,
        passengers_over_30=passengers_over_30,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        top_fares=top_fares,
    )