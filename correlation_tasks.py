"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.path, sep="\t", na_values="NA")

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    female_df = df[df["Gender"] == "Female"]
    male_df = df[df["Gender"] == "Male"]

    female_correlations: dict[str, float] = {}
    male_correlations: dict[str, float] = {}

    for feat in features:
        female_correlations[feat] = float(female_df[feat].corr(female_df["MRI_Count"], method="pearson"))
        male_correlations[feat] = float(male_df[feat].corr(male_df["MRI_Count"], method="pearson"))

    strongest_mri_feature = None
    max_abs_corr = -1.0

    for feat in features:
        f_abs = abs(female_correlations[feat])
        m_abs = abs(male_correlations[feat])

        if f_abs > max_abs_corr:
            max_abs_corr = f_abs
            strongest_mri_feature = feat

        if m_abs > max_abs_corr:
            max_abs_corr = m_abs
            strongest_mri_feature = feat

    return BrainCorrelationSummary(
        female_correlations=female_correlations,
        male_correlations=male_correlations,
        strongest_mri_feature=strongest_mri_feature,
    )