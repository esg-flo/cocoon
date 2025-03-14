import ast
from typing import Dict, List

import pandas as pd


def _df_row_to_dict(df: pd.DataFrame, idx: int, exclude_columns: List[str] = None):
    if exclude_columns is None:
        exclude_columns = []

    df_reduced = df.drop(columns=exclude_columns, errors="ignore")

    if idx not in df_reduced.index:
        raise IndexError(f"Index {idx} is out of bounds for the DataFrame.")

    record = df_reduced.iloc[idx].to_dict()
    for key, value in record.items():
        if not isinstance(value, str):
            record[key] = None

    return record


def _generate_cluster_record(
    df: pd.DataFrame,
    cluster_key: int,
    cluster_value: List[int],
    exclude_columns: List[str],
    match_col: str,
):
    output = {
        "input_data": {"entity": list(), "ai_input_description": ""},
        "exact_match": {"entity": list(), "reason": ""},
        "related": {
            "conflicted_assumption": {"entity": list(), "reason": ""},
            "additional_assumption": {"entity": list(), "reason": ""},
        },
        "general": {"entity": list(), "reason": ""},
        "similar_to": list(),
    }

    if cluster_value is None:
        cluster_value = []

    output["input_data"]["entity"] = _df_row_to_dict(
        df, idx=int(cluster_key), exclude_columns=exclude_columns
    )

    similar_data = list()
    for value in cluster_value:
        record = _df_row_to_dict(df, idx=value, exclude_columns=exclude_columns)
        similar_data.append(record)
    output["similar_to"] = similar_data

    matches = df[match_col][int(cluster_key)]
    if not isinstance(matches, dict):
        matches = ast.literal_eval(matches)

    for key, value in matches.items():
        key = key.lower()
        if key == "input entity guess":
            output["input_data"]["ai_input_description"] = value
        elif key == "exact_match":
            output[key] = value
        elif key in ["conflicted_assumption", "additional_assumption"]:
            output["related"][key] = value
        elif key == "general":
            output[key] = value

    return output


def generate_output(
    df: pd.DataFrame,
    exclude_columns: List[str] = ["label", "index_ids", "embedding", "matches"],
    match_col: str = "matches",
):
    results = list()

    for idx, value in enumerate(df["matches"]):
        similar_to = value.get("Similar_to") or []

        record = _generate_cluster_record(
            df=df,
            cluster_key=idx,
            cluster_value=similar_to,
            exclude_columns=exclude_columns,
            match_col=match_col,
        )
        results.append(record)

    return results
