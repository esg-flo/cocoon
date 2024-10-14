import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from core.llm import LLM
from core.prompts import entity_relation_matches_prompt, relation_statisfy_description_prompt
from core.search.faiss_search import FaissEmbeddingSearchManager
from utils.logging import logger
from utils.utils import extract_json_code_safe, parse_json_col


def find_similar_indices(df, index, embed_col: str = "embedding", top_k: int = 10):
    if not isinstance(df[embed_col].iloc[0], list):
        df = parse_json_col(df, col=embed_col)

    embeddings = np.array(df[embed_col].tolist(), dtype=np.float32)
    distances, indexes = index.search(embeddings, top_k)
    return distances, indexes


def _entity_relation_match_one(input_desc: str, refernece_desc: str, llm: LLM):
    prompt = entity_relation_matches_prompt.prompt.format(input_desc, refernece_desc)

    response, raw_output, metadata = llm.invoke(prompt)

    json_code = extract_json_code_safe(response)
    json_var = json.loads(json_code)
    return json_var


def _find_relation_satisfy_description(entity_desc: str, related_rows_desc_str: str, llm: LLM):
    prompt = relation_statisfy_description_prompt.prompt.format(related_rows_desc_str, entity_desc)

    response, raw_output, metadata = llm.invoke(prompt)

    json_code = extract_json_code_safe(response)
    json_var = json.loads(json_code)
    return json_var


def _replace_indices_with_entities(json_var, reference_entities):
    for category in json_var:
        if isinstance(json_var[category], dict) and "entity" in json_var[category]:
            json_var[category]["entity"] = [
                reference_entities[int(idx) - 1] for idx in json_var[category]["entity"]
            ]
    return json_var


def find_entity_relation_matches_and_cluster(
    input_df: pd.DataFrame,
    similar_indexes: np.array,
    refernece_df: pd.DataFrame,
    llm: LLM,
    columns_to_use: List[str] = None,
    label: str = "label",
    match_col: str = "matches",
    verbose: bool = False,
    **kwargs: Optional[Dict],
):
    _search_kwargs = {}
    if "embed_col" in kwargs:
        _search_kwargs["embed_col"] = kwargs["embed_col"]
    faiss_embed_search_obj = FaissEmbeddingSearchManager(input_df, **_search_kwargs)

    if match_col not in input_df:
        input_df[match_col] = None

    # Remove rows for which matches already found
    for idx in range(input_df.shape[0]):
        if input_df[match_col].iloc[idx] is not None:
            faiss_embed_search_obj.remove_rows(idx)

    if columns_to_use is None:
        columns_to_use = input_df.columns.tolist()
        columns_to_use.remove("label")
        columns_to_use.remove("index_ids")
        columns_to_use.remove("embedding")

    while not faiss_embed_search_obj.is_index_empty():
        idx = faiss_embed_search_obj.get_valid_id()
        faiss_embed_search_obj.remove_rows(idx)

        input_desc = ""
        for attribute in columns_to_use:
            input_desc += attribute + ": " + input_df.iloc[idx][attribute] + "\n"

        reference_entities = list(refernece_df[label].iloc[similar_indexes[idx]])
        reference_desc = ""
        for i, output in enumerate(refernece_df[label].iloc[similar_indexes[idx]]):
            reference_desc += str(i + 1) + ". " + output + "\n"

        if verbose:
            logger.info(f"👉 Input: {input_desc}")
            logger.info(f"👉 Reference: {reference_desc}")

        json_var = _entity_relation_match_one(input_desc, reference_desc, llm)
        json_var = _replace_indices_with_entities(json_var, reference_entities)

        if verbose:
            logger.info(f"👉 Match: {json.dumps(json_var, indent=4)}")

        all_indicies = []
        related_rows = faiss_embed_search_obj.search_by_row_index(idx, k=30)

        if len(related_rows) > 0:
            entity_desc = json_var["Summary of Relations"]
            reference_desc = related_rows[columns_to_use].reset_index(drop=True).to_csv(quoting=1)

            json_var2 = _find_relation_satisfy_description(
                entity_desc=entity_desc, related_rows_desc_str=reference_desc, llm=llm
            )

            indicies = json_var2["indices"]

            ids = []
            for index in indicies:
                ids.append(list(related_rows.index)[index])

            all_indicies += ids
            faiss_embed_search_obj.remove_rows(ids)

            json_var3 = {"similar_to": idx}
            for i, index in enumerate(all_indicies):
                input_df.at[index, match_col] = json_var3

        input_df.at[idx, match_col] = json_var
