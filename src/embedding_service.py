import faiss
import numpy as np
import pandas as pd
from core.embeddings.base import Embeddings
from utils.logging import logger


def _initialize_embedding_df(
    df: pd.DataFrame, output_csv_filepath: str, target_col: str, embed_col: str
):
    try:
        embedding_df = pd.read_csv(output_csv_filepath)
    except FileNotFoundError:
        df = df.dropna(subset=[target_col])
        group_df = df.groupby(target_col)
        embedding_df = (
            group_df[df.columns].apply(lambda x: x.index.tolist()).reset_index(name="index_ids")
        )
        embedding_df[embed_col] = None
    return embedding_df


def _find_first_nan_index(df: pd.DataFrame, embed_col: str = "embedding"):
    nan_index = df[embed_col].isna().idxmax()
    value = df.loc[nan_index, embed_col]

    if not isinstance(value, list) and pd.isna(value):
        return nan_index
    return None


def create_embeddings(
    embed_model: Embeddings,
    df: pd.DataFrame,
    output_csv_filepath: str,
    chunk_size: int = 1000,
    target_col: str = "label",
    embed_col: str = "embedding",
):
    if target_col not in df.columns:
        raise ValueError(f"Column with name '{target_col}' should be present in the dataframe.")

    logger.info(f"'{embed_model.model_id}' used for Embedding Text")

    embedding_df = _initialize_embedding_df(df, output_csv_filepath, target_col, embed_col)
    start_index = _find_first_nan_index(embedding_df)

    if start_index and start_index != 0:
        logger.info(
            f"{start_index} rows already embedded. Resume embedding from {start_index + 1}..."
        )
    elif start_index is None:
        logger.info("All labels already embedded.")
        return embedding_df

    for chunk_start in range(start_index, embedding_df.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(embedding_df))
        labels_chunk = embedding_df.loc[chunk_start:chunk_end, target_col].tolist()

        for idx, label in enumerate(labels_chunk):
            embeddings = embed_model.embed_query(label)
            embedding_df.at[chunk_start + idx, embed_col] = embeddings

        embedding_df.to_csv(output_csv_filepath, index=False)

    logger.info("All labels embedded and CSV updated.")

    return embedding_df


def initialize_faiss_index_from_embeddings(
    embed_model: Embeddings, df: pd.DataFrame, embed_col="embedding"
):
    embeddings_array = np.array(df[embed_col].tolist(), dtype=np.float32)
    index = faiss.IndexFlatL2(embed_model.dims)
    index.add(embeddings_array)

    return index
