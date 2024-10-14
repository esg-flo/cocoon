from typing import List, Optional, Union

import faiss
import numpy as np
import pandas as pd


class FaissEmbeddingSearchManager:
    def __init__(self, df: pd.DataFrame, embed_col: str = "embedding"):
        self.df = df
        self.embed_col = embed_col
        self.index = None
        self._initialize_index()

    def _initialize_index(self) -> None:
        embeddings = np.vstack(self.df[self.embed_col].values)
        dims = embeddings.shape[1]

        base_index = faiss.IndexFlatL2(dims)
        self.index = faiss.IndexIDMap(base_index)

        ids = np.array(range(len(self.df))).astype(np.int64)
        self.index.add_with_ids(embeddings, ids)

    def remove_rows(self, row_indices: Union[int, List[int]]) -> None:
        if np.isscalar(row_indices):
            row_indices = [row_indices]

        ids_to_remove = np.array(row_indices, dtype=np.int64)
        self.index.remove_ids(ids_to_remove)

    def get_closest_indices(self, query_embedding: List[float], k: Optional[int] = 5) -> List[int]:
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        if query_embedding.shape[1] != self.index.d:
            raise ValueError("Dimension of query_embedding does not match the index dimension.")

        distances, indices = self.index.search(query_embedding, k)
        valid_indices = indices[0][indices[0] != -1]
        return valid_indices

    def get_closest_rows(self, query_embedding: List[float], k: Optional[int] = 5):
        closest_indices = self.get_closest_indices(query_embedding, k)
        return self.df.iloc[closest_indices]

    def search_by_row_index(self, row_idx: int, k: Optional[int] = 10):
        if row_idx < 0 or row_idx >= len(self.df):
            raise IndexError("Row index is out of bounds.")

        query_embedding = self.df.iloc[row_idx][self.embed_col]
        search_results = self.get_closest_rows(query_embedding, k + 1)

        if row_idx in search_results.index:
            filtered_results = search_results.drop(index=row_idx)
        else:
            filtered_results = search_results

        return filtered_results.head(k)

    def is_index_empty(self) -> bool:
        return self.index.ntotal == 0

    def get_valid_id(self) -> int:
        if self.is_index_empty():
            raise ValueError("The index is empty. No valid ID can be returned.")

        ids = faiss.vector_to_array(self.index.id_map)
        return ids[0]

    def get_size(self) -> int:
        return self.index.ntotal
