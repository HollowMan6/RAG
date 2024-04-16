from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.vector_stores import (
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
)
from llama_index.core.schema import BaseNode

import numpy as np
import cupy as cp
# import jax.numpy as jnp
# from jax import jit, vmap

from typing import Tuple, List, Any, Dict, cast
from timeit import default_timer as timer
import logging


def get_top_k_embeddings(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    doc_ids: List[str],
    similarity_top_k: int = 5,
) -> Tuple[List[float], List]:
    """Get top nodes by similarity to the query."""
    # dimensions: D
    qembed_np = np.array(query_embedding)
    # dimensions: N x D
    dembed_np = np.array(doc_embeddings)
    # dimensions: N
    dproduct_arr = np.dot(dembed_np, qembed_np)
    # dimensions: N
    norm_arr = np.linalg.norm(qembed_np) * np.linalg.norm(
        dembed_np, axis=1, keepdims=False
    )
    # dimensions: N
    cos_sim_arr = dproduct_arr / norm_arr

    # now we have the N cosine similarities for each document
    # sort by top k cosine similarity, and return ids
    tups = [(cos_sim_arr[i], doc_ids[i]) for i in range(len(doc_ids))]
    sorted_tups = sorted(tups, key=lambda t: t[0], reverse=True)

    sorted_tups = sorted_tups[:similarity_top_k]

    result_similarities = [s for s, _ in sorted_tups]
    result_ids = [n for _, n in sorted_tups]
    return result_similarities, result_ids


def get_top_k_embeddings_gpu(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    doc_ids: List[str],
    similarity_top_k: int = 5,
) -> Tuple[List[float], List]:
    """Get top nodes by similarity to the query."""
    # dimensions: D
    qembed_cp = cp.array(query_embedding)
    # dimensions: N x D
    dembed_cp = cp.array(doc_embeddings)
    # dimensions: N
    dproduct_arr = cp.dot(dembed_cp, qembed_cp)
    # dimensions: N
    norm_arr = cp.linalg.norm(qembed_cp) * cp.linalg.norm(
        dembed_cp, axis=1, keepdims=False
    )
    # dimensions: N
    cos_sim_arr = dproduct_arr / norm_arr

    # now we have the N cosine similarities for each document
    # sort by top k cosine similarity, and return ids
    tups = [(cos_sim_arr[i], doc_ids[i]) for i in range(len(doc_ids))]
    sorted_tups = sorted(tups, key=lambda t: t[0], reverse=True)

    sorted_tups = sorted_tups[:similarity_top_k]

    result_similarities = [s for s, _ in sorted_tups]
    result_ids = [n for _, n in sorted_tups]
    return result_similarities, result_ids


# @jit
# def get_top_k_jit(
#     query_embedding: List[float],
#     doc_embeddings: List[List[float]],
# ):
#     """Get top nodes by similarity to the query."""
#     # Convert input lists to JAX arrays
#     # dimensions: D
#     qembed_jax = jnp.array(query_embedding)
#     # dimensions: N x D
#     dembed_jax = jnp.array(doc_embeddings)

#     # Define a function for computing cosine similarity
#     @vmap
#     def cosine_similarity(dembed):
#         # Compute dot product and norms
#         # dimensions: N
#         dproduct = jnp.dot(dembed, qembed_jax)
#         # dimensions: N
#         norm = jnp.linalg.norm(qembed_jax) * jnp.linalg.norm(dembed)
#         return dproduct / norm

#     # Vectorize the cosine_similarity function to handle multiple document embeddings
#     # Compute cosine similarities for all document embeddings
#     cos_sim_arr = cosine_similarity(dembed_jax)

#     # now we have the N cosine similarities for each document
#     # sort by cosine similarity
#     sorted_indices = jnp.argsort(cos_sim_arr)[::-1]

#     return sorted_indices, cos_sim_arr


# def get_top_k_embeddings_accelerated(
#     query_embedding: List[float],
#     doc_embeddings: List[List[float]],
#     doc_ids: List[str],
#     similarity_top_k: int = 5,
# ) -> Tuple[List[float], List[str]]:
#     # Compile the function with JIT for performance optimization

#     sorted_indices, cos_sim_arr = get_top_k_jit(query_embedding, doc_embeddings)
#     top_k_indices = sorted_indices[:similarity_top_k]
#     result_ids = [doc_ids[i] for i in top_k_indices.tolist()]

#     return cos_sim_arr[top_k_indices].tolist(), result_ids


def filter_nodes(nodes: List[BaseNode], filters: MetadataFilters):
    filtered_nodes = []
    for node in nodes:
        matches = True
        for f in filters.filters:
            if f.key not in node.metadata:
                matches = False
                continue
            if f.value != node.metadata[f.key]:
                matches = False
                continue
        if matches:
            filtered_nodes.append(node)
    return filtered_nodes


def dense_search(query: VectorStoreQuery, nodes: List[BaseNode]):
    """Dense search."""
    query_embedding = cast(List[float], query.query_embedding)
    doc_embeddings = [n.embedding for n in nodes]
    doc_ids = [n.node_id for n in nodes]
    start = timer()
    get_top_k_embeddings(
        query_embedding,
        doc_embeddings,
        doc_ids,
        similarity_top_k=query.similarity_top_k,
    )
    end = timer()
    print(f"get_top_k_embeddings, CPU: {end - start}s")
    # start = timer()
    # result = get_top_k_embeddings_accelerated(
    #     query_embedding,
    #     doc_embeddings,
    #     doc_ids,
    #     similarity_top_k=query.similarity_top_k,
    # )
    # end = timer()
    # print(f"Execution time for jax accelerated one is {end - start} seconds")
    start = timer()
    result = get_top_k_embeddings_gpu(
        query_embedding,
        doc_embeddings,
        doc_ids,
        similarity_top_k=query.similarity_top_k,
    )
    end = timer()
    print(f"get_top_k_embeddings, GPU: {end - start}s")
    return result


class MemoryVectorStore(VectorStore):
    """Simple custom Vector Store.

    Stores documents in a simple in-memory dict.

    """

    stores_text: bool = True

    def __init__(self) -> None:
        """Init params."""
        self.node_dict: Dict[str, BaseNode] = {}

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        return self.node_dict[text_id]

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            self.node_dict[node.node_id] = node

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with node_id.

        Args:
            node_id: str

        """
        del self.node_dict[node_id]

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        # 1. First filter by metadata
        nodes = self.node_dict.values()
        if query.filters is not None:
            nodes = filter_nodes(nodes, query.filters)
        if len(nodes) == 0:
            result_nodes = []
            similarities = []
            node_ids = []
        else:
            # 2. Then perform semantic search
            similarities, node_ids = dense_search(query, nodes)
            result_nodes = [self.node_dict[node_id] for node_id in node_ids]
        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )

    def persist(self, persist_path, fs=None) -> None:
        """Persist the SimpleVectorStore to a directory.

        NOTE: we are not implementing this for now.

        """
        pass
