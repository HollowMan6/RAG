from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.vector_stores import (
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
)
from llama_index.core.schema import BaseNode
from pylibraft.common import DeviceResources
from pylibraft.neighbors import cagra, hnsw, ivf_flat, ivf_pq
from pylibraft.neighbors.brute_force import knn

import numpy as np
import cupy as cp

import jax.numpy as jnp
from jax import jit, vmap

from typing import Tuple, List, Any, Dict, cast
from timeit import default_timer as timer
import logging

gpu_device_handle = DeviceResources()


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
    qembed_cp = cp.array(query_embedding, dtype=cp.float32)
    # dimensions: N x D
    dembed_cp = cp.array(doc_embeddings, dtype=cp.float32)
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


def get_top_k_jit(
    qembed_jax: jnp.ndarray,
    dembed_jax: jnp.ndarray,
):
    """Get top nodes by similarity to the query."""

    # Define a function for computing cosine similarity
    @vmap
    def cosine_similarity(dembed):
        # Compute dot product and norms
        # dimensions: N
        dproduct = jnp.dot(dembed, qembed_jax)
        # dimensions: N
        norm = jnp.linalg.norm(qembed_jax) * jnp.linalg.norm(dembed)
        return dproduct / norm

    # Vectorize the cosine_similarity function to handle multiple document embeddings
    # Compute cosine similarities for all document embeddings
    cos_sim_arr = cosine_similarity(dembed_jax)

    # now we have the N cosine similarities for each document
    # sort by cosine similarity
    sorted_indices = jnp.argsort(cos_sim_arr)[::-1]

    return sorted_indices, cos_sim_arr


def get_top_k_embeddings_accelerated(
    query_embedding: List[float],
    doc_embeddings: List[List[float]],
    doc_ids: List[str],
    similarity_top_k: int = 5,
) -> Tuple[List[float], List[str]]:
    # Compile the function with JIT for performance optimization
    # Convert input lists to JAX arrays
    # dimensions: D
    qembed_jax = jnp.array(query_embedding)
    # dimensions: N x D
    dembed_jax = jnp.array(doc_embeddings)

    start = timer()
    sorted_indices, cos_sim_arr = jit(get_top_k_jit, backend="cpu")(
        qembed_jax, dembed_jax
    )
    end = timer()
    print(f"Execution time for jax accelerated one on CPU is {end - start} seconds")
    start = timer()
    sorted_indices, cos_sim_arr = jit(get_top_k_jit, backend="gpu")(
        qembed_jax, dembed_jax
    )
    end = timer()
    print(f"Execution time for jax accelerated one on GPU is {end - start} seconds")
    top_k_indices = sorted_indices[:similarity_top_k]
    result_ids = [doc_ids[i] for i in top_k_indices.tolist()]

    return cos_sim_arr[top_k_indices].tolist(), result_ids


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
    result = get_top_k_embeddings(
        query_embedding,
        doc_embeddings,
        doc_ids,
        similarity_top_k=query.similarity_top_k,
    )
    end = timer()
    print(f"get_top_k_embeddings, CPU: {end - start}s")

    get_top_k_embeddings_accelerated(
        query_embedding,
        doc_embeddings,
        doc_ids,
        similarity_top_k=query.similarity_top_k,
    )

    start = timer()
    get_top_k_embeddings_gpu(
        query_embedding,
        doc_embeddings,
        doc_ids,
        similarity_top_k=query.similarity_top_k,
    )
    end = timer()
    print(f"get_top_k_embeddings, GPU: {end - start}s")
    start = timer()
    distances, neighbors = knn(
        cp.array(doc_embeddings, dtype=cp.float32),
        cp.array([query_embedding], dtype=cp.float32),
        query.similarity_top_k,
    )
    result_similarities = [1 - n for n in cp.asarray(distances)[0].tolist()]
    result_ids = [doc_ids[id] for id in cp.asarray(neighbors)[0].tolist()]
    # result = (result_similarities, result_ids)
    end = timer()
    print(f"Brute Force, knn, GPU: {end - start}s")
    return result


def ann_search(query_embedding, similarity_top_k, index, params, node_ids, func, name):
    """ANN search."""
    start = timer()

    distances, neighbors = func(
        params,
        index,
        query_embedding,
        similarity_top_k,
        handle=gpu_device_handle,
    )
    # pylibraft functions are often asynchronous so the
    # handle needs to be explicitly synchronized
    gpu_device_handle.sync()
    result_similarities = [1 - n for n in cp.asarray(distances)[0].tolist()]
    result_ids = [node_ids[id] for id in cp.asarray(neighbors)[0].tolist()]
    end = timer()
    print(f"{name}: {end - start}s")
    return result_similarities, result_ids


class MemoryVectorStore(VectorStore):
    """Simple custom Vector Store.

    Stores documents in a simple in-memory dict.

    """

    stores_text: bool = True

    def __init__(self) -> None:
        """Init params."""
        self.node_dict: Dict[str, BaseNode] = {}
        # Maintain a list of node_ids in order for easy access
        self.node_ids: List[str] = []
        self.hnsw_search_params = hnsw.SearchParams(ef=20, num_threads=20)
        # On small batch sizes, using "multi_cta" algorithm is efficient
        self.cagra_index_params = cagra.IndexParams(graph_degree=32)
        self.cagra_search_params = cagra.SearchParams(algo="multi_cta")
        self.ivf_flat_index_params = ivf_flat.IndexParams(
            n_lists=128, metric="sqeuclidean"
        )
        self.ivf_flat_search_params = ivf_flat.SearchParams()
        self.ivf_pq_index_params = ivf_pq.IndexParams(n_lists=128, metric="sqeuclidean")
        self.ivf_pq_search_params = ivf_pq.SearchParams()

    def update_index(self) -> None:
        """Update index."""
        docs_embeddings = [self.node_dict[id].embedding for id in self.node_ids]
        docs_cp = cp.array(docs_embeddings, dtype=cp.float32)
        self.cagra_index = cagra.build(
            self.cagra_index_params,
            docs_cp,
            handle=gpu_device_handle,
        )
        self.hnsw_index = hnsw.from_cagra(self.cagra_index, handle=gpu_device_handle)
        self.ivf_flat_index = ivf_flat.build(
            self.ivf_flat_index_params, docs_cp, handle=gpu_device_handle
        )
        self.ivf_pq_index = ivf_pq.build(
            self.ivf_pq_index_params, docs_cp, handle=gpu_device_handle
        )

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        return self.node_dict[text_id].embedding

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            self.node_dict[node.node_id] = node
            self.node_ids.append(node.node_id)
        self.update_index()

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with node_id.

        Args:
            node_id: str

        """
        self.node_ids.remove(node_id)
        del self.node_dict[node_id]
        self.update_index()

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        nodes = self.node_dict.values()
        # dtype float64 not supported
        query_embedding_gpu = cp.array([query.query_embedding], dtype=cp.float32)
        similarities, node_ids = ann_search(
            query_embedding_gpu,
            query.similarity_top_k,
            self.cagra_index,
            self.cagra_search_params,
            self.node_ids,
            cagra.search,
            "cagra, GPU",
        )
        similarities, node_ids = ann_search(
            query_embedding_gpu,
            query.similarity_top_k,
            self.ivf_flat_index,
            self.ivf_flat_search_params,
            self.node_ids,
            ivf_flat.search,
            "ivf_flat, GPU",
        )
        similarities, node_ids = ann_search(
            query_embedding_gpu,
            query.similarity_top_k,
            self.ivf_pq_index,
            self.ivf_pq_search_params,
            self.node_ids,
            ivf_pq.search,
            "ivf_pq, GPU",
        )
        similarities, node_ids = ann_search(
            np.array([query.query_embedding], dtype=np.float32),
            query.similarity_top_k,
            self.hnsw_index,
            self.hnsw_search_params,
            self.node_ids,
            hnsw.search,
            "hnsw, CPU",
        )
        # 1. First filter by metadata
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
