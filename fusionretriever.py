from typing import List

from llama_index.core import QueryBundle, PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank

from tqdm.asyncio import tqdm
import asyncio

import logging
from timeit import default_timer as timer


def generate_queries(llm, query_str: str, num_queries: int = 4):
    query_gen_prompt_str = (
        "You are a helpful assistant that generates multiple search queries based on a "
        "single input query. Generate {num_queries} search queries, one on each line, "
        "related to the following input query:\n"
        "Query: {query}\n"
        "Queries:\n"
    )
    query_gen_prompt = PromptTemplate(query_gen_prompt_str)
    fmt_prompt = query_gen_prompt.format(num_queries=num_queries - 1, query=query_str)
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    return queries


async def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks)

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict


# def run_queries(queries, retrievers):
#     """Run queries against retrievers."""
#     results_dict = {}
#     for query in queries:
#         for i, retriever in enumerate(retrievers):
#             results_dict[(query, i)] = retriever.retrieve(query)

#     return results_dict


def fuse_results(results_dict, similarity_top_k: int = 5):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]


class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 5,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._llm = llm
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        start = timer()
        queries = generate_queries(self._llm, query_bundle.query_str, num_queries=4)
        end = timer()
        print(f"Queries generation time: {end - start}s")

        logging.info(f"Generated queries: {queries}")

        start = timer()
        results = asyncio.run(run_queries(queries, self._retrievers))
        end = timer()
        print(f"retrieve time for all the queries: {end - start}s")

        start = timer()
        final_results = fuse_results(results, similarity_top_k=self._similarity_top_k)
        end = timer()
        print(f"Result fuse time: {end - start}s")

        start = timer()
        # configure reranker
        reranker = RankGPTRerank(
            llm=self._llm,
            top_n=2,
            verbose=True,
        )
        final_results = reranker.postprocess_nodes(final_results, query_bundle)
        end = timer()
        print(f"Reranking time: {end - start}s")

        return final_results
