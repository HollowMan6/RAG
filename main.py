#!/usr/bin/env python3

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.utils import embed_nodes

import logging
from timeit import default_timer as timer

import parsing
import components
import dbretriever

logging.basicConfig(level=logging.INFO)


def construct_index(md_dir_path, html_dir_path):

    text_parser = SentenceSplitter(chunk_size=512, chunk_overlap=40)

    nodes = parsing.parse_files_into_nodes(
        text_parser, md_dir_path, lambda x: open(x).read()
    )
    nodes.extend(
        parsing.parse_files_into_nodes(
            text_parser, html_dir_path, parsing.parse_html_files, ".html"
        )
    )

    start = timer()
    id_to_embed_map = embed_nodes(nodes, components.embed_model, show_progress=True)

    results = []
    for node in nodes:
        embedding = id_to_embed_map[node.node_id]
        result = node.copy()
        result.embedding = embedding
        results.append(result)
    end = timer()
    logging.info(f"Embedding generation time: {end - start}s")

    start = timer()
    components.vector_store.add(results)
    end = timer()
    logging.info(f"Vector store add time: {end - start}s")

    index = VectorStoreIndex.from_vector_store(
        components.vector_store, embed_model=components.embed_model
    )

    return index


def generate_response(retrieved_nodes, query_str, qa_prompt, llm):
    context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
    fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
    response_iter = llm.stream_complete(fmt_qa_prompt)
    for response in response_iter:
        print(response.delta, end="", flush=True)
    return fmt_qa_prompt


if __name__ == "__main__":
    index = construct_index("docs-data", "html-data")

    logging.info("Index generated successfully!")

    print("Hi, I'm DocsGPT!")
    print("Q&A mode")
    print("Exit with Ctrl-C")

    try:
        while True:
            question = input("You:\n")
            print("\nAI:\n", end="")

            qa_prompt = PromptTemplate(
                """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \
"""
            )
            retriever = dbretriever.VectorDBRetriever(
                components.vector_store, components.embed_model
            )
            retrieved_nodes = retriever.retrieve(question)

            print("\033[1;32;40m", end="")
            fmt_qa_prompt = generate_response(
                retrieved_nodes, question, qa_prompt, components.llm
            )

            print("\033[0m")

            # logging.debug(f"*****Formatted Prompt*****:\n{fmt_qa_prompt}\n\n")

    except KeyboardInterrupt:
        print("\nBye!")
