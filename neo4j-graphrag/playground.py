#!/usr/bin/env python
# coding: utf-8

import json
import os
from typing import Dict, Any, List, Optional
import asyncio
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
import numpy as np
from llama_index.core.vector_stores import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.types import VectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, PromptTemplate
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
import openai as ai
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.core.schema import TextNode
import pandas as pd
from neo4j import GraphDatabase, Result
import time

MAP_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
        {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
    ]
}}
"""


REDUCE_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""


SUBQ_SYSTEM_PROMPT = """

---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

---Goal---

Given a user question and a list of background summaries that this question may partially fit into, output a list of relevant sub-questions in order of importance, such that the answers to all the sub-questions put together will answer the question. Fine-grain the sub-questions and separate them as much as possible to cover all the background summaries that are enough to answer the original question.

For each sub-question, focus on the key information that would be needed from the corresponding background summary to build the full answer. Ensure that the sub-questions are clear, specific, and cover all relevant aspects of the user's question.

---Input Format---

User Question: {user_question}

Background Summaries: {communities}

---Output Format---

Respond with a list of sub-questions in json format. The structure should be:

{{
    "sub_questions": [
        "<sub_question 1>",
        "<sub_question 2>",
        "<sub_question 3>",
        ...
        "<sub_question n-1>",
        "<sub_question n>"
    ]
}}

Only include the sub-questions in the response, no other words or sentences.

---Example---

User Question: How can I increase user engagement on my app?

Communities: [ {{ "description": "Community focused on UX design strategies.", "name": "UX Design", "fn_schema": {{ "type": "object", "properties": {{ "input": {{"title": "input query string", "type": "string"}} }}, "required": ["input"] }} }}, {{ "description": "Community focused on marketing strategies.", "name": "Marketing", "fn_schema": {{ "type": "object", "properties": {{ "input": {{"title": "input query string", "type": "string"}} }}, "required": ["input"] }} }}, {{ "description": "Community focused on data analytics and user behavior analysis.", "name": "Data Analytics", "fn_schema": {{ "type": "object", "properties": {{ "input": {{"title": "input query string", "type": "string"}} }}, "required": ["input"] }} }}, {{ "description": "Community focused on product management and feature prioritization.", "name": "Product Management", "fn_schema": {{ "type": "object", "properties": {{ "input": {{"title": "input query string", "type": "string"}} }}, "required": ["input"] }} }} ]

Response:

{{
    "sub_questions": [
        "What design changes can improve user interaction and reduce friction?",
        "What marketing strategies can attract more users to the app and keep them engaged?",
        "What does the user data suggest about current engagement patterns and drop-off points?",
        "What new features or updates can be prioritized to enhance user engagement?"
    ]
}}
"""


NEO4J_URI = "bolt://localhost"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "kiwi-mental-isotope-alfonso-telex-4918"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def db_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """Executes a Cypher statement and returns a DataFrame"""
    return driver.execute_query(
        cypher, parameters_=params, result_transformer_=Result.to_df
    )


index_name = "entity"

db_query(
    """
CREATE VECTOR INDEX """
    + index_name
    + """ IF NOT EXISTS FOR (e:__Entity__) ON e.description_embedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 4096,
 `vector.similarity_function`: 'cosine'
}}
"""
)

db_query(
    """
MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:HAS_ENTITY]-(c)
WITH n, count(distinct c) AS chunkCount
SET n.weight = chunkCount
"""
)

content = node_to_metadata_dict(
    TextNode(), remove_text=True, flat_metadata=False)

db_query(
    """
  MATCH (e:__Entity__)
  SET e += $content""",
    {"content": content},
)

ai.api_key = ""
ai.base_url = ""
os.environ["OPENAI_API_KEY"] = ai.api_key
os.environ["OPENAI_API_BASE"] = ai.base_url


topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10


embed_dim = 4096

retrieval_query = f"""
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
nodes,
collect {{
    UNWIND nodes as n
    MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)
    WITH c, count(distinct n) as freq
    RETURN c.text AS chunkText
    ORDER BY freq DESC
    LIMIT {topChunks}
}} AS text_mapping,
// Entity - Report Mapping
collect {{
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH c, c.rank as rank, c.weight AS weight
    RETURN c.summary
    ORDER BY rank, weight DESC
    LIMIT {topCommunities}
}} AS report_mapping,
// Outside Relationships
collect {{
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m)
    WHERE NOT m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC
    LIMIT {topOutsideRels}
}} as outsideRels,
// Inside Relationships
collect {{
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m)
    WHERE m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC
    LIMIT {topInsideRels}
}} as insideRels,
// Entities description
collect {{
    UNWIND nodes as n
    RETURN n.description AS descriptionText
}} as entities
// We don't have covariates or claims here
RETURN "Chunks:" + apoc.text.join(text_mapping, '|') + "\nReports: " + apoc.text.join(report_mapping,'|') +
       "\nRelationships: " + apoc.text.join(outsideRels + insideRels, '|') +
       "\nEntities: " + apoc.text.join(entities, "|") AS text, 1.0 AS score, nodes[0].id AS id,
       {{_node_type:nodes[0]._node_type, _node_content:nodes[0]._node_content}} AS metadata
"""


neo4j_vector = Neo4jVectorStore(
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_URI,
    embed_dim,
    index_name=index_name,
    retrieval_query=retrieval_query,
)
loaded_index = VectorStoreIndex.from_vector_store(neo4j_vector).as_query_engine(
    similarity_top_k=topEntities, llm=OpenAI(model="gpt-4o"), embed_model=OpenAIEmbedding(model="text-embedding-3-large")
)


ai.api_key = ""
ai.base_url = ""
os.environ["OPENAI_API_KEY"] = ai.api_key
os.environ["OPENAI_API_BASE"] = ai.base_url
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = Settings.llm


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
        return self.node_ids

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with node_id.

        Args:
            node_id: str

        """
        self.node_ids.remove(node_id)
        del self.node_dict[node_id]

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        nodes = self.node_dict.values()
        if len(nodes) <= 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        doc_ids = [n.node_id for n in nodes]
        qembed_np = np.array(query.query_embedding)
        # dimensions: N x D
        dembed_np = np.array([n.embedding for n in nodes])
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
        if query.similarity_top_k < len(sorted_tups):
            sorted_tups = sorted_tups[:query.similarity_top_k]
        similarities = [s for s, _ in sorted_tups]
        node_ids = [n for _, n in sorted_tups]
        result_nodes = [self.node_dict[node_id] for node_id in node_ids]
        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )

    def persist(self, persist_path, fs=None) -> None:
        """Persist the SimpleVectorStore to a directory.

        NOTE: we are not implementing this for now.

        """
        pass

    @property
    def client(self) -> Any:
        raise NotImplementedError


memory_store = MemoryVectorStore()


class VectorDBRetriever(BaseRetriever):
    """Retriever over a vector store."""

    def __init__(
        self,
        vector_store,
        similarity_top_k: int = 1,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        vector_store_query = VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


vecdb_retriever = VectorDBRetriever(memory_store, similarity_top_k=1)


async def concurrent_run(tasks, concurrency_limit, timeout=1800):
    sem = asyncio.Semaphore(concurrency_limit)

    async def sem_task(task):
        async with sem:
            try:
                # Apply timeout to each task
                return await asyncio.wait_for(task, timeout)
            except Exception as e:
                print(f"Task timed out after {timeout} seconds, {e}")
                return None

    return await asyncio.gather(*(sem_task(task) for task in tasks))


class QueryEvent(Event):
    questions: List[str]


class AnswerEvent(Event):
    questions: List[str]
    answers: List[List[str]]


class SubQuestionQueryEngine(Workflow):
    @step(pass_context=True)
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        if hasattr(ev, "query"):
            ctx.data["original_query"] = ev.query
            print(f"Query is {ctx.data['original_query']}")

        if hasattr(ev, "llm"):
            ctx.data["llm"] = ev.llm

        if hasattr(ev, "communities"):
            ctx.data["communities"] = ev.communities

        if hasattr(ev, "response_type"):
            ctx.data["response_type"] = ev.response_type

        response = str(ev.llm.complete(
            PromptTemplate(SUBQ_SYSTEM_PROMPT).format(
                user_question=ev.query, communities="\n\n".join(ev.communities))))

        print(f"Sub-questions are {response}")

        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response_obj = json.loads(response)

        sub_questions = response_obj["sub_questions"]

        # # TODO: Limit the number of sub-questions
        # if len(sub_questions) > 5:
        #     sub_questions=sub_questions[:5]

        ctx.data["sub_question_count"] = len(
            sub_questions)  # plus original question

        return QueryEvent(questions=sub_questions)

    async def handle_sub_question(self, llm, question, communities, response_type) -> tuple[str, str, str]:
        print(f"Handling Sub-question {question}")

        # TODO: Select the best llm to answer the question

        local_answer = ""
        global_answer = ""

        question_embedding = Settings.embed_model.get_query_embedding(question)
        node_list = await vecdb_retriever.aretrieve(QueryBundle(query_str=question, embedding=question_embedding))

        if len(node_list) > 0 and node_list[0].score and node_list[0].score > 0.98:
            local_answer = node_list[0].node.metadata["local_answer"]
            global_answer = node_list[0].node.metadata["global_answer"]
        else:
            # local retrieval
            local_response = loaded_index.aquery(question)
            # global retrieval

            tasks = []
            for community in communities:
                community_prompt = PromptTemplate(MAP_SYSTEM_PROMPT).format(
                    question=question, context_data=community)
                task = llm.acomplete(community_prompt)
                tasks.append(task)

            print(f"Sending out {len(tasks)} global retrieval tasks")
            intermediate_global_results = await concurrent_run(tasks, 20)

            global_prompt = PromptTemplate(REDUCE_SYSTEM_PROMPT).format(
                report_data=intermediate_global_results, question=question, response_type=response_type)
            global_response = llm.acomplete(global_prompt)

            global_answer = str(await global_response)
            local_answer = str(await local_response)

            question_node: TextNode = TextNode(
                text=question, embedding=question_embedding)
            question_node.metadata = {
                "local_answer": local_answer, "global_answer": global_answer}

            memory_store.add([question_node])

            print(f"Local answer & Global answer generated for {question}")
        return question, local_answer, global_answer

    @step(pass_context=True)
    async def sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        tasks = []
        for sub_question in [*ev.questions, ctx.data["original_query"]]:
            task = self.handle_sub_question(
                ctx.data["llm"], sub_question, ctx.data["communities"], ctx.data["response_type"])
            tasks.append(task)

        print(f"Sending out {len(ev.questions)} answering tasks")
        sub_questions_answers = await concurrent_run(tasks, 15)

        questions = []
        answers = []
        for sub_question_answer in sub_questions_answers:
            if sub_question_answer:
                q, la, ga = sub_question_answer
                questions.append(q)
                answers.append([la, ga])
        return AnswerEvent(questions=questions, answers=answers)

    @step(pass_context=True)
    async def combine_answers(
        self, ctx: Context, ev: AnswerEvent
    ) -> StopEvent | None:

        answers = "\n\n".join(
            [
                f"Question: {ev.questions[index]} \n Local Answer: {ev.answers[index][0]} \n Global Answer: {ev.answers[index][1]}"
                for index in range(len(ev.questions))
            ]
        )

        prompt = f"""
            You are given an overall question that has been split into sub-questions,
            each of which has been answered, either by global retrival or local retrival.
            Combine the answers to all the sub-questions into a single answer to the
            original question, do not include any irrelevant information, and do not indicate
            that it's a combination of multiple answers in the final response.

            Response type: {ctx.data["response_type"]}

            Original question: {ctx.data['original_query']}

            Sub-questions and answers:
            {answers}
        """

        print(f"Final prompt is {prompt}")

        response = ctx.data["llm"].complete(prompt)

        print("Final response is", response)

        return StopEvent(result=str(response))


communities = db_query(
    """
    MATCH (c:__Community__)
    WHERE c.level = $level
    RETURN c.full_content AS output
    """,
    params={"level": 1},
)

engine = SubQuestionQueryEngine(timeout=36000, verbose=True)

start = time.perf_counter()
result = asyncio.run(engine.run(
    llm=llm,
    communities=communities["output"].to_list(),
    query="What is the article about?",
    response_type="multiple paragraphs"
))
print("Time taken", time.perf_counter() - start)

start = time.perf_counter()
result = asyncio.run(engine.run(
    llm=llm,
    communities=communities["output"].to_list(),
    query="What is the article about?",
    response_type="multiple paragraphs"
))
print("Time taken", time.perf_counter() - start)
