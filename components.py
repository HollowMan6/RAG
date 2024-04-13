from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP

import vectorstore

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en", device="cuda")

llm = LlamaCPP(
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="llama-2-13b-chat.Q4_0.gguf",
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

vector_store = vectorstore.MemoryVectorStore()
