import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. SETUP THE LOCAL BRAIN
# This replaces OpenAI with your local Llama 3
Settings.llm = Ollama(model="llama3", request_timeout=120.0)
# This handles the math for your local "map" (Vector Index)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Use the full path to avoid any confusion between folders
TOOLKIT_DIR = r"A:\Personal Files\Career Folder\Data_Science\data-science-toolbox"
STORAGE_DIR = "./storage_local"

def get_query_engine():
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

    if os.path.exists(os.path.join(STORAGE_DIR, "docstore.json")):
        print("--- Loading Local Knowledge Index ---")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        print("--- Building Local Index (This stays on your A: Drive) ---")
        reader = SimpleDirectoryReader(
            input_dir=TOOLKIT_DIR,
            recursive=True,
            required_exts=[".md", ".py"],
            exclude=["*.ipynb", "*.csv", "*.zip", "ARCHIVE/*"]
        )
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        print(f"--- Index built with {len(documents)} documents ---")
        
    return index.as_query_engine(streaming=True)

if __name__ == "__main__":
    engine = get_query_engine()
    response = engine.query("What are my standard imports in 01_data_ingestion?")
    response.print_response_stream()