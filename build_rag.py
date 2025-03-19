from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

def build_rag(data_dir='data', storage_dir='index_storage'):
    """
    Build RAG from documents and store it locally
    """
    print("Starting RAG building process...")
    
    # Set up embeddings
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Configure settings
    Settings.embed_model = embed_model
    
    # Load documents
    print(f"Loading documents from {data_dir}...")
    documents = SimpleDirectoryReader(
        data_dir="./data",
        recursive=True,
        exclude_hidden=True,
    ).load_data()
    print(f"Loaded {len(documents)} documents")
    
    # Create and save index
    print("Creating index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # Store the index
    print(f"Saving index to {storage_dir}...")
    index.storage_context.persist(storage_dir)
    print("RAG building completed!")

if __name__ == "__main__":
    build_rag()
