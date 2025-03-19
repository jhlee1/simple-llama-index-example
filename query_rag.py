from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

class RAGQueryEngine:
    def __init__(self, storage_dir='index_storage', model_name="deepseek-r1"):
        """
        Initialize the RAG query engine
        """
        # Set up LLM and embeddings
        self.llm = Ollama(model=model_name, temperature=0.1)
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Load the index
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        self.index = load_index_from_storage(storage_context)
        
        # Create query engine
        self.query_engine = self.index.as_query_engine()
    
    def query(self, question: str) -> str:
        """
        Query the RAG system
        """
        response = self.query_engine.query(question)
        return str(response)

def main():
    # Example usage
    rag_engine = RAGQueryEngine()
    
    # Interactive query loop
    print("RAG Query System (type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
            
        try:
            response = rag_engine.query(question)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
