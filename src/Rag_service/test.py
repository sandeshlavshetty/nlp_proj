from vectorstore import FaissVectorStore
from search import RAGSearch
from data_loader import load_all_documents


def ingest():
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    query = "give generator polynomial question"
    results = store.query(query, top_k=3)
    print("Query Results:", results)
    

def query_only():
    rag_search = RAGSearch(llm_model="gemini-2.5-flash")
    query = "give generator polynomial question "
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)

if __name__ == "__main__":
    ingest()
    query_only()