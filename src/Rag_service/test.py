from vectorstore import FaissVectorStore
from search import RAGSearch
from data_loader import load_all_documents
from task_builder import LLMtool

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

def test_llmtool():
    llm_tool = LLMtool(llm_model="gemini-2.5-flash")
    sample_text = llm_tool.describe_image("C:\\Users\\sandesh lavshetty\\OneDrive\\Desktop\\clg\\iiitn_1st_yr\\5th_sem\\NLP\\nlp_proj\\src\\Rag_service\\data\\test1.png")
    print("LLMtool Describe Image Result:", sample_text)
    result = llm_tool.tagger(sample_text)
    print("LLMtool Tagger Result:", result)


if __name__ == "__main__":
    # ingest()
    # query_only()
    res = test_llmtool()
    print("Test LLMtool Result:", res)