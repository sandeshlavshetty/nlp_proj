# from vectorstore import FaissVectorStore
# from search import RAGSearch
# from data_loader import load_all_documents
from task_builder import LLMtool

# def ingest():
#     docs = load_all_documents("data")
#     store = FaissVectorStore("faiss_store")
#     store.build_from_documents(docs)
#     query = "give generator polynomial question"
#     results = store.query(query, top_k=3)
#     print("Query Results:", results)
    

# def query_only():
#     rag_search = RAGSearch(llm_model="gemini-2.5-flash")
#     query = "give generator polynomial question "
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)

def test_llmtool():
    llm_tool = LLMtool(llm_model="gemini-2.5-flash")
    with open(r"C:\Users\sandesh lavshetty\OneDrive\Desktop\clg\iiitn_1st_yr\5th_sem\NLP\nlp_proj\src\Rag_service\data\CSL302S124-25.pdf", "rb") as f:
        text = llm_tool.ocr_pdf_with_genai(f)  # pass the file-like object
        print("Extracted Text:", text)
    print("LLMtool Describe Image Result:", text)
    result = llm_tool.tagger(text)
    print("LLMtool Tagger Result:", result)

# def just_a_test():
#     llm_tool = LLMtool(llm_model="gemini-2.5-flash")
#     with open(r"C:\Users\sandesh lavshetty\OneDrive\Desktop\clg\iiitn_1st_yr\5th_sem\NLP\nlp_proj\src\Rag_service\data\CSL302S124-25.pdf", "rb") as f:
#         text = llm_tool.ocr_pdf_with_genai(f)  # pass the file-like object
#         print("Extracted Text:", text)
#     return "Just a test function."

if __name__ == "__main__":
    # ingest()
    # query_only()
    res = test_llmtool()
    print("Test LLMtool Result:", res)
    
