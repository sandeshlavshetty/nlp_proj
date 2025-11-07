from task_builder import LLMtool
from json_chunker import JSONQuestionChunker
from vectorstore import FaissVectorStore

# # Step 1: Extract JSON
llm_tool = LLMtool()
print("just started")
pdf_text = open("C:\\Users\\sandesh lavshetty\\OneDrive\\Desktop\\clg\\iiitn_1st_yr\\5th_sem\\NLP\\nlp_proj\\src\\Rag_service\\data\\CSL302S124-25.pdf", "rb")  # or OCR
extracted_text = llm_tool.ocr_pdf_with_genai(pdf_text)
tagged = llm_tool.tagger(extracted_text)

# # Step 2: Convert to question chunks
chunker = JSONQuestionChunker()
question_chunks = chunker.json_to_chunks(tagged)
embeddings = chunker.embed_chunks(question_chunks)

# # Step 3: Build hybrid vector store
# store = FaissVectorStore("faiss_store")
# store.build_from_question_chunks(question_chunks)

# pdf_text = open("C:\\Users\\sandesh lavshetty\\OneDrive\\Desktop\\clg\\iiitn_1st_yr\\5th_sem\\NLP\\nlp_proj\\src\\Rag_service\\data\\CSL302S222-23.pdf", "rb")  # or OCR
# extracted_text = llm_tool.ocr_pdf_with_genai(pdf_text)
# tagged = llm_tool.tagger(extracted_text)

# # Step 2: Convert to question chunks
# chunker = JSONQuestionChunker()
# question_chunks = chunker.json_to_chunks(tagged)
# embeddings = chunker.embed_chunks(question_chunks)

# Step 3: Build hybrid vector store
store = FaissVectorStore("faiss_store")
if store.index is None:
    store.load()
store.build_from_question_chunks(question_chunks)


# Step 4: Query hybrid search
results = store.hybrid_query("Explain link Maximum Transmission Unit", top_k=3)
print(results)
# for r in results:
#     print(r["metadata"].get("question_id"), "→", r.get("text", r["metadata"].get("text", "[No text found]")))

