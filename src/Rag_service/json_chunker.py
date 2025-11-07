# json_chunker.py
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

class JSONQuestionChunker:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)

    def json_to_chunks(self, tagged_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert parsed exam paper JSON (from tagger) into structured question chunks.
        Each question = 1 chunk.
        """
        chunks = []
        common_meta = {
            "date_of_exam": tagged_json.get("date_of_exam"),
            "type_of_exam": tagged_json.get("type_of_exam"),
            "paper_code": tagged_json.get("paper_code"),
            "subject_name": tagged_json.get("subject_name"),
        }
        questions = tagged_json.get("questions", {})
        for q_key, q_text in questions.items():
            chunk = {
                "id": q_key,
                "text": q_text,
                "title": f"{common_meta.get('paper_code') or ''} {q_key}",
                "metadata": {**common_meta, "question_id": q_key, "question_text": q_text}
            }
            chunks.append(chunk)
        print(f"[INFO] Created {len(chunks)} question chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        texts = [c["text"] for c in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} question chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
