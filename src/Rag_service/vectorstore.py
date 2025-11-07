import os
import faiss
import numpy as np
import pickle
import json
import csv
from typing import List, Any, Dict
from sentence_transformers import SentenceTransformer
from embedding import EmbeddingPipeline
from rank_bm25 import BM25Okapi
import re

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        # metadata: mapping from integer id -> metadata dict
        self.metadata: Dict[int, Dict[str, Any]] = {}
        # internal counter to assign stable IDs
        self._next_id = 0
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")
        
        self.bm25_corpus = []
        self.bm25 = None

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        # Build richer metadata for each chunk including stable doc and chunk ids
        metadatas = []
        for i, chunk in enumerate(chunks):
            # try to extract source/document info if available on the chunk
            source = None
            doc_id = None
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                source = chunk.metadata.get("source") or chunk.metadata.get("source_id")
                doc_id = chunk.metadata.get("doc_id") or chunk.metadata.get("source")

            meta = {
                "text": chunk.page_content,
                "source": source,
                "doc_id": doc_id,
                "chunk_index": i,
            }
            metadatas.append(meta)

        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        # Use IndexFlatL2 wrapped by IndexIDMap to maintain stable integer IDs
        if self.index is None:
            flat = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIDMap(flat)

        n = embeddings.shape[0]
        # assign new ids for these embeddings
        ids = np.arange(self._next_id, self._next_id + n).astype('int64')
        # add to index with explicit ids
        self.index.add_with_ids(embeddings, ids)

        # attach metadata for each id
        if metadatas:
            for offset, meta in enumerate(metadatas):
                assigned_id = int(self._next_id + offset)
                # enrich metadata with the stable id
                meta_with_id = dict(meta)
                meta_with_id["id"] = assigned_id
                self.metadata[assigned_id] = meta_with_id

        self._next_id += n
        print(f"[INFO] Added {n} vectors to Faiss index with ids {self._next_id - n}..{self._next_id - 1}.")

    def build_from_question_chunks(self, question_chunks: List[Dict[str, Any]]):
            """
            Build store directly from question-based chunks (JSON parsed).
            """
            texts = [c["text"] for c in question_chunks]
            embeddings = self.model.encode(texts, show_progress_bar=True)

            self.add_embeddings(np.array(embeddings).astype('float32'), [c["metadata"] for c in question_chunks])

            # Build BM25
            tokenized_corpus = [self._tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_corpus = texts
            print(f"[INFO] Built BM25 index for {len(self.bm25_corpus)} question texts.")
            self.save()

    def _tokenize(self, text):
        return re.findall(r"\w+", text.lower())
    
    def hybrid_query(self, query: str, alpha: float = 0.7, top_k: int = 5):
        """
        alpha = weight for semantic similarity (0_1)
        (1 - alpha) = weight for keyword (BM25)
        """
        # semantic
        query_emb = self.model.encode([query]).astype('float32')
        semantic_results = self.search(query_emb, top_k=top_k * 2)

        # keyword (BM25)
        if self.bm25:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k * 2]
        else:
            bm25_scores = np.zeros(len(self.bm25_corpus))
            bm25_top_idx = []

        # normalize & merge scores
        hybrid_scores = {}
        for r in semantic_results:
            hybrid_scores[r["id"]] = alpha * (1 / (r["distance"] + 1e-8))

        for idx in bm25_top_idx:
            doc_id = list(self.metadata.keys())[idx]
            bm25_score = bm25_scores[idx]
            hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0) + (1 - alpha) * bm25_score

        # final ranking
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = [{"id": doc_id, "score": score, "metadata": self.metadata[doc_id]} for doc_id, score in ranked]
        return results
    

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        # write faiss index
        faiss.write_index(self.index, faiss_path)

        # save metadata dict and next id in pickle for exact restore
        state = {"next_id": self._next_id, "metadata": self.metadata, "bm25_corpus": self.bm25_corpus}
        with open(meta_path, "wb") as f:
            pickle.dump(state, f)

        # also write human-readable summaries (JSON and CSV)
        json_path = os.path.join(self.persist_dir, "metadata_summary.json")
        csv_path = os.path.join(self.persist_dir, "metadata_summary.csv")
        with open(json_path, "w", encoding="utf-8") as fjson:
            json.dump({str(k): v for k, v in self.metadata.items()}, fjson, ensure_ascii=False, indent=2)

        # CSV: id, doc_id, source, chunk_index, text (truncated)
        with open(csv_path, "w", encoding="utf-8", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["id", "doc_id", "source", "chunk_index", "text_preview"])
            for k in sorted(self.metadata.keys()):
                m = self.metadata[k]
                text_preview = (m.get("text") or "").replace("\n", " ")[:200]
                writer.writerow([k, m.get("doc_id"), m.get("source"), m.get("chunk_index"), text_preview])

        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        # load index
        self.index = faiss.read_index(faiss_path)

        # if the index isn't an IDMap, try to wrap it for compatibility
        if not isinstance(self.index, faiss.IndexIDMap):
            try:
                # convert to IDMap by wrapping
                self.index = faiss.IndexIDMap(self.index)
            except Exception:
                pass

        # load metadata state; support old format where metadata was a list
        with open(meta_path, "rb") as f:
            state = pickle.load(f)

        if isinstance(state, dict) and "metadata" in state and "next_id" in state:
            self._next_id = state.get("next_id", 0)
            self.metadata = {int(k): v for k, v in state.get("metadata", {}).items()} if isinstance(state.get("metadata"), dict) else {}
        else:
            # legacy: a list of metadata dicts was saved - convert to id map
            legacy = state
            self.metadata = {}
            for i, m in enumerate(legacy):
                try:
                    mid = int(i)
                except Exception:
                    mid = i
                mm = dict(m)
                mm["id"] = mid
                self.metadata[mid] = mm
            self._next_id = max(self.metadata.keys()) + 1 if self.metadata else 0

        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir} (next_id={self._next_id})")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            # Faiss may return -1 for empty slots / missing neighbors
            if int(idx) < 0:
                continue
            meta = self.metadata.get(int(idx))
            results.append({"id": int(idx), "distance": float(dist), "metadata": meta})

        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from src.Rag_service.data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
