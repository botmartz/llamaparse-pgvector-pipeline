import os
from typing import List, Dict, Any, Optional
from .parser import LlamaParseExtractor, ParsedChunk
from .db import PgVectorStorage

class DocumentIngestionPipeline:
    """
    End-to-End Enterprise Ingestion Pipeline:
    PDF Document -> LlamaParse AST Parsing -> Dense Embeddings -> Neon pgvector HNSW Storage
    """
    def __init__(
        self,
        llama_api_key: Optional[str] = None,
        db_url: Optional[str] = None,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        vector_dim: int = 384
    ):
        self.extractor = LlamaParseExtractor(api_key=llama_api_key)
        self.storage = PgVectorStorage(db_url=db_url, vector_dim=vector_dim)
        self.embedding_model_name = embedding_model_name
        self._model = None

    def _get_embedding_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def run(self, file_path: str) -> List[ParsedChunk]:
        print(f"[Botmartz Pipeline] Initializing database schema...")
        self.storage.init_schema()

        print(f"[Botmartz Pipeline] Parsing document layout with LlamaParse: {file_path}")
        chunks = self.extractor.parse_file(file_path)
        print(f"[Botmartz Pipeline] Extracted {len(chunks)} structural chunks.")

        print(f"[Botmartz Pipeline] Generating embeddings with {self.embedding_model_name}...")
        model = self._get_embedding_model()
        texts = [c.content_markdown for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        print(f"[Botmartz Pipeline] Ingesting into PostgreSQL pgvector HNSW index...")
        self.storage.store_chunks(chunks)
        print(f"[Botmartz Pipeline] Successfully indexed {len(chunks)} chunks with SHA-256 provenance.")

        return chunks
