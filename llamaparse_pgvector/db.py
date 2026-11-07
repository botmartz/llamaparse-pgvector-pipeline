import os
import json
import psycopg
from typing import List, Dict, Any, Optional
from pgvector.psycopg import register_vector
from .parser import ParsedChunk

class PgVectorStorage:
    """
    Manages PostgreSQL pgvector storage, HNSW indexing, and hybrid Lucene/BM25 retrieval.
    Compatible with Neon Serverless Postgres, AWS RDS Postgres, and Supabase.
    """
    def __init__(self, db_url: Optional[str] = None, vector_dim: int = 1536):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable or argument is required.")
        self.vector_dim = vector_dim
        self.table_name = "botmartz_document_chunks"

    def init_schema(self):
        """Initializes pgvector extension, chunks table, HNSW vector index, and GIN full-text index."""
        with psycopg.connect(self.db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        doc_name TEXT NOT NULL,
                        page_number INTEGER NOT NULL,
                        content_markdown TEXT NOT NULL,
                        chunk_type TEXT NOT NULL,
                        sha256_hash TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.vector_dim}),
                        search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content_markdown)) STORED,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """)
                # HNSW cosine distance index
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_hnsw
                    ON {self.table_name} USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                # GIN index for sparse lexical BM25 matching
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tsv
                    ON {self.table_name} USING gin (search_tsv);
                """)
                conn.commit()

    def store_chunks(self, chunks: List[ParsedChunk]):
        with psycopg.connect(self.db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for c in chunks:
                    cur.execute(f"""
                        INSERT INTO {self.table_name}
                        (id, doc_name, page_number, content_markdown, chunk_type, sha256_hash, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content_markdown = EXCLUDED.content_markdown,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata;
                    """, (
                        c.chunk_id,
                        c.doc_name,
                        c.page_number,
                        c.content_markdown,
                        c.chunk_type,
                        c.sha256_hash,
                        json.dumps(c.metadata),
                        c.embedding
                    ))
                conn.commit()
