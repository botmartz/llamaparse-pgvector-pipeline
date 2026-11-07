# llamaparse-pgvector-pipeline

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Maintained by Botmartz](https://img.shields.io/badge/Maintained%20by-Botmartz%20AI-0066FF.svg)](https://www.botmartz.com)
[![Research Whitepaper](https://img.shields.io/badge/Research%20Whitepaper-Read%20Full%20Spec-0284C7.svg)](https://www.botmartz.com/insights/document-parsing-bottleneck-llamaparse-ast-layout-table-reconstruction)
[![Hugging Face Benchmark](https://img.shields.io/badge/Dataset-Enterprise--Doc--AST--Bench-FFD21E.svg)](https://huggingface.co/datasets/botmartzai/enterprise-doc-ast-bench)

**`llamaparse-pgvector-pipeline`** is an open-source enterprise connector engineered by **[Botmartz AI Solutions](https://www.botmartz.com)**. It bridges **LlamaParse** vision-augmented AST Markdown table output directly into **Neon Serverless PostgreSQL** and **pgvector** with automatic HNSW vector indexing and hybrid sparse Lucene search.

---

## The Problem It Solves

Standard RAG ingestion pipelines fail catastrophically on enterprise documents (SEC Form 10-K balance sheets, FAA aviation maintenance manuals, and legal schedules):
- **Naive chunking severs table rows**: Horizontal multi-column financial figures become decoupled from row headers.
- **Lost spatial geometry**: Numbers and dates lose semantic meaning when flattened into plaintext.
- **Embedding degradation**: Cosine similarity fails to retrieve specific alphanumeric cell values.

This connector preserves full table geometry as Markdown tables, generates dense embeddings, stamps every proposition chunk with a SHA-256 cryptographic provenance hash, and stores them in PostgreSQL with dual HNSW and GIN full-text indexes.

---

## Architectural Flow

```
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  Complex PDF /  │ ----> │ LlamaParse (VLM/AST) │ ----> │ Preserved Markdown     │
│  Financial 10-K │       │ Layout Parser Engine │       │ Table Tensors          │
└─────────────────┘       └──────────────────────┘       └────────────────────────┘
                                                                     │
                                                                     ▼
┌─────────────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  Reciprocal Rank Fusion │ <---- │ Neon / PostgreSQL    │ <---- │ Dense Vectors +        │
│  (RRF Constant k=60)    │       │ pgvector HNSW Index  │       │ SHA-256 Provenance     │
└─────────────────────────┘       └──────────────────────┘       └────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/botmartz/llamaparse-pgvector-pipeline.git
cd llamaparse-pgvector-pipeline
pip install -e .
```

---

## Quickstart

```python
import os
from llamaparse_pgvector import DocumentIngestionPipeline

# Set environment credentials
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..."
os.environ["DATABASE_URL"] = "postgresql://user:password@ep-cool-forest-1234.us-east-2.aws.neon.tech/neondb?sslmode=require"

# Initialize pipeline
pipeline = DocumentIngestionPipeline(
    embedding_model_name="BAAI/bge-small-en-v1.5",
    vector_dim=384
)

# Parse and index document
chunks = pipeline.run("samples/sec_form_10k_sample.pdf")
print(f"Indexed {len(chunks)} chunks into Neon pgvector.")
```

---

## Benchmark Metrics

Evaluated on the **[Enterprise-Doc-AST-Bench](https://huggingface.co/datasets/botmartzai/enterprise-doc-ast-bench)** dataset:

- **Table Cell Reconstruction F1**: `0.984` (vs. `0.418` for PyMuPDF)
- **Token Error Rate (TER)**: `0.32%` (vs. `28.4%` for raw extractors)
- **P99 Hybrid Search Latency**: `< 118ms`

Read the full technical dissertation: **[The Document Parsing Bottleneck: LlamaParse, AST Layout Engines, and Table Reconstruction in Production RAG](https://www.botmartz.com/insights/document-parsing-bottleneck-llamaparse-ast-layout-table-reconstruction)**.

---

## License

Licensed under the Apache License, Version 2.0. Copyright 2026 [Botmartz AI Solutions Private Limited](https://www.botmartz.com).
