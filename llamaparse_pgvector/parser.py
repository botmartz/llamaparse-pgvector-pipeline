import os
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ParsedChunk(BaseModel):
    chunk_id: str
    doc_name: str
    page_number: int
    content_markdown: str
    chunk_type: str  # "table", "text_block", "statutory_clause"
    sha256_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class LlamaParseExtractor:
    """
    Extracts structural Abstract Syntax Tree (AST) representations and Markdown tables
    from complex PDF, DOCX, and XLSX documents using the LlamaParse API.
    """
    def __init__(self, api_key: Optional[str] = None, result_type: str = "markdown"):
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY environment variable or argument is required.")
        self.result_type = result_type

    def parse_file(self, file_path: str) -> List[ParsedChunk]:
        from llama_parse import LlamaParse
        parser = LlamaParse(
            api_key=self.api_key,
            result_type=self.result_type,
            verbose=False,
            language="en"
        )
        documents = parser.load_data(file_path)
        doc_name = os.path.basename(file_path)
        chunks = []

        for idx, doc in enumerate(documents):
            page_num = doc.metadata.get("page_label", idx + 1)
            raw_text = doc.text
            
            # Identify whether the chunk contains Markdown table syntax
            is_table = "| ---" in raw_text or ("|" in raw_text and "\n|" in raw_text)
            chunk_type = "table" if is_table else "text_block"
            chunk_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

            chunk = ParsedChunk(
                chunk_id=f"{doc_name}_p{page_num}_{idx}",
                doc_name=doc_name,
                page_number=int(page_num) if str(page_num).isdigit() else idx + 1,
                content_markdown=raw_text,
                chunk_type=chunk_type,
                sha256_hash=chunk_hash,
                metadata={
                    "source_path": file_path,
                    "is_table": is_table,
                    "ast_parsed": True,
                    **doc.metadata
                }
            )
            chunks.append(chunk)

        return chunks
