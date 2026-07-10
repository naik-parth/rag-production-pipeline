import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class DocumentIngestionEngine:
    """Handles directory scanning, document loading, and semantic text chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # The Recursive text splitter keeps paragraphs and sentences together by default
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Loads and parses a PDF file page-by-page using PyPDFLoader."""
        print(f"📄 Loading PDF: {file_path}")
        try:
            # PyPDFLoader extracts text and automatically appends page numbers to metadata
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"❌ Error parsing PDF {file_path}: {str(e)}")
            return []

    def ingest_directory(self, directory_path: str) -> List[Document]:
        """Scans a local directory for PDFs, parses them, and returns semantic chunks."""
        raw_documents = []
        
        if not os.path.exists(directory_path):
            print(f"📁 Directory '{directory_path}' not found. Creating it now...")
            os.makedirs(directory_path)
            return []

        # Recursively scan the folder for any PDF files
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    loaded_pages = self.load_pdf(file_path)
                    raw_documents.extend(loaded_pages)

        if not raw_documents:
            print(f"⚠️ No PDF documents found in '{directory_path}/'.")
            return []

        # Break down the large raw pages into overlapping contextual chunks
        chunks = self.text_splitter.split_documents(raw_documents)
        print(f"✅ Success: Processed {len(raw_documents)} raw pages into {len(chunks)} split chunks.")
        return chunks