from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any


class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.client = chromadb.PersistentClient(path="./chroma_db")

    def extract_text(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def chunk_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)

    def generate_embeddings(self, chunks: List[str]):
        return self.embedding_model.encode(chunks)

    def store_in_vector_db(self,
                           chunks: List[str],
                           embeddings,
                           metadata: Dict[str, Any]):
        collection = self.client.get_or_create_collection(name="reports")  # to avoid creating a new collection every time
        ids = [f"doc_{hash(chunk)}_{i}" for i, chunk in enumerate(chunks)]

        # Create one metadata dict per chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_length": len(chunk)
            })
            metadatas.append(chunk_metadata)

        collection.upsert( # upsert instead of add to avoid adding the same documents every time
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        return ids

    def process_document(self, pdf_path: str, report_metadata: Dict[str, Any]):
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)
        embeddings = self.generate_embeddings(chunks)
        stored_ids = self.store_in_vector_db(chunks, embeddings, report_metadata)
        return len(chunks), stored_ids