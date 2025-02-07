#!/usr/bin/env python3
import os
import time
from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate

# Disable detailed logs
set_debug(False)
set_verbose(False)

# -------------------------------
# CONFIGURATION: Set your PDF file path here.
PDF_FILE_PATH = "Data.pdf"  # Change this to your PDF file path

class ChatPDF:
    def __init__(self, llm_model: str = "mistral"):
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful assistant that can answer questions about the PDF document provided.",
                ),
                (
                    "human",
                    "Here are some document pieces: {context}\nQuestion: {question}",
                ),
            ]
        )
        self.vector_store = None
        self.retriever = None
        self.chain = None

    def ingest(self, pdf_file_path: str):
        print("Ingesting PDF document...")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory="chroma_db",
        )
        print("Ingestion completed.\n")

    def ask(self, query: str) -> str:
        if not self.vector_store:
            self.vector_store = Chroma(
            persist_directory="chroma_db", embedding=FastEmbedEmbeddings()
        )

        self.retriever = self.vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        result = self.chain.invoke(query)
    
    # Clean up the result to ensure no direct reference to the document is made
        if isinstance(result, dict):
            result = result.get("output", result)
    
    # Clean any reference to "PDF" or similar in the response
        if "document" in result.lower() or "pdf" in result.lower():
            result = result.replace("document", "").replace("pdf", "").strip()
        
        return result


    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

def main():
    # Ensure the PDF file exists.
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: The file '{PDF_FILE_PATH}' does not exist. Please update the PDF_FILE_PATH.")
        return

    # Initialize ChatPDF and ingest the PDF document.
    chat_pdf = ChatPDF()
    t0 = time.time()
    chat_pdf.ingest(PDF_FILE_PATH)
    t1 = time.time()
    print(f"Ingested PDF in {t1 - t0:.2f} seconds.\n")

    # Ask questions in a loop.
    while True:
        query = input("Enter your question (or type 'exit' to quit): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        print("\nProcessing your query...\n")
        answer = chat_pdf.ask(query)
        print("Answer:")
        print(answer)
        print("-" * 80)

if __name__ == "__main__":
    main()
