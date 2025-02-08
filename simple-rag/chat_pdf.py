#!/usr/bin/env python3
import os
import time
from typing import Iterator
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

PDF_FILE_PATH = "C:\\Users\\FA0555TX\\Desktop\\kaam\\local-assistant-examples\\simple-rag\\Data.pdf"

class ChatPDF:
    def __init__(self, llm_model: str = "mistral"):
        # Enable streaming in the model
        self.model = ChatOllama(model=llm_model, temperature=0.2, streaming=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful assistant that answers questions about provided content. "
                    "Answer based on the context below. If unsure, say you don't know.\n"
                    "Context:\n{context}",
                ),
                ("human", "Question: {question}"),
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

    def ask(self, query: str) -> Iterator[str]:
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

        return self.chain.stream(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

def main():
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: File '{PDF_FILE_PATH}' not found.")
        return

    chat_pdf = ChatPDF()
    t0 = time.time()
    chat_pdf.ingest(PDF_FILE_PATH)
    print(f"Ingested in {time.time()-t0:.2f}s\n")

    while True:
        query = input("\nQuestion (type 'exit' to quit): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        print("\nProcessing...\nAnswer:")
        try:
            # Stream the response
            for chunk in chat_pdf.ask(query):
                print(chunk, end="", flush=True)
                time.sleep(0.02)  # Simulate typing speed
            print("\n" + "-" * 80)
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()