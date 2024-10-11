from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    WebBaseLoader,
    DataFrameLoader
)

from data_manager import Data, DataType

import pandas as pd
import bs4

class Splitter():
    def __init__(self, dir_path: dict):
        self.dir_path = dir_path
    
    def TextChunks(self, data: Data) -> list[Document]:
        try:
            path = self.dir_path + data.path
            loader = TextLoader(path, encoding="utf-8")
            if data.chunk_size == 0:
                data.chunk_size = len(open(path, "r", encoding="utf-8").read()) # Set chunk size to the length of the document
            splitter = RecursiveCharacterTextSplitter(chunk_size=data.chunk_size, chunk_overlap=data.chunk_overlap)
            loaded = loader.load()
            splits = splitter.split_documents(loaded)
            new_splits = []
            for s in splits:
                title = open(path, "r", encoding="utf-8").readline().strip()
                s.metadata["title"] = title
                source = data.path
                content = s.page_content
                if title == content and len(splits) > 1:
                    continue # Skip if title is the same as content and there are multiple chunks
                final_content = f"\\TITLE: {title}\\SOURCE: {source}\\BODY: {content}"
                s.page_content = final_content
                new_splits.append(s)
            print(f"\33[1;32m[Splitter]\33[0m: Creati {len(new_splits)} chunks di tipo Text per", data.path)
            return new_splits
        except Exception as e:
            print(f"\33[1;31m[Splitter]\33[0m: Errore durante la creazione dei chunks di tipo Text di {data.path}: {e}")
            raise e
    
    def WebChunks(self, data: Data) -> list[Document]:
        try:
            loader = WebBaseLoader(
                web_paths=(data.path,),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=(data.extra)
                    )
                ),
                encoding="utf-8"
            )
            splitter = RecursiveCharacterTextSplitter(chunk_size=data.chunk_size, chunk_overlap=data.chunk_overlap)
            loaded = loader.load()
            splits = splitter.split_documents(loaded)
            print(f"\33[1;32m[Splitter]\33[0m: Creati {len(splits)} chunks di tipo Web per", data.path)
            return splits
        except Exception as e:
            print(f"\33[1;31m[Splitter]\33[0m: Errore durante la creazione dei chunks di tipo Web di {data.path}: {e}")
            raise e
    
    def PDFChunks(self, data: Data) -> list[Document]:
        try:
            path = self.dir_path + data.path
            loader = PyPDFDirectoryLoader(path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=data.chunk_size, chunk_overlap=data.chunk_overlap)
            loaded = loader.load()
            splits = splitter.split_documents(loaded)
            print(f"\33[1;32m[Splitter]\33[0m: Creati {len(splits)} chunks di tipo PDF per", data.path)
            return splits
        except Exception as e:
            print(f"\33[1;31m[Splitter]\33[0m: Errore durante la creazione dei chunks di tipo PDF di {data.path}: {e}")
            raise e
    
    def DFChunks(self, data: Data) -> list[Document]:
        try:
            path = self.dir_path + data.path
            df = pd.read_csv(path)
            loader = DataFrameLoader(df, page_content_column=data.extra)
            splitter = RecursiveCharacterTextSplitter(chunk_size=data.chunk_size, chunk_overlap=data.chunk_overlap)
            loaded = loader.load()
            splits = splitter.split_documents(loaded)
            for s in splits:
                title = s.metadata["title"]
                description = s.metadata["description"]
                content = s.page_content
                url = s.metadata["url"]
                final_content = f"\\TITLE: {title}\\DESCRIPTION: {description}\\BODY: {content}\\nURL: {url}"
                s.page_content = final_content
            print(f"\33[1;32m[Splitter]\33[0m: Creati {len(splits)} chunks di tipo DataFrame per", data.path)
            return splits
        except Exception as e:
            print(f"\33[1;31m[Splitter]\33[0m: Errore durante la creazione dei chunks di tipo DataFrame di {data.path}: {e}")
            raise e
    
    def create_chunks(self, data: list[Data]) -> list[Document]:
        """
        Create chunks of given data

        Args:
            data (list[Data]): List of data

        Returns:
            list[Document]: List of chunks
        """
        chunks = []

        for d in data:
            if d.data_type == DataType.TEXT:
                chunks += self.TextChunks(d)
            elif d.data_type == DataType.WEB:
                chunks += self.WebChunks(d)
            elif d.data_type == DataType.PDF:
                chunks += self.PDFChunks(d)
            elif d.data_type == DataType.CSV:
                chunks += self.DFChunks(d)
        
        # Assign unique IDs to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["id"] = i
        
        print(f"\33[1;32m[Splitter]\33[0m: Creati {len(chunks)} chunks totali")
        return chunks