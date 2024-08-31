from langchain_community.docstore import InMemoryDocstore
from langchain_cohere import CohereEmbeddings
from data_manager import DataList
from db_maker import DBMaker
from utilities import load_config
from vectorstore import CustomStore
from dotenv import load_dotenv, find_dotenv
import faiss
import os

def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("\33[1;34m[Main]\33[0m: Avvio dello script di creazione database")
    
    config = load_config()
    load_dotenv(find_dotenv())
    print("\33[1;32m[Main]\33[0m: File di configurazione caricato")
    
    # Load data
    content_data_list = DataList(config)
    content_data_list.add_dir(
        path="txts_parags/",
        chunk_size=1000,
        chunk_overlap=0
    )
    content_data_list.add(path="link.txt")
    
    # Check if data is valid
    if not content_data_list.test():
        print("\33[1;31m[Main]\33[0m: Errore nei dati")
        return
    data = content_data_list.get_data()
    
    embedder = CohereEmbeddings(model=config["embedder"])

    index = faiss.IndexFlatL2(len(embedder.embed_query("index")))
    vectorstore = CustomStore(
        embedding_function=embedder,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    db_maker = DBMaker(config, vectorstore)
    db_maker.make(data)
    print("\33[1;32m[Main]\33[0m: Database creato")

if __name__ == "__main__":
    main()