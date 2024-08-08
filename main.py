from data_manager import DataList
from session import Session
from utilities import load_config

from dotenv import load_dotenv, find_dotenv

import os

def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("\33[1;36m[Main]\33[0m: Avvio del programma")
    
    load_dotenv(find_dotenv())
    print("\33[1;32m[Main]\33[0m: File .env caricato")
    
    data_dir = load_config("config.yaml")["paths"]['content']["data"]
    data_list = DataList()
    data_list.add_dir(main_dir=data_dir, path="parags/", chunk_size=1000, chunk_overlap=0)
    data_list.add(path="link.txt")
    data = data_list.get_data()
    
    session = Session(title="Il tuo Assistente 👍", icon="👍")
    session.initialize_session_state(data)
    session.update()

if __name__ == "__main__":
    main()