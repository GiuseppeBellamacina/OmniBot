from session import Session
from dotenv import load_dotenv, find_dotenv

def main():
    print("\33[1;36m[Main]\33[0m: Avvio del programma")
    
    load_dotenv(find_dotenv())
    print("\33[1;32m[Main]\33[0m: File .env caricato")
    
    session = Session(title="Il tuo Assistente 👍", icon="👍")
    session.initialize_session_state()
    session.update()

if __name__ == "__main__":
    main()