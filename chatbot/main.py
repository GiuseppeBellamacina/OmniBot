from session import Session
from dotenv import load_dotenv, find_dotenv
import asyncio

async def main():
    print("\33[1;36m[Main]\33[0m: Avvio del programma")
    
    load_dotenv(find_dotenv())
    print("\33[1;32m[Main]\33[0m: File .env caricato")
    
    session = Session(page_title="Chatbot", title="Il tuo Assistente 🤖", icon="🤖")
    session.initialize_session_state()
    await session.update()

if __name__ == "__main__":
    asyncio.run(main())