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
    #asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    #asyncio.run(main())

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
    finally:
        loop.run_until_complete(main())