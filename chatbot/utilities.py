from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.messages import HumanMessage, AIMessage

from time import time
import yaml

import asyncio
import httpx

from debugger import debug
from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str
    id: int

###  Messages ###

class MessageWithDocs():
    def __init__(self, message, documents):
        self.message = message
        self.documents = documents

    def embed_self(self, vectorizer):
        if not vectorizer:
            return []
        full_text = self.message.content
        if self.documents:
            full_text += "\n" + docs_to_string(self.documents)
        return vectorizer.transform([full_text]).toarray()[0]

class ChatHistory():
    def __init__(self, limit: int = 0):
        self.messages:  list[MessageWithDocs] = []
        self.vectorizer = None
        self.limit = limit
    
    def limit_history(self):
        if self.limit != 0:
            self.messages = self.messages[-self.limit:] # lascio solo gli ultimi messaggi
    
    def add_message_from_user(self, user_input: dict): # * ste funzioni vanno chiamate dopo che il modello ha finito di rispondere
        message = MessageWithDocs(
            message = HumanMessage(content = user_input.get('input', '')),
            documents = []
        )
        self.messages.append(message)
        self.limit_history()
    
    def add_message_from_response(self, response: dict):
        message = MessageWithDocs(
            message = AIMessage(content = response.get('answer', '')),
            documents = response.get("documents", [])
        )
        self.messages.append(message)
        self.limit_history()

    def train_vectorizer(self):
        all_texts = []
        ai_messagges = [msg for msg in self.messages if isinstance(msg.message, AIMessage)]
        for msg in ai_messagges:
            all_texts.append(msg.message.content + "\n" + docs_to_string(msg.documents))
        if all_texts:
            self.vectorizer = TfidfVectorizer(encoding='utf-8').fit(all_texts)
    
    def get_old_messages_ctx(self, threshold: float):
        if not self.vectorizer:
            return []
        user_message_vector = self.messages[-1].embed_self(self.vectorizer)
        ctx = []
        ai_messagges = [msg for msg in self.messages if isinstance(msg.message, AIMessage)]
        for msg in ai_messagges:
            vector = msg.embed_self(self.vectorizer)
            print("\33[1;33m[COSINE]\33[0m: Sto per far esplodere tutto")
            try:
                similarity = cosine_similarity([user_message_vector], [vector])[0][0]
                print("\33[1;32m[COSINE]\33[0m: Ce l'ho fatta")
            except Exception as e:
                print("\33[1;31m[COSINE]\33[0m: Non ce l'ho fatta")
                print("User Vector:", user_message_vector)
                print("Vector", vector)
                raise e
            if similarity > threshold:
                ctx.extend(msg.documents)
        if not ctx: # Se non ho trovato nessun contesto, prendo l'ultimo contesto
            ctx.extend(ai_messagges[-1].documents)
        return ctx
    
    @debug()
    def get_followup_ctx(self, threshold: float):
        self.train_vectorizer()
        if self.vectorizer is None:
            return []
        return self.get_old_messages_ctx(threshold)
    
    def get_all_messages(self):
        if self.messages == []:
            return []
        return [msg.message for msg in self.messages]

    def get_last_messages(self, n: int):
        if self.messages == []:
            return []
        if n > len(self.messages):
            n = len(self.messages)
        return [msg.message for msg in self.messages[-n:]]

    def clear(self):
        self.messages = []
        self.vectorizer = None

### Handler ###


class StdOutHandler:
    """
    Class to manage token's stream
    """
    def __init__(self, config, debug=False):
        self.containers = None
        self.text = ""
        self.chunks = []
        self.completed_chunks = []
        self.time = 0
        self.config = config
        self.debug = debug
        self.lock = asyncio.Lock()

    def start(self, containers=None):
        self.time = time()
        self.text = ""
        self.containers = containers
        self.chunks = []
        self.completed_chunks = []

    async def on_new_token(self, token: dict) -> None:
        token = token.get('answer', None)
        if self.debug:
            if token:
                print(token, sep="", end="", flush=True)
        if token:
            async with self.lock:
                self.text += token
            try:
                await self.generate_audio_stream()
            except Exception as e:
                print("\33[1;31m[STDOUTHANDLER]\33[0m: Errore durante la generazione dell'audio")
                self.error(e)
            if self.containers:
                self.containers[0].markdown(self.text)
    
    def sanitize_text(self, text: str) -> str:
        return text.translate({ord(i): None for i in "*\n\t"})
    
    def chunk_text(self, text: str) -> list[str]:
        stripped_text = self.sanitize_text(text)
        return [chunk.strip() for chunk in stripped_text.split(".") if chunk.strip()]

    async def generate_audio_stream(self):
        async with self.lock:
            if self.text:
                self.chunks = self.chunk_text(self.text)
        
        if len(self.chunks) > 1:
            async with httpx.AsyncClient() as client:
                for i in range(len(self.chunks) - 1):
                    async with self.lock:
                        if self.chunks[i] and i not in self.completed_chunks:
                            self.completed_chunks.append(i)
                            response = await client.post(
                                "http://localhost:8000/",
                                json=TextRequest(text=self.chunks[i], id=i).model_dump()
                            )
                            if response.json().get("status", 'error') == 'error':
                                self.error(Exception("Errore nell'invio del chunk"))

    async def end(self):
        async with self.lock:
            self.time = time() - self.time
            text_time = f"⏱ Tempo di risposta: {self.time:.2f} secondi"
            if self.debug:
                print(text_time)
            if self.containers:
                self.containers[1].markdown(text_time)
            if self.text:
                self.chunks = self.chunk_text(self.text)
                if self.chunks:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://localhost:8000/",
                            json=TextRequest(text=self.chunks[-1], id=len(self.chunks) - 1).model_dump()
                        )
                        if response.json().get("status", 'error') == 'error':
                            self.error(Exception("Errore nell'invio del chunk"))

                        # Controllo finale per il completamento
                        final_response = await client.get("http://localhost:8000/")
                        while final_response.json().get("status", 'error') == 'processing':
                            print("Risposta finale in elaborazione")
                            await asyncio.sleep(1)
                            final_response = await client.get("http://localhost:8000/")
                        if final_response.json().get("status", 'error') == 'ok':
                            print("Risposta finale ricevuta")
                        elif final_response.json().get("status", 'error') == 'error':
                            print("Errore nella risposta finale")
                            self.error(Exception("Errore nella risposta finale"))
            self.text = ""
            self.chunks = []
            self.completed_chunks = []

    def error(self, error: Exception):
        self.text = ""
        self.chunks = []
        self.completed_chunks = []
        self.time = time() - self.time
        text_time = f"⏱ Tempo di risposta: {self.time:.2f} secondi"
        if self.debug:
            print(f"\33[1;31m[STDOUTHANDLER]\33[0m: {error}")
            print(text_time)
        if self.containers:
            self.containers[0].markdown(f"Errore: {error}")
            self.containers[1].markdown(text_time)
        raise error

def load_config(file_path='config.yaml'):
    """
    Load configuration file
    
    Args:
        file_path (str): Configuration file path
    
    Returns:
        dict: Configuration file
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def docs_to_string(docs, sep="\n\n"):
    if docs:
        return f"{sep}".join([d.page_content for d in docs])
    return ""