from retriever import RetrieverBuilder
from langchain_ollama.llms import OllamaLLM
from chains import ChainOfThoughts
from utilities import (
    load_config,
    StdOutHandler,
    ChatHistory
)

import streamlit as st
import os

import sounddevice as sd
import soundfile as sf

import httpx

class Session():
    def __init__(self, page_title:str, title: str, icon: str, header: str = ""):
        st.set_page_config(page_title=page_title, page_icon=icon)
        st.title(title)
        if header != "":
            st.header(header)
        
        self.state = st.session_state
        
    def initialize_session_state(self):
        if "is_initialized" not in self.state or not self.state.is_initialized:
            self.state.is_initialized = False
            self.state.config = load_config()
            print("\33[1;36m[Session]\33[0m: Avvio inizializzazione")
            
            # Messaggi
            self.state.messages = []
            print("\33[1;32m[Session]\33[0m: Messaggi inizializzati")
            
            # History
            self.state.history = ChatHistory(self.state.config['history_size'])
            print("\33[1;32m[Session]\33[0m: ChatHistory inizializzata")

            # Handler
            self.state.handler = StdOutHandler(self.state.config, debug=False)
            print("\33[1;32m[Session]\33[0m: StreamHandler inizializzato")
            
            # Retriever
            self.state.retriever = RetrieverBuilder.build(self.state.config)
            if self.state.retriever is None:
                print("\33[1;31m[Session]\33[0m: Retriever non inizializzato")
                return self.state.is_initialized
            print("\33[1;32m[Session]\33[0m: Retriever inizializzato")

            # LLM
            self.state.llm = OllamaLLM(
                model=self.state.config['model']['name'],
                base_url=self.state.config['model']['base_url'],
                temperature=self.state.config['model']['temperature'],
                num_ctx=self.state.config['model']['num_ctx'],
                num_predict=self.state.config['model']['num_predict']
            )
            print("\33[1;32m[Session]\33[0m: LLM inizializzato")

            # Chain
            self.state.chain = ChainOfThoughts(
                llm=self.state.llm,
                handler=self.state.handler,
                name="ChainOfThoughts",
                history=self.state.history,
                retriever=self.state.retriever,
                retrieval_threshold=self.state.config['retrieval_threshold'],
                followup_threshold=self.state.config['followup_threshold'],
                distance_threshold=self.state.config['distance_threshold']
            )
            print("\33[1;32m[Session]\33[0m: Chain inizializzata")

            response = httpx.get("http://localhost:8000/start", timeout=20)
            if response.json().get("status", 'error') == 'error':
                error = response.json().get("message", "Error")
                print(error)
                raise Exception(error)
            elif response.json().get("status", 'error') == 'ready':
                print("\33[1;32m[Session]\33[0m: TTS inizializzato")

            self.state.is_initialized = True
            print("\33[1;32m[Session]\33[0m: Inizializzazione completata")
            return self.state.is_initialized
    
    async def update(self):
        if "is_initialized" not in self.state or not self.state.is_initialized:
            print("\33[1;31m[Session]\33[0m: Sessione non inizializzata")
            raise Exception(RuntimeError)

        faq_prompt = ""
        with st.sidebar:
            st.markdown("FAQ")
            if st.button("- Qual è l'iter formativo dei piloti in Accademia?"):
                faq_prompt = "Qual è l'iter formativo dei piloti in Accademia?"
                
            if st.button("- In cosa consiste la laurea in Medicina e Chirurgia?"):
                faq_prompt = "In cosa consiste la laurea in Medicina e Chirurgia?"

            if st.button("- Cosa sai dirmi sui concorsi per gli ufficiali?"):
                faq_prompt = "Cosa sai dirmi sui concorsi per gli ufficiali?"

            if st.button("- Come si fa la pasta alla carbonara?", use_container_width=True):
                faq_prompt = "Come si fa la pasta alla carbonara?"
            
            for _ in range(15):
                st.write("")
            
            if st.button("Clear", use_container_width=True):
                self.state.messages = []
                self.state.history.clear()
                print("\33[1;32m[Session]\33[0m: Sessione ripulita")
                st.success("Session cleared")

        for message in self.state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                if message['role'] == 'ai':
                    st.markdown(f"⏱ Tempo di risposta: {message['response_time']:.2f} secondi")
        
        #AUDIO
        if len(self.state.messages) >= 2:
            if os.path.exists("tmp.wav"):
                if st.button("Parla"):
                    data, fs = sf.read("tmp.wav", dtype="float32")
                    sd.play(data, fs)
                    sd.wait()
        
        print("\33[1;35m[Chatbot]\33[0m: Chatbot pronto")

        if (faq_prompt == ""):
            if prompt := st.chat_input("Scrivi un messaggio...", key="first_question"):
                os.system("cls" if os.name == "nt" else "clear")
                self.state.messages.append({
                    "role": "human",
                    "content": prompt
                })
                with st.chat_message("human"):
                    st.markdown(prompt)

                response = None
                input_dict = {"input": prompt}

                with st.chat_message("ai"):
                    containers = (st.empty(), st.empty())
                    with st.spinner("Elaborazione in corso..."):
                        response = await self.state.chain.astream(input_dict, containers)
                
                self.state.messages.append({
                    "role": "ai",
                    "content": response.get('answer', None),
                    "response_time": self.state.handler.time
                })

                st.rerun()
        else:
            os.system("cls" if os.name == "nt" else "clear")
            self.state.messages.append({
                    "role": "human",
                    "content": faq_prompt
            })
            
            with st.chat_message("human"):
                st.markdown(faq_prompt)

            response = None
            input_dict = {"input": faq_prompt}
            with st.chat_message("ai"):
                containers = (st.empty(), st.empty())
                with st.spinner("Elaborazione in corso..."):
                    response = await self.state.chain.astream(input_dict, containers)

            self.state.messages.append({
                "role": "ai",
                "content": response.get('answer', None),
                "response_time": self.state.handler.time
            })
            
            faq_prompt = ""
            st.rerun() # se lo tolgo, non si aggiorna la chat