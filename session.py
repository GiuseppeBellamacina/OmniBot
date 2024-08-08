from retriever import Retriever
from langchain_ollama.llms import OllamaLLM
from chains import ChainOfThoughts
from utilities import (
    load_config,
    StdOutHandler,
    ChatHistory
)

import streamlit as st

class Session():
    def __init__(self, title: str, icon: str, header: str = "", config_path: str = "config.yaml"):
        self.config_path = config_path
        st.set_page_config(page_title=title, page_icon=icon)
        st.title(title)
        if header != "":
            st.header(header)
        
        self.state = st.session_state
        print("\33[1;36m[Session]\33[0m: Sessione inizializzata")
        
    def initialize_session_state(self, data: list):
        if "is_initialized" not in self.state or not self.state.is_initialized:
            config = load_config(self.config_path)
            print("\33[1;34m[Session]\33[0m: Avvio inizializzazione")
            
            # Messaggi
            self.state.messages = []
            print("\33[1;32m[Session]\33[0m: Messaggi inizializzati")
            
            # History
            self.state.history = ChatHistory()
            print("\33[1;32m[Session]\33[0m: Store inizializzato")
            
            # Handler
            self.state.handler = StdOutHandler()
            print("\33[1;32m[Session]\33[0m: StreamHandler inizializzato")
            
            # Retriever
            self.state.retriever = Retriever(config, data)
            if self.state.retriever is None:
                print("\33[1;31m[Session]\33[0m: Retriever non inizializzato")
                return
            print("\33[1;32m[Session]\33[0m: Retriever inizializzato")

            # LLM
            self.state.llm = OllamaLLM(
                model=config['model']['name'],
                base_url=config['model']['base_url'],
                temperature=config['model']['temperature'],
                num_ctx=config['model']['num_ctx'],
                num_predict=config['model']['num_predict']
            )
            print("\33[1;32m[Session]\33[0m: LLM inizializzato")

            # Chain
            self.state.chain = ChainOfThoughts(
                llm=self.state.llm,
                retriever=self.state.retriever,
                threshold=config['retriever']['threshold'],
                simplifier=config['retriever']['simplifier'],
                history=self.state.history,
                handler=self.state.handler
            )
            print("\33[1;32m[Session]\33[0m: Chain inizializzata")
            self.state.is_initialized = True
            print("\33[1;32m[Session]\33[0m: Inizializzazione completata")
    
    def update(self):
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
        
        print("\33[1;35m[Chatbot]\33[0m: Chatbot pronto")

        if (faq_prompt == ""):
            if prompt := st.chat_input("Scrivi un messaggio...", key="first_question"):
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
                        response = self.state.chain.stream(input_dict, containers)
                
                signature = response.get('signature', None)
                if signature and signature != 'NegativeChain':
                    self.state.history.add_message_from_user(prompt)
                    self.state.history.add_message_from_response(response)
                
                self.state.messages.append({
                    "role": "ai",
                    "content": response.get('answer', None),
                    "response_time": self.state.handler.time
                })
        else:
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
                    response = self.state.chain.stream(input_dict, containers)
            
            signature = response.get('signature', None)
            if signature and signature != 'ConversationalChain':
                self.state.history.add_message_from_user(faq_prompt)
                self.state.history.add_message_from_response(response)
                
                # Limit history to 10 messages
                if len(self.state.history.messages) > 10:
                    self.state.history.messages = self.state.history.messages[-10:]
            
            self.state.messages.append({
                "role": "ai",
                "content": response.get('answer', None),
                "response_time": self.state.handler.time
            })
            
            faq_prompt = ""
            st.rerun()