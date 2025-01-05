from fastapi import FastAPI, BackgroundTasks
import uvicorn
import numpy as np
import soundfile as sf
from TTS.api import TTS
import asyncio
from utilities import load_config, TextRequest
import tiktoken

app = FastAPI()
config = None
buffer = None
maker = None

class AudioFragment:
    """Rappresenta un frammento audio generato dal TTS."""
    def __init__(self, content, id):
        self.content = content
        self.id = id

    def __repr__(self):
        return f"AudioFragment {self.id} (len={len(self.content)})"

class AudioBuffer:
    """Gestisce il buffering di testi e frammenti audio."""
    def __init__(self):
        self.texts = []  # lista dei testi in arrivo
        self.fragments = []  # lista dei frammenti audio
        self.lock = asyncio.Lock()

    async def add_text(self, text: str, id: int):
        """Aggiunge un testo al buffer."""
        async with self.lock:
            self.texts.append((text, id))
    
    async def add_fragment(self, fragment: AudioFragment):
        """Aggiunge un frammento audio al buffer."""
        async with self.lock:
            self.fragments.append(fragment)

    async def get_audio(self):
        """Restituisce l'audio completo concatenando i frammenti."""
        async with self.lock:
            sorted_fragments = sorted(self.fragments, key=lambda x: x.id)
            audio = np.concatenate([fragment.content for fragment in sorted_fragments])
            return audio

    async def is_complete(self):
        """Verifica se tutti i testi hanno un frammento audio associato."""
        async with self.lock:
            return len(self.fragments) == len(self.texts) and len(self.texts) > 0

    async def clear(self):
        """Resetta il buffer."""
        async with self.lock:
            self.texts = []
            self.fragments = []

class AudioMaker:
    """Gestisce la generazione di audio con TTS."""
    def __init__(self, config, buffer: AudioBuffer):
        self.config = config
        self.tts = TTS(model_name=self.config["tts_model"]).to("cuda")
        self.buffer = buffer
        print("\33[1;34m[AUDIO MAKER]\33[0m Audio maker initialized")

    def split_text_into_chunks(self, text, max_tokens, encoding="cl100k_base"):
        """Divide il testo in segmenti rispettando il limite massimo di token."""
        tokenizer = tiktoken.get_encoding(encoding)
        tokens = tokenizer.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(tokenizer.decode(chunk_tokens))
        return chunks

    async def generate_audio_fragment(self, text: str, id: int):
        """Genera frammenti audio gestendo il limite massimo di token."""
        max_tokens = 400 # Limite massimo di token per frammento
        text_chunks = self.split_text_into_chunks(text, max_tokens)

        for index, chunk in enumerate(text_chunks):
            fragment = await asyncio.to_thread( # TODO: gestire la memoria della GPU
                self.tts.tts,
                text=chunk,
                language="it",
                speaker=self.config["speakers"][self.config["speaker_index"]],
                speed=2.0
            )
            audio_fragment = AudioFragment(content=fragment, id=id)
            await self.buffer.add_fragment(audio_fragment)
            print(f"\33[1;34m[AUDIO MAKER]\33[0m Generated fragment for ID {id}-{index}")

    async def save_audio(self, path: str):
        """Salva l'audio concatenato in un file."""
        audio = await self.buffer.get_audio()
        sf.write(path, audio, 22050, format="WAV")
        await self.buffer.clear()
        print("\33[1;34m[AUDIO MAKER]\33[0m Audio saved at", path)

@app.post("/")
async def stream(text: TextRequest, background_tasks: BackgroundTasks):
    """Riceve un testo e avvia la generazione del frammento audio."""
    try:
        await buffer.add_text(text.text, text.id)  # Aggiungi il testo al buffer
        background_tasks.add_task(maker.generate_audio_fragment, text.text, text.id)  # Genera frammento in background
        return {"status": "processing"}
    except Exception as e:
        print("\33[1;31m[AUDIO MAKER]\33[0m Error:", e)
        return {"status": "error", "message": str(e)}

@app.get("/")
async def save_audio_file():
    """Controlla se l'audio è completo e lo salva."""
    try:
        if await buffer.is_complete():
            await maker.save_audio("tmp.wav")
            return {"status": "ok"}
        else:
            return {"status": "processing"}
    except Exception as e:
        print("\33[1;31m[AUDIO MAKER]\33[0m Error:", e)
        return {"status": "error", "message": str(e)}
    
@app.get("/start")
def start():
    global config, buffer, maker
    try:
        config = load_config()
        buffer = AudioBuffer()
        maker = AudioMaker(config, buffer)
        return {"status": "ready"}
    except Exception as e:
        print("\33[1;31m[AUDIO MAKER]\33[0m Error:", e)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)