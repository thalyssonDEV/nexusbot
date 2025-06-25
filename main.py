import os
import re
import io
import base64
import logging
import uuid
import pickle
from typing import Optional
from PIL import Image

import redis
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração do Cliente Redis ---
redis_client = None
try:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise ValueError("A variável de ambiente REDIS_URL não foi encontrada.")
    redis_client = redis.from_url(redis_url, decode_responses=False)
    redis_client.ping()
    logging.info("Conexão com o Redis estabelecida com sucesso.")
except Exception as e:
    logging.critical(f"Falha crítica ao conectar com o Redis: {e}")

# --- Configuração da API do Gemini ---
model = None
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente GEMINI_API_KEY não foi encontrada.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    logging.info("Modelo Gemini inicializado e pronto para uso.")
except Exception as e:
    logging.critical(f"Falha crítica ao inicializar a API do Gemini: {e}")

# --- Configuração do FastAPI ---
app = FastAPI()

class ChatRequest(BaseModel):
    text: Optional[str] = ""
    image_base64: Optional[str] = None
    language: Optional[str] = "Português (Brasil)"
    session_id: Optional[str] = None

app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/")
async def read_index(request: Request):
    if not os.path.exists("templates/index.html"):
        logging.error("Arquivo 'templates/index.html' não encontrado.")
        return {"error": "Arquivo 'templates/index.html' não encontrado."}
    return FileResponse('templates/index.html')


@app.post("/chat")
async def chat(request: ChatRequest):
    if not model or not redis_client:
        detail_msg = "O serviço de IA ou o serviço de sessão (Redis) não estão disponíveis."
        logging.error(f"Tentativa de chamada ao chat, mas um serviço essencial não está inicializado. IA: {'OK' if model else 'FALHA'}, Redis: {'OK' if redis_client else 'FALHA'}")
        raise HTTPException(status_code=503, detail=detail_msg)
    
    if not request.text and not request.image_base64:
        raise HTTPException(status_code=400, detail="É necessário enviar texto ou imagem.")

    try:
        session_id = request.session_id
        history = [] # <--- ALTERAÇÃO: Começamos com um histórico vazio
        SESSION_EXPIRATION_SECONDS = 1800 # 30 minutos

        if session_id:
            # <--- ALTERAÇÃO: Tenta recuperar o histórico, não o objeto de chat.
            serialized_history = redis_client.get(session_id)
            if serialized_history:
                logging.info(f"Continuando sessão de chat existente: {session_id}")
                history = pickle.loads(serialized_history)
                redis_client.expire(session_id, SESSION_EXPIRATION_SECONDS)
        
        if not session_id or not history:
            # Se não há sessão ou o histórico está vazio, inicia uma nova sessão.
            session_id = str(uuid.uuid4())
            history = []
            logging.info(f"Iniciando nova sessão de chat: {session_id}")

        # <--- ALTERAÇÃO: Cria um novo objeto de chat em cada requisição, usando o histórico recuperado.
        convo = model.start_chat(history=history)

        logging.info(f"Recebida requisição para a sessão {session_id}. Imagem anexada: {'Sim' if request.image_base64 else 'Não'}")
        
        response_text = ""
        response_from_api = None

        if request.image_base64:
            prompt_parts = []
            try:
                image_data = base64.b64decode(request.image_base64)
                img = Image.open(io.BytesIO(image_data))
                prompt_parts.append(img)
            except Exception as e:
                logging.error(f"Erro ao processar imagem em base64: {e}")
                raise HTTPException(status_code=400, detail="Formato de imagem inválido ou corrompido.")

            prompt_parts.append(f"Responda em {request.language}. {request.text or 'Descreva esta imagem.'}")
            
            logging.info(f"Enviando prompt com imagem (stateless) para a sessão {session_id}.")
            response_from_api = model.generate_content(prompt_parts)
            response_text = response_from_api.text
        
        elif request.text:
            prompt_with_lang = f"Responda em {request.language}. {request.text}"
            logging.info(f"Enviando prompt de texto (stateful) para a sessão {session_id}.")
            response_from_api = convo.send_message(prompt_with_lang)
            response_text = response_from_api.text

            # --- ALTERAÇÃO: Salva o histórico atualizado, não o objeto de chat.
            updated_history = convo.history
            serialized_history_updated = pickle.dumps(updated_history)
            redis_client.set(session_id, serialized_history_updated, ex=SESSION_EXPIRATION_SECONDS)
            logging.info(f"Histórico da sessão {session_id} salvo/atualizado no Redis.")

        if not response_text:
            feedback = getattr(response_from_api, 'prompt_feedback', 'N/A')
            logging.warning(f"Resposta da API para a sessão {session_id} está vazia. Detalhes: {feedback}")
            raise HTTPException(status_code=500, detail="A API não retornou uma resposta de texto válida.")

        logging.info(f"Resposta enviada com sucesso para a sessão {session_id}.")
        return {"response": response_text, "session_id": session_id}

    except redis.exceptions.RedisError as e:
        logging.critical(f"Erro de comunicação com o Redis: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Não foi possível conectar ao serviço de sessão.")
    except Exception as e:
        # Apanha o erro de pickle aqui também, por segurança.
        if "pickle" in str(e).lower():
             logging.critical(f"Erro de serialização (pickle): {e}", exc_info=True)
             raise HTTPException(status_code=500, detail="Ocorreu um erro ao tentar salvar o estado da conversa.")
        logging.critical(f"Erro inesperado no endpoint /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno inesperado no servidor.")
