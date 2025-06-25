import os
import re
import io
import base64
import logging
import uuid
import pickle  # Importado para serializar/desserializar objetos Python
from typing import Optional
from PIL import Image

import redis  # Importado para interagir com o Redis
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
    # A URL do Redis deve ser configurada na sua variável de ambiente
    # Exemplo no docker-compose: REDIS_URL=redis://redis:6379
    # Exemplo local: REDIS_URL=redis://localhost:6379
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise ValueError("A variável de ambiente REDIS_URL não foi encontrada.")
    
    # decode_responses=False é crucial porque o pickle trabalha com bytes, não com strings.
    redis_client = redis.from_url(redis_url, decode_responses=False)
    redis_client.ping() # Verifica se a conexão com o Redis foi bem-sucedida
    logging.info("Conexão com o Redis estabelecida com sucesso.")
except Exception as e:
    logging.critical(f"Falha crítica ao conectar com o Redis: {e}")
    # O app continuará, mas o endpoint de chat retornará erro.

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

# Modelo de dados para a requisição
class ChatRequest(BaseModel):
    text: Optional[str] = ""
    image_base64: Optional[str] = None
    language: Optional[str] = "Português (Brasil)"
    session_id: Optional[str] = None

# Servindo a pasta de imagens estáticas e o frontend
# Certifique-se que estas pastas existem na raiz do seu projeto.
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
        # --- Gerenciamento da Sessão de Chat com Redis ---
        session_id = request.session_id
        convo = None
        SESSION_EXPIRATION_SECONDS = 1800 # 30 minutos

        if session_id:
            # Tenta recuperar a conversa do Redis
            serialized_convo = redis_client.get(session_id)
            if serialized_convo:
                logging.info(f"Continuando sessão de chat existente: {session_id}")
                convo = pickle.loads(serialized_convo)
                # Atualiza o tempo de expiração da chave a cada nova mensagem
                redis_client.expire(session_id, SESSION_EXPIRATION_SECONDS)
        
        if convo is None:
            # Se não há sessão ou a sessão expirou, cria uma nova
            session_id = str(uuid.uuid4())
            logging.info(f"Iniciando nova sessão de chat: {session_id}")
            convo = model.start_chat(history=[])
            # A nova conversa será salva no Redis após o primeiro envio de mensagem

        # --- Processamento da Mensagem ---
        logging.info(f"Recebida requisição para a sessão {session_id}. Imagem anexada: {'Sim' if request.image_base64 else 'Não'}")
        
        response_text = ""
        response_from_api = None

        # Lógica para tratar imagens continua stateless para não poluir o histórico de texto
        if request.image_base64:
            prompt_parts = []
            try:
                # Importante: O frontend deve enviar a imagem em base64 puro, sem o prefixo.
                image_data = base64.b64decode(request.image_base64)
                img = Image.open(io.BytesIO(image_data)) # <--- Esta linha agora funciona
                prompt_parts.append(img)
            except Exception as e:
                logging.error(f"Erro ao processar imagem em base64: {e}")
                raise HTTPException(status_code=400, detail="Formato de imagem inválido ou corrompido.")

            prompt_parts.append(f"Responda em {request.language}. {request.text or 'Descreva esta imagem.'}")
            
            logging.info(f"Enviando prompt com imagem (stateless) para a sessão {session_id}.")
            response_from_api = model.generate_content(prompt_parts)
            response_text = response_from_api.text
        
        # Lógica para texto usa a conversa com histórico (stateful)
        elif request.text:
            prompt_with_lang = f"Responda em {request.language}. {request.text}"
            logging.info(f"Enviando prompt de texto (stateful) para a sessão {session_id}.")
            response_from_api = convo.send_message(prompt_with_lang)
            response_text = response_from_api.text

            # --- Salva o estado atualizado da conversa de volta no Redis ---
            # O objeto 'convo' foi modificado por send_message, então precisa ser salvo.
            serialized_convo_updated = pickle.dumps(convo)
            redis_client.set(session_id, serialized_convo_updated, ex=SESSION_EXPIRATION_SECONDS)
            logging.info(f"Sessão {session_id} salva/atualizada no Redis.")

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
        logging.critical(f"Erro inesperado no endpoint /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno inesperado no servidor.")