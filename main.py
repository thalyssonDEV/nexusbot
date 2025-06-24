import os
import re
import io
import base64
import logging
import uuid # Importado para gerar IDs de sessão únicos
from typing import Optional
from PIL import Image

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

# --- Configuração da API do Gemini ---
model = None
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente GEMINI_API_KEY não foi encontrada.")
    
    logging.info("Chave da API carregada com sucesso.")
    genai.configure(api_key=api_key)
    
    # Ajuste no nome do modelo para um mais comum e recente do Gemini
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    
    logging.info("Modelo Gemini inicializado e pronto para uso.")

except Exception as e:
    logging.critical(f"Falha crítica ao inicializar a API do Gemini: {e}")
    # O model permanecerá como None, e o endpoint retornará um erro.

# --- Configuração do FastAPI ---
app = FastAPI()

conversations = {}

# Modelo de dados para a requisição, agora com session_id
class ChatRequest(BaseModel):
    text: Optional[str] = ""
    image_base64: Optional[str] = None
    language: Optional[str] = "Português (Brasil)"
    session_id: Optional[str] = None # ID para rastrear a conversa

# Servindo a pasta de imagens estáticas
app.mount("/images", StaticFiles(directory="images"), name="images")


# Servindo o frontend
@app.get("/")
async def read_index(request: Request):
    if not os.path.exists("templates/index.html"):
        return {"error": "Arquivo 'templates/index.html' não encontrado."}
    return FileResponse('templates/index.html')


@app.post("/chat")
async def chat(request: ChatRequest):
    if not model:
        logging.error("Tentativa de chamada ao chat, mas o modelo não foi inicializado.")
        raise HTTPException(status_code=503, detail="O serviço de IA não está disponível. Verifique os logs do servidor.")
    
    if not request.text and not request.image_base64:
        raise HTTPException(status_code=400, detail="É necessário enviar texto ou imagem.")

    try:
        # --- Gerenciamento da Sessão de Chat ---
        session_id = request.session_id
        convo = None

        if session_id and session_id in conversations:
            # Se a sessão já existe, continua usando ela
            logging.info(f"Continuando sessão de chat existente: {session_id}")
            convo = conversations[session_id]
        else:
            # Se não, cria uma nova sessão
            session_id = str(uuid.uuid4())
            logging.info(f"Iniciando nova sessão de chat: {session_id}")
            convo = model.start_chat(history=[])
            conversations[session_id] = convo

        # --- Processamento da Mensagem ---
        logging.info(f"Recebida requisição para a sessão {session_id}. Imagem anexada: {'Sim' if request.image_base64 else 'Não'}")
        
        response_text = ""

        # Lógica para tratar imagens continua stateless para não poluir o histórico de texto
        if request.image_base64:
            prompt_parts = []
            try:
                image_data_str = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
                image_data = base64.b64decode(image_data_str)
                img = Image.open(io.BytesIO(image_data))
                prompt_parts.append(img)
            except (base64.binascii.Error, IOError) as e:
                logging.error(f"Erro ao processar imagem em base64: {e}")
                raise HTTPException(status_code=400, detail="Formato de imagem inválido ou corrompido.")

            if request.text:
                 prompt_parts.append(f"Responda em {request.language}. {request.text}")
            else: # Adiciona um prompt padrão se só houver imagem
                prompt_parts.append(f"Descreva esta imagem em {request.language}.")

            logging.info(f"Enviando prompt com imagem (stateless) para a sessão {session_id}.")
            response = model.generate_content(prompt_parts)
            response_text = response.text
        
        # Lógica para texto usa a conversa com histórico (stateful)
        elif request.text:
            prompt_with_lang = f"Responda em {request.language}. {request.text}"
            logging.info(f"Enviando prompt de texto (stateful) para a sessão {session_id}.")
            response = convo.send_message(prompt_with_lang)
            response_text = response.text

        # Verificação da resposta da API
        if not response_text:
            logging.warning(f"Resposta da API recebida, mas está vazia. Detalhes: {response.prompt_feedback}")
            raise HTTPException(status_code=500, detail="A API não retornou uma resposta de texto válida.")

        logging.info(f"Resposta enviada com sucesso para a sessão {session_id}.")
        # Retorna a resposta E o ID da sessão para o frontend
        return {"response": response_text, "session_id": session_id}

    except HTTPException as e:
        # Re-lança as exceções HTTP para que o FastAPI as trate
        raise e
    except Exception as e:
        logging.critical(f"Erro inesperado no endpoint /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno inesperado no servidor.")