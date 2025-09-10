import azure.functions as func
import json
import logging
import os
from datetime import datetime

# Imports com tratamento de erro
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests não disponível")

try:
    import azure.cognitiveservices.speech as speechsdk
    SPEECH_SDK_AVAILABLE = True
except ImportError:
    SPEECH_SDK_AVAILABLE = False
    logging.warning("Azure Speech SDK não disponível")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI não disponível")

try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logging.warning("Azure Storage não disponível")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Função iniciada')
    
    try:
        # Teste de ping
        if req.params.get("ping") == "1":
            return func.HttpResponse(
                json.dumps({
                    "status": "success",
                    "message": "Serviço funcionando",
                    "libraries": {
                        "requests": REQUESTS_AVAILABLE,
                        "speech_sdk": SPEECH_SDK_AVAILABLE, 
                        "openai": OPENAI_AVAILABLE,
                        "storage": STORAGE_AVAILABLE
                    }
                }),
                mimetype="application/json",
                status_code=200
            )
        
        # Verificar se bibliotecas necessárias estão disponíveis
        if not all([REQUESTS_AVAILABLE, SPEECH_SDK_AVAILABLE, OPENAI_AVAILABLE, STORAGE_AVAILABLE]):
            return func.HttpResponse(
                json.dumps({
                    "error": "Bibliotecas necessárias não disponíveis",
                    "libraries": {
                        "requests": REQUESTS_AVAILABLE,
                        "speech_sdk": SPEECH_SDK_AVAILABLE,
                        "openai": OPENAI_AVAILABLE, 
                        "storage": STORAGE_AVAILABLE
                    }
                }),
                mimetype="application/json",
                status_code=500
            )
        
        return func.HttpResponse(
            json.dumps({"message": "Use ?ping=1 para testar"}),
            mimetype="application/json",
            status_code=400
        )
        
    except Exception as e:
        logging.error(f'Erro: {str(e)}')
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
