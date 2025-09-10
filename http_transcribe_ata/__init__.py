import azure.functions as func
import json
import logging
import os
import traceback
from datetime import datetime
from urllib.parse import urlparse
import requests
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
import time
import threading
import unicodedata

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Função com todas as importações executando')
    
    try:
        if req.params.get("ping") == "1":
            return func.HttpResponse(
                json.dumps({
                    "status": "success", 
                    "message": "Todas as importações funcionando",
                    "imports_ok": True
                }),
                mimetype="application/json",
                status_code=200
            )
        
        return func.HttpResponse(
            json.dumps({"error": "Use ping=1"}),
            mimetype="application/json",
            status_code=400
        )
        
    except Exception as e:
        logging.error(f"Erro: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
