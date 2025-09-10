import azure.functions as func
import json
import logging
import os
import traceback
from datetime import datetime
from urllib.parse import urlparse
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from openai import AzureOpenAI

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Testando OpenAI')
    
    try:
        if req.params.get("ping") == "1":
            return func.HttpResponse(
                json.dumps({
                    "status": "success", 
                    "message": "OpenAI OK",
                    "tested": ["basic_imports", "azure.identity", "azure.storage.blob", "openai"]
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
