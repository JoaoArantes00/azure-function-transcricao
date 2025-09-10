import azure.functions as func
import json
import logging

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Função executando')
    
    if req.params.get("ping") == "1":
        return func.HttpResponse(
            '{"status": "success", "message": "Versão super simples funcionando"}',
            mimetype="application/json",
            status_code=200
        )
    
    return func.HttpResponse(
        '{"error": "Use ping=1"}',
        mimetype="application/json",
        status_code=400
    )
