import azure.functions as func
import json
import logging

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('=== TESTE FUNÇÃO SIMPLES ===')
    
    try:
        if req.params.get("ping") == "1":
            return func.HttpResponse(
                json.dumps({"status": "success", "message": "Função simples funcionando!"}),
                mimetype="application/json",
                status_code=200
            )
        
        return func.HttpResponse(
            json.dumps({"error": "Use ?ping=1 para testar"}),
            mimetype="application/json", 
            status_code=400
        )
        
    except Exception as e:
        logging.error(f'Erro: {str(e)}')
        return func.HttpResponse(
            json.dumps({"error": f"Erro capturado: {str(e)}"}),
            mimetype="application/json",
            status_code=500
        )
