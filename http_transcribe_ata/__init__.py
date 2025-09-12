import azure.functions as func
import json
import logging

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Versão ultra-simples apenas para testar conectividade"""
    
    try:
        # Ping test
        if req.params.get("ping") == "1":
            return func.HttpResponse(
                json.dumps({
                    "ok": True,
                    "status": "pong",
                    "marker": "ULTRA_SIMPLE_TEST",
                    "message": "Backend ultra-simples funcionando!"
                }),
                mimetype="application/json",
                status_code=200
            )
        
        # Para qualquer requisição POST, retornar sucesso fake
        if req.method == "POST":
            audio_bytes = req.get_body()
            
            return func.HttpResponse(
                json.dumps({
                    "ok": True,
                    "marker": "ULTRA_SIMPLE_TEST",
                    "message": "Arquivo recebido com sucesso!",
                    "processing": {
                        "audio_size_bytes": len(audio_bytes),
                        "transcript_chars": 150,
                        "ata_generated": True
                    },
                    "preview": {
                        "transcript": "Esta é uma transcrição de teste. O arquivo foi recebido corretamente e o sistema está funcionando.",
                        "ata": "# ATA de Teste\n\nO sistema de transcrição está funcionando corretamente. Arquivo processado com sucesso."
                    }
                }),
                mimetype="application/json",
                status_code=200
            )
        
        return func.HttpResponse(
            json.dumps({"ok": False, "error": "Método não suportado"}),
            mimetype="application/json",
            status_code=405
        )
        
    except Exception as e:
        return func.HttpResponse(
            json.dumps({
                "ok": False,
                "error": f"Erro: {str(e)}",
                "marker": "ULTRA_SIMPLE_TEST"
            }),
            mimetype="application/json",
            status_code=500
        )
