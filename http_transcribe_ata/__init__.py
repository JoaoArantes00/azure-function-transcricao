import azure.functions as func
import json
import logging

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Função iniciada')
    
    results = {}
    
    # Testar requests
    try:
        import requests
        results["requests"] = "OK"
    except Exception as e:
        results["requests"] = str(e)
    
    # Testar azure-identity  
    try:
        from azure.identity import DefaultAzureCredential
        results["azure_identity"] = "OK"
    except Exception as e:
        results["azure_identity"] = str(e)
    
    # Testar azure-storage-blob
    try:
        from azure.storage.blob import BlobServiceClient
        results["azure_storage"] = "OK"
    except Exception as e:
        results["azure_storage"] = str(e)
    
    # Testar openai
    try:
        import openai
        results["openai"] = "OK"
    except Exception as e:
        results["openai"] = str(e)
    
    # Testar speech sdk
    try:
        import azure.cognitiveservices.speech as speechsdk
        results["speech_sdk"] = "OK"
    except Exception as e:
        results["speech_sdk"] = str(e)
    
    return func.HttpResponse(
        json.dumps({
            "status": "testing_imports",
            "results": results
        }, indent=2),
        mimetype="application/json",
        status_code=200
    )
