import azure.functions as func
import json
import logging
import os
import traceback
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import requests
import time
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings, BlobSasPermissions, generate_blob_sas

MARKER = "BATCH_TRANSCRIPTION_V1"

def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, x-functions-key",
        "Access-Control-Max-Age": "3600"
    }

def _blob_service():
    account = os.getenv("STORAGE_ACCOUNT_NAME")
    cred = DefaultAzureCredential()
    return BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=cred
    )

def _make_upload_sas(container, blob_name, minutes_valid=120):
    bsc = _blob_service()
    account = os.getenv("STORAGE_ACCOUNT_NAME")
    now = datetime.now(tz=timezone.utc)
    
    udk = bsc.get_user_delegation_key(
        key_start_time=now - timedelta(minutes=5),
        key_expiry_time=now + timedelta(minutes=minutes_valid)
    )
    
    sas = generate_blob_sas(
        account_name=account,
        container_name=container,
        blob_name=blob_name,
        user_delegation_key=udk,
        permission=BlobSasPermissions(create=True, write=True, add=True, read=True),
        expiry=now + timedelta(minutes=minutes_valid),
        start=now - timedelta(minutes=5)
    )
    
    blob_url = f"https://{account}.blob.core.windows.net/{container}/{quote(blob_name)}"
    put_url = f"{blob_url}?{sas}"
    
    return {
        "blob_url": blob_url,
        "put_url": put_url,
        "sas_valid_minutes": minutes_valid
    }

def _upload_to_storage(container, blob_name, content, content_type="text/plain"):
    try:
        bsc = _blob_service()
        container_client = bsc.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        
        if isinstance(content, bytes):
            data = content
        else:
            data = content.encode("utf-8")
        
        if content_type.startswith("text/") or content_type.startswith("application/json"):
            cs = ContentSettings(content_type=f"{content_type}; charset=utf-8")
        else:
            cs = ContentSettings(content_type=content_type)
        
        blob_client.upload_blob(data=data, overwrite=True, content_settings=cs)
        
        account = os.getenv("STORAGE_ACCOUNT_NAME")
        return f"https://{account}.blob.core.windows.net/{container}/{quote(blob_name)}"
    except Exception as e:
        logging.exception("Erro no upload para storage")
        raise RuntimeError(f"Erro no upload: {str(e)}")

def _download_to_bytes(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def _require_env(keys):
    missing = [k for k in keys if not os.getenv(k)]
    ok = len(missing) == 0
    msg = f"Variaveis nao configuradas: {', '.join(missing)}" if missing else ""
    return (ok, msg)

def _create_batch_transcription(audio_blob_url, language="pt-BR"):
    """Cria um batch transcription job no Azure Speech Service"""
    ok, msg = _require_env(["SPEECH_KEY", "SPEECH_REGION"])
    if not ok:
        raise RuntimeError(msg)
    
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    # URL da API de Batch Transcription
    api_url = f"https://{speech_region}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"
    
    # Configura o job
    transcription_config = {
        "contentUrls": [audio_blob_url],
        "locale": language,
        "displayName": f"ATA_Transcription_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "properties": {
            "diarizationEnabled": False,
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "None"
        }
    }
    
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/json"
    }
    
    logging.info(f"Criando batch transcription job...")
    response = requests.post(api_url, json=transcription_config, headers=headers)
    
    if response.status_code not in [200, 201]:
        logging.error(f"Erro ao criar batch transcription: {response.status_code} - {response.text}")
        raise RuntimeError(f"Falha ao criar transcription job: {response.text}")
    
    result = response.json()
    transcription_id = result["self"].split("/")[-1]
    
    logging.info(f"Batch transcription criado: {transcription_id}")
    return {
        "transcription_id": transcription_id,
        "status_url": result["self"],
        "status": result.get("status", "NotStarted")
    }

def _get_transcription_status(transcription_id):
    """Verifica o status de um batch transcription job"""
    ok, msg = _require_env(["SPEECH_KEY", "SPEECH_REGION"])
    if not ok:
        raise RuntimeError(msg)
    
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    api_url = f"https://{speech_region}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions/{transcription_id}"
    
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key
    }
    
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        logging.error(f"Erro ao obter status: {response.status_code} - {response.text}")
        raise RuntimeError(f"Falha ao obter status: {response.text}")
    
    result = response.json()
    status = result.get("status", "Unknown")
    
    logging.info(f"Status do job {transcription_id}: {status}")
    
    return {
        "transcription_id": transcription_id,
        "status": status,
        "created_date": result.get("createdDateTime"),
        "last_action_date": result.get("lastActionDateTime"),
        "files_url": result.get("links", {}).get("files") if status == "Succeeded" else None
    }

def _get_transcription_result(transcription_id):
    """Busca o resultado de uma transcricao completa"""
    ok, msg = _require_env(["SPEECH_KEY", "SPEECH_REGION"])
    if not ok:
        raise RuntimeError(msg)
    
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    # Primeiro pega os arquivos
    files_url = f"https://{speech_region}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions/{transcription_id}/files"
    
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key
    }
    
    response = requests.get(files_url, headers=headers)
    
    if response.status_code != 200:
        raise RuntimeError(f"Falha ao obter arquivos: {response.text}")
    
    files = response.json().get("values", [])
    
    # Procura o arquivo de transcricao
    transcription_file = None
    for file in files:
        if file.get("kind") == "Transcription":
            transcription_file = file
            break
    
    if not transcription_file:
        raise RuntimeError("Arquivo de transcricao nao encontrado")
    
    # Baixa o conteudo da transcricao
    content_url = transcription_file.get("links", {}).get("contentUrl")
    if not content_url:
        raise RuntimeError("URL do conteudo nao encontrada")
    
    content_response = requests.get(content_url)
    
    if content_response.status_code != 200:
        raise RuntimeError(f"Falha ao baixar transcricao: {content_response.text}")
    
    transcription_data = content_response.json()
    
    # Extrai o texto
    combined_phrases = transcription_data.get("combinedRecognizedPhrases", [])
    if not combined_phrases:
        return {"text": "", "parts": []}
    
    full_text = combined_phrases[0].get("display", "")
    
    # Extrai partes detalhadas
    recognized_phrases = transcription_data.get("recognizedPhrases", [])
    parts = []
    
    for phrase in recognized_phrases:
        best = phrase.get("nBest", [{}])[0]
        text = best.get("display", "")
        offset_seconds = phrase.get("offset", 0) / 10000000
        duration_seconds = phrase.get("duration", 0) / 10000000
        
        parts.append({
            "text": text,
            "start_seconds": offset_seconds,
            "end_seconds": offset_seconds + duration_seconds
        })
    
    logging.info(f"Transcricao obtida: {len(full_text)} caracteres, {len(parts)} partes")
    
    return {
        "text": full_text,
        "parts": parts
    }

def _generate_ata(transcript):
    ok, msg = _require_env(["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT"])
    if not ok:
        raise RuntimeError(msg)
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    system_prompt = """Voce e um assistente especializado em criar ATAs (Atas de Reuniao) profissionais.
Analise a transcricao fornecida e crie uma ATA estruturada com:
- Titulo da reuniao
- Data e horario
- Participantes mencionados
- Resumo executivo
- Topicos discutidos com detalhes
- Decisoes tomadas
- Acoes e responsaveis
- Proximos passos

Formato em Markdown, profissional e objetivo."""
    
    user_prompt = f"Transcricao da reuniao:\n\n{transcript}\n\nGere a ATA completa:"
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=2000
    )
    
    ata_content = response.choices[0].message.content
    return ata_content

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        if req.method == "OPTIONS":
            return func.HttpResponse(
                "",
                headers=_cors_headers(),
                status_code=200
            )
        
        # Endpoint para gerar SAS de upload
        if req.params.get("get_upload_url") == "1":
            filename = req.params.get("filename") or "audio.wav"
            container = os.getenv("STORAGE_CONTAINER_INPUT")
            
            if not container:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": "STORAGE_CONTAINER_INPUT ausente"}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=500
                )
            
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            blob_name = f"uploads/{ts}/{filename}"
            
            try:
                sas = _make_upload_sas(container, blob_name, minutes_valid=int(os.getenv("UPLOAD_SAS_MINUTES", "120")))
                return func.HttpResponse(
                    json.dumps({"ok": True, "upload": sas, "marker": MARKER}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=200
                )
            except Exception as e:
                logging.exception("Erro ao gerar SAS")
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": f"Erro ao gerar SAS: {str(e)}"}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=500
                )
        
        # Endpoint para verificar status
        if req.params.get("check_status") == "1":
            transcription_id = req.params.get("transcription_id")
            if not transcription_id:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": "transcription_id ausente"}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )
            
            try:
                status_info = _get_transcription_status(transcription_id)
                return func.HttpResponse(
                    json.dumps({"ok": True, **status_info, "marker": MARKER}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=200
                )
            except Exception as e:
                logging.exception("Erro ao verificar status")
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": str(e)}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=500
                )
        
        # Endpoint para buscar resultado
        if req.params.get("get_result") == "1":
            transcription_id = req.params.get("transcription_id")
            if not transcription_id:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": "transcription_id ausente"}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )
            
            try:
                # Busca transcricao
                transcript_data = _get_transcription_result(transcription_id)
                transcript_text = transcript_data["text"]
                
                if not transcript_text:
                    return func.HttpResponse(
                        json.dumps({"ok": False, "error": "Transcricao vazia"}),
                        mimetype="application/json; charset=utf-8",
                        headers=_cors_headers(),
                        status_code=400
                    )
                
                # Salva transcricao
                container_atas = os.getenv("STORAGE_CONTAINER_ATAS")
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                
                transcript_blob_name = f"transcripts/{ts}/transcript.txt"
                transcript_url = _upload_to_storage(container_atas, transcript_blob_name, transcript_text, "text/plain")
                
                # Gera ATA
                logging.info("Gerando ATA...")
                ata_content = _generate_ata(transcript_text)
                
                ata_blob_name = f"atas/{ts}/ata.md"
                ata_url = _upload_to_storage(container_atas, ata_blob_name, ata_content, "text/markdown")
                
                result = {
                    "ok": True,
                    "transcript": {
                        "text": transcript_text,
                        "url": transcript_url,
                        "parts": transcript_data.get("parts", [])
                    },
                    "ata": {
                        "content": ata_content,
                        "url": ata_url
                    },
                    "marker": MARKER
                }
                
                return func.HttpResponse(
                    json.dumps(result),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=200
                )
                
            except Exception as e:
                logging.exception("Erro ao buscar resultado")
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": str(e)}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=500
                )
        
        # Endpoint principal - Inicia batch transcription
        audio_bytes = None
        blob_url_final = None
        content_type = req.headers.get('Content-Type', 'application/octet-stream')
        
        logging.info(f"Requisicao: Method={req.method}, Content-Type={content_type}")
        
        # Tenta obter audio de URL
        try:
            body = req.get_json()
            blob_url = body.get("blob_url") or body.get("audio_url")
            if blob_url:
                logging.info(f"Usando audio de: {blob_url}")
                blob_url_final = blob_url
        except:
            pass
        
        # Se nao tem audio, pega do body
        if blob_url_final is None:
            audio_bytes = req.get_body()
            if not audio_bytes:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": "Nenhum audio fornecido"}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )
            
            logging.info(f"Audio no body: {len(audio_bytes)} bytes")
            
            # Salva audio
            container_input = os.getenv("STORAGE_CONTAINER_INPUT")
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            audio_blob_name = f"audios/{ts}/audio.wav"
            
            blob_url_final = _upload_to_storage(container_input, audio_blob_name, audio_bytes, "audio/wav")
            logging.info(f"Audio salvo: {blob_url_final}")
        
        # Cria batch transcription job
        logging.info("Iniciando batch transcription...")
        transcription_info = _create_batch_transcription(blob_url_final, language="pt-BR")
        
        result = {
            "ok": True,
            "mode": "batch",
            "transcription_id": transcription_info["transcription_id"],
            "status": transcription_info["status"],
            "audio_url": blob_url_final,
            "marker": MARKER
        }
        
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json; charset=utf-8",
            headers=_cors_headers(),
            status_code=200
        )
    
    except Exception as e:
        logging.exception("Erro no processamento")
        return func.HttpResponse(
            json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()}),
            mimetype="application/json; charset=utf-8",
            headers=_cors_headers(),
            status_code=500
        )
