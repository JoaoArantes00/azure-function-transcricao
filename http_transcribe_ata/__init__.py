import azure.functions as func
import json
import logging
import os
import traceback
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import requests
import azure.cognitiveservices.speech as speechsdk
import openai
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings, BlobSasPermissions, generate_blob_sas

MARKER = "PRODUCTION_VERSION_V4_SIMPLIFIED"

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

def _format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def _transcribe_audio(audio_bytes, language="pt-BR"):
    ok, msg = _require_env(["SPEECH_KEY", "SPEECH_REGION"])
    if not ok:
        raise RuntimeError(msg)
    
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = language
    speech_config.request_word_level_timestamps()
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    
    push_stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    results = []
    done = False
    
    def recognized_cb(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            results.append(evt.result)
    
    def session_stopped_cb(evt):
        nonlocal done
        done = True
    
    recognizer.recognized.connect(recognized_cb)
    recognizer.session_stopped.connect(session_stopped_cb)
    recognizer.canceled.connect(session_stopped_cb)
    
    recognizer.start_continuous_recognition()
    
    chunk_size = 32000
    for i in range(0, len(audio_bytes), chunk_size):
        push_stream.write(audio_bytes[i:i+chunk_size])
    push_stream.close()
    
    import time
    while not done:
        time.sleep(0.1)
    
    recognizer.stop_continuous_recognition()
    
    transcript_parts = []
    
    for result in results:
        try:
            import json as json_lib
            detailed = json_lib.loads(result.json)
            
            if "NBest" in detailed and len(detailed["NBest"]) > 0:
                best = detailed["NBest"][0]
                text = best.get("Display", result.text)
                offset_ticks = result.offset
                duration_ticks = result.duration
                
                offset_seconds = offset_ticks / 10000000
                duration_seconds = duration_ticks / 10000000
                
                timestamp_start = _format_timestamp(offset_seconds)
                timestamp_end = _format_timestamp(offset_seconds + duration_seconds)
                
                transcript_parts.append({
                    "text": text,
                    "start": timestamp_start,
                    "end": timestamp_end,
                    "start_seconds": offset_seconds,
                    "end_seconds": offset_seconds + duration_seconds
                })
        except:
            text = result.text
            transcript_parts.append({
                "text": text,
                "start": "00:00",
                "end": "00:00"
            })
    
    full_text = " ".join([p["text"] for p in transcript_parts])
    
    return {
        "text": full_text,
        "parts": transcript_parts
    }

def _generate_ata(transcript):
    ok, msg = _require_env(["OPENAI_API_KEY", "OPENAI_ENDPOINT"])
    if not ok:
        raise RuntimeError(msg)
    
    openai.api_type = "azure"
    openai.api_base = os.getenv("OPENAI_ENDPOINT")
    openai.api_version = "2024-02-15-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-4")
    
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
    
    response = openai.ChatCompletion.create(
        engine=deployment_name,
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
        
        audio_bytes = None
        blob_url_final = None
        
        try:
            body = req.get_json()
            blob_url = body.get("blob_url")
            if blob_url:
                logging.info(f"Download de audio de: {blob_url}")
                audio_bytes = _download_to_bytes(blob_url)
                blob_url_final = blob_url
        except:
            pass
        
        if audio_bytes is None:
            audio_bytes = req.get_body()
            if not audio_bytes:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": "Nenhum audio fornecido"}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )
            
            container_input = os.getenv("STORAGE_CONTAINER_INPUT")
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            audio_blob_name = f"audios/{ts}/audio.wav"
            blob_url_final = _upload_to_storage(container_input, audio_blob_name, audio_bytes, "audio/wav")
        
        logging.info("Iniciando transcricao...")
        transcript_data = _transcribe_audio(audio_bytes, language="pt-BR")
        transcript_text = transcript_data["text"]
        transcript_parts = transcript_data.get("parts", [])
        
        container_atas = os.getenv("STORAGE_CONTAINER_ATAS")
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        
        transcript_blob_name = f"transcripts/{ts}/transcript.txt"
        transcript_url = _upload_to_storage(container_atas, transcript_blob_name, transcript_text, "text/plain")
        
        logging.info("Gerando ATA...")
        ata_content = _generate_ata(transcript_text)
        
        ata_blob_name = f"atas/{ts}/ata.md"
        ata_url = _upload_to_storage(container_atas, ata_blob_name, ata_content, "text/markdown")
        
        result = {
            "ok": True,
            "audio_url": blob_url_final,
            "transcript": {
                "text": transcript_text,
                "url": transcript_url,
                "parts": transcript_parts
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
        logging.exception("Erro no processamento")
        return func.HttpResponse(
            json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()}),
            mimetype="application/json; charset=utf-8",
            headers=_cors_headers(),
            status_code=500
        )
