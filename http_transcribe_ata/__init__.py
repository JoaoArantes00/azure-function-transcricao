import azure.functions as func
import json
import logging
import os
import traceback
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import requests
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings, BlobSasPermissions, generate_blob_sas
import tempfile
import subprocess

MARKER = "PRODUCTION_VERSION_V8_DEBUG"

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

def _detect_audio_format(audio_bytes):
    """Detecta o formato do audio pelos magic bytes"""
    if len(audio_bytes) < 12:
        return "unknown"
    
    header = audio_bytes[:12]
    
    # WAV: "RIFF....WAVE"
    if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
        return "wav"
    
    # MP3: ID3 tag ou frame sync
    if header[:3] == b'ID3' or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return "mp3"
    
    # AAC: ADTS header
    if header[0] == 0xFF and (header[1] & 0xF6) == 0xF0:
        return "aac"
    
    # M4A/MP4: ftyp box
    if header[4:8] == b'ftyp':
        return "m4a"
    
    # OGG: "OggS"
    if header[:4] == b'OggS':
        return "ogg"
    
    return "unknown"

def _convert_to_wav(input_bytes, input_format="aac"):
    """Converte audio de qualquer formato para WAV PCM 16kHz mono usando FFmpeg"""
    logging.info(f"Convertendo {input_format.upper()} para WAV ({len(input_bytes)} bytes)")
    
    input_temp = tempfile.NamedTemporaryFile(suffix=f'.{input_format}', delete=False)
    output_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    
    try:
        input_temp.write(input_bytes)
        input_temp.close()
        output_temp.close()
        
        cmd = [
            'ffmpeg',
            '-i', input_temp.name,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-y',
            output_temp.name
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            logging.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg falhou: {result.stderr}")
        
        with open(output_temp.name, 'rb') as f:
            wav_bytes = f.read()
        
        logging.info(f"Conversao concluida: {len(wav_bytes)} bytes WAV")
        return wav_bytes
        
    finally:
        try:
            os.unlink(input_temp.name)
            os.unlink(output_temp.name)
        except:
            pass

def _transcribe_audio(audio_bytes, content_type=None, language="pt-BR"):
    ok, msg = _require_env(["SPEECH_KEY", "SPEECH_REGION"])
    if not ok:
        raise RuntimeError(msg)
    
    audio_size_mb = len(audio_bytes) / (1024 * 1024)
    logging.info(f"=== INICIO TRANSCRICAO ===")
    logging.info(f"Tamanho: {len(audio_bytes)} bytes ({audio_size_mb:.2f} MB)")
    logging.info(f"Content-Type recebido: {content_type}")
    
    # DEBUG: Salva arquivo original para analise
    debug_path = f"/tmp/debug_original_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.bin"
    try:
        with open(debug_path, 'wb') as f:
            f.write(audio_bytes)
        logging.info(f"DEBUG: Arquivo original salvo em {debug_path}")
    except Exception as e:
        logging.warning(f"Nao foi possivel salvar debug: {e}")
    
    # Detecta formato real pelos bytes
    detected_format = _detect_audio_format(audio_bytes)
    logging.info(f"Formato detectado pelos magic bytes: {detected_format}")
    logging.info(f"Primeiros 20 bytes (hex): {audio_bytes[:20].hex()}")
    
    # Verifica se precisa converter
    needs_conversion = detected_format not in ['wav', 'mp3', 'unknown']
    
    if content_type:
        ct_lower = content_type.lower()
        if 'aac' in ct_lower or 'm4a' in ct_lower or 'ogg' in ct_lower or 'wma' in ct_lower:
            needs_conversion = True
            logging.info(f"Conversao forcada pelo Content-Type: {content_type}")
    
    if needs_conversion and detected_format != 'unknown':
        logging.info(f"Iniciando conversao de {detected_format} para WAV...")
        try:
            audio_bytes = _convert_to_wav(audio_bytes, detected_format)
            logging.info(f"Conversao concluida: {len(audio_bytes)} bytes")
            
            # DEBUG: Salva arquivo convertido
            debug_conv_path = f"/tmp/debug_converted_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.wav"
            try:
                with open(debug_conv_path, 'wb') as f:
                    f.write(audio_bytes)
                logging.info(f"DEBUG: Arquivo convertido salvo em {debug_conv_path}")
            except:
                pass
        except Exception as e:
            logging.error(f"Erro na conversao: {e}")
            logging.info("Tentando transcrever arquivo original...")
    
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = language
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    
    # Salva em arquivo temporario
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    logging.info(f"Arquivo temporario criado: {tmp_path}")
    
    try:
        audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        results = []
        done = False
        error_details = None
        
        def recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                results.append(evt.result)
                logging.info(f"Reconhecido: {evt.result.text[:50]}...")
        
        def canceled_cb(evt):
            nonlocal error_details
            if evt.reason == speechsdk.CancellationReason.Error:
                error_details = f"ErrorCode: {evt.error_code}, ErrorDetails: {evt.error_details}"
                logging.error(f"Reconhecimento cancelado: {error_details}")
        
        def session_stopped_cb(evt):
            nonlocal done
            done = True
        
        recognizer.recognized.connect(recognized_cb)
        recognizer.canceled.connect(canceled_cb)
        recognizer.session_stopped.connect(session_stopped_cb)
        
        logging.info("Iniciando reconhecimento continuo...")
        recognizer.start_continuous_recognition()
        
        import time
        timeout = 600
        start = time.time()
        while not done and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        recognizer.stop_continuous_recognition()
        
        if error_details:
            raise RuntimeError(f"Erro no reconhecimento de fala: {error_details}")
        
        logging.info(f"Transcricao finalizada. Total de resultados: {len(results)}")
        
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
                transcript_parts.append({
                    "text": result.text,
                    "start": "00:00",
                    "end": "00:00"
                })
        
        full_text = " ".join([p["text"] for p in transcript_parts])
        logging.info(f"Transcricao completa: {len(full_text)} caracteres")
        
        return {"text": full_text, "parts": transcript_parts}
    
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

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
        content_type = req.headers.get('Content-Type', 'application/octet-stream')
        
        logging.info(f"Requisicao recebida - Method: {req.method}, Content-Type: {content_type}")
        
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
            
            logging.info(f"Audio recebido diretamente no body: {len(audio_bytes)} bytes")
            
            container_input = os.getenv("STORAGE_CONTAINER_INPUT")
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            audio_blob_name = f"audios/{ts}/audio_original.bin"
            
            blob_url_final = _upload_to_storage(container_input, audio_blob_name, audio_bytes, content_type)
            logging.info(f"Audio salvo em: {blob_url_final}")
        
        logging.info("Iniciando transcricao...")
        transcript_data = _transcribe_audio(audio_bytes, content_type, language="pt-BR")
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
import azure.functions as func
import json
import logging
import os
import traceback
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import requests
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings, BlobSasPermissions, generate_blob_sas
import tempfile
import subprocess

MARKER = "PRODUCTION_VERSION_V7_FILE_METHOD"

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

def _convert_to_wav(input_bytes, input_format="aac"):
    """
    Converte audio de qualquer formato para WAV PCM 16kHz mono usando FFmpeg
    """
    logging.info(f"Convertendo {input_format.upper()} para WAV ({len(input_bytes)} bytes)")
    
    # Arquivos temporarios
    input_temp = tempfile.NamedTemporaryFile(suffix=f'.{input_format}', delete=False)
    output_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    
    try:
        # Salva input
        input_temp.write(input_bytes)
        input_temp.close()
        output_temp.close()
        
        # Converte com FFmpeg
        cmd = [
            'ffmpeg',
            '-i', input_temp.name,
            '-ar', '16000',      # 16kHz sample rate
            '-ac', '1',          # Mono
            '-c:a', 'pcm_s16le', # PCM 16-bit
            '-y',                # Sobrescrever
            output_temp.name
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos max
        )
        
        if result.returncode != 0:
            logging.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg falhou: {result.stderr}")
        
        # Le o WAV convertido
        with open(output_temp.name, 'rb') as f:
            wav_bytes = f.read()
        
        logging.info(f"Conversao concluida: {len(wav_bytes)} bytes WAV")
        return wav_bytes
        
    finally:
        # Limpa arquivos temporarios
        try:
            os.unlink(input_temp.name)
            os.unlink(output_temp.name)
        except:
            pass

def _transcribe_audio(audio_bytes, content_type=None, language="pt-BR"):
    ok, msg = _require_env(["SPEECH_KEY", "SPEECH_REGION"])
    if not ok:
        raise RuntimeError(msg)
    
    audio_size_mb = len(audio_bytes) / (1024 * 1024)
    logging.info(f"Iniciando transcricao de {len(audio_bytes)} bytes ({audio_size_mb:.2f} MB)")
    
    # Detecta se precisa converter
    needs_conversion = False
    input_format = "wav"
    
    if content_type:
        ct_lower = content_type.lower()
        if 'aac' in ct_lower:
            needs_conversion = True
            input_format = "aac"
        elif 'm4a' in ct_lower:
            needs_conversion = True
            input_format = "m4a"
        elif 'ogg' in ct_lower:
            needs_conversion = True
            input_format = "ogg"
        elif 'wma' in ct_lower:
            needs_conversion = True
            input_format = "wma"
        elif 'mp4' in ct_lower:
            needs_conversion = True
            input_format = "mp4"
    
    # Converte se necessario
    if needs_conversion:
        logging.info(f"Formato {input_format.upper()} detectado - convertendo para WAV...")
        audio_bytes = _convert_to_wav(audio_bytes, input_format)
    
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = language
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    
    # Usar arquivo temporario
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        results = []
        done = False
        
        def recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                results.append(evt.result)
                logging.info(f"Reconhecido: {evt.result.text[:50]}...")
        
        def session_stopped_cb(evt):
            nonlocal done
            done = True
        
        recognizer.recognized.connect(recognized_cb)
        recognizer.session_stopped.connect(session_stopped_cb)
        recognizer.canceled.connect(session_stopped_cb)
        
        recognizer.start_continuous_recognition()
        
        import time
        timeout = 600
        start = time.time()
        while not done and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        recognizer.stop_continuous_recognition()
        
        logging.info(f"Transcricao finalizada. Total de resultados: {len(results)}")
        
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
                transcript_parts.append({
                    "text": result.text,
                    "start": "00:00",
                    "end": "00:00"
                })
        
        full_text = " ".join([p["text"] for p in transcript_parts])
        logging.info(f"Transcricao completa: {len(full_text)} caracteres")
        
        return {"text": full_text, "parts": transcript_parts}
    
    finally:
        # Limpa arquivo temporario
        try:
            os.unlink(tmp_path)
        except:
            pass

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
        content_type = req.headers.get('Content-Type', 'audio/wav')
        
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
            
            # Salva o audio EXATAMENTE como recebeu
            logging.info(f"Salvando audio de {len(audio_bytes)} bytes (Content-Type: {content_type})")
            blob_url_final = _upload_to_storage(container_input, audio_blob_name, audio_bytes, content_type)
            logging.info(f"Audio salvo em: {blob_url_final}")
        
        logging.info("Iniciando transcricao...")
        transcript_data = _transcribe_audio(audio_bytes, content_type, language="pt-BR")
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
