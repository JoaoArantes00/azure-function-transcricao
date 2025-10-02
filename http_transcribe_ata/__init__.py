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
import struct
import io

MARKER = "PRODUCTION_VERSION_V10_NO_FFMPEG"

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

def _validate_wav_format(audio_bytes):
    """
    Valida se o audio e um WAV valido e retorna informacoes do formato
    """
    if len(audio_bytes) < 44:
        return False, "Arquivo muito pequeno para ser um WAV valido"
    
    # Verifica header RIFF
    if audio_bytes[:4] != b'RIFF':
        return False, "Nao e um arquivo WAV (falta header RIFF)"
    
    # Verifica WAVE
    if audio_bytes[8:12] != b'WAVE':
        return False, "Nao e um arquivo WAV (falta marker WAVE)"
    
    try:
        # Le informacoes do formato
        fmt_pos = audio_bytes.find(b'fmt ')
        if fmt_pos == -1:
            return False, "Chunk 'fmt' nao encontrado"
        
        # Le dados do formato (offset do fmt + 8 bytes)
        fmt_data_start = fmt_pos + 8
        
        # AudioFormat (2 bytes) - 1 = PCM
        audio_format = struct.unpack('<H', audio_bytes[fmt_data_start:fmt_data_start+2])[0]
        
        # NumChannels (2 bytes)
        num_channels = struct.unpack('<H', audio_bytes[fmt_data_start+2:fmt_data_start+4])[0]
        
        # SampleRate (4 bytes)
        sample_rate = struct.unpack('<I', audio_bytes[fmt_data_start+4:fmt_data_start+8])[0]
        
        # BitsPerSample (2 bytes) - offset 14 do fmt data
        bits_per_sample = struct.unpack('<H', audio_bytes[fmt_data_start+14:fmt_data_start+16])[0]
        
        info = {
            'format': 'PCM' if audio_format == 1 else f'Unknown ({audio_format})',
            'channels': num_channels,
            'sample_rate': sample_rate,
            'bits_per_sample': bits_per_sample,
            'is_pcm': audio_format == 1
        }
        
        logging.info(f"WAV validado: {info}")
        
        # Verifica se e um formato aceitavel
        if not info['is_pcm']:
            return False, f"Formato WAV nao suportado: {info['format']} (apenas PCM e aceito)"
        
        if info['bits_per_sample'] not in [16, 32]:
            return False, f"Bits por amostra nao suportado: {info['bits_per_sample']} (apenas 16 ou 32 bits)"
        
        return True, info
        
    except Exception as e:
        logging.error(f"Erro ao validar WAV: {e}")
        return False, f"Erro ao analisar header WAV: {str(e)}"

def _transcribe_audio(audio_bytes, content_type=None, language="pt-BR"):
    ok, msg = _require_env(["SPEECH_KEY", "SPEECH_REGION"])
    if not ok:
        raise RuntimeError(msg)
    
    audio_size_mb = len(audio_bytes) / (1024 * 1024)
    logging.info(f"=== TRANSCRICAO INICIADA ===")
    logging.info(f"Tamanho: {len(audio_bytes)} bytes ({audio_size_mb:.2f} MB)")
    logging.info(f"Content-Type: {content_type}")
    logging.info(f"Primeiros 20 bytes (hex): {audio_bytes[:20].hex()}")
    
    # Valida se e WAV
    is_valid, result = _validate_wav_format(audio_bytes)
    
    if not is_valid:
        error_msg = f"Formato de audio invalido: {result}"
        logging.error(error_msg)
        logging.error("IMPORTANTE: O frontend deve converter o audio para WAV PCM antes de enviar")
        raise RuntimeError(error_msg)
    
    wav_info = result
    logging.info(f"Audio WAV validado com sucesso: {wav_info}")
    
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = language
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    
    # Salva WAV em arquivo temporario
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    logging.info(f"Arquivo temporario WAV criado: {tmp_path}")
    
    try:
        audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        results = []
        done = False
        error_details = None
        
        def recognized_cb(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                results.append(evt.result)
                logging.info(f"Reconhecido: {evt.result.text[:100]}...")
        
        def canceled_cb(evt):
            nonlocal error_details
            if evt.reason == speechsdk.CancellationReason.Error:
                error_details = f"ErrorCode: {evt.error_code}, ErrorDetails: {evt.error_details}"
                logging.error(f"Cancelado: {error_details}")
        
        def session_stopped_cb(evt):
            nonlocal done
            done = True
            logging.info("Sessao finalizada")
        
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
            raise RuntimeError(f"Erro no Speech SDK: {error_details}")
        
        if len(results) == 0:
            logging.warning("Nenhum resultado de transcricao - audio pode estar vazio ou sem fala")
            return {"text": "", "parts": []}
        
        logging.info(f"Transcricao completa: {len(results)} segmentos")
        
        transcript_parts = []
        for result in results:
            try:
                import json as json_lib
                detailed = json_lib.loads(result.json)
                if "NBest" in detailed and len(detailed["NBest"]) > 0:
                    best = detailed["NBest"][0]
                    text = best.get("Display", result.text)
                    offset_seconds = result.offset / 10000000
                    duration_seconds = result.duration / 10000000
                    transcript_parts.append({
                        "text": text,
                        "start": _format_timestamp(offset_seconds),
                        "end": _format_timestamp(offset_seconds + duration_seconds),
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
        logging.info(f"Texto final: {len(full_text)} caracteres")
        
        return {"text": full_text, "parts": transcript_parts, "wav_info": wav_info}
    
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
        
        logging.info(f"Requisicao: Method={req.method}, Content-Type={content_type}")
        
        # Tenta obter audio de URL (fluxo SAS)
        try:
            body = req.get_json()
            blob_url = body.get("blob_url") or body.get("audio_url")
            if blob_url:
                logging.info(f"Download de: {blob_url}")
                audio_bytes = _download_to_bytes(blob_url)
                blob_url_final = blob_url
        except:
            pass
        
        # Se nao tem audio, pega do body
        if audio_bytes is None:
            audio_bytes = req.get_body()
            if not audio_bytes:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": "Nenhum audio fornecido"}),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )
            
            logging.info(f"Audio no body: {len(audio_bytes)} bytes")
            
            # Salva audio original
            container_input = os.getenv("STORAGE_CONTAINER_INPUT")
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            audio_blob_name = f"audios/{ts}/original.wav"
            
            blob_url_final = _upload_to_storage(container_input, audio_blob_name, audio_bytes, "audio/wav")
            logging.info(f"Audio salvo: {blob_url_final}")
        
        # Transcreve
        logging.info("Iniciando transcricao...")
        transcript_data = _transcribe_audio(audio_bytes, content_type, language="pt-BR")
        transcript_text = transcript_data["text"]
        transcript_parts = transcript_data.get("parts", [])
        wav_info = transcript_data.get("wav_info", {})
        
        if not transcript_text:
            logging.warning("Transcricao vazia")
            return func.HttpResponse(
                json.dumps({"ok": False, "error": "Nenhuma fala detectada no audio"}),
                mimetype="application/json; charset=utf-8",
                headers=_cors_headers(),
                status_code=400
            )
        
        # Salva transcript
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
            "audio_info": wav_info,
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
