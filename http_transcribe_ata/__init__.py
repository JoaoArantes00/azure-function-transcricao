import azure.functions as func
import json, logging, os, traceback
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, quote
import requests
import azure.cognitiveservices.speech as speechsdk
import openai
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobServiceClient, ContentSettings, BlobSasPermissions, generate_blob_sas
)
import threading
import unicodedata

MARKER = "PRODUCTION_VERSION_V2_TIMESTAMPS_UTF8+SAS_UPLOAD"

# =============================
# Headers CORS
# =============================

def _cors_headers():
    """Headers CORS para todas as respostas"""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, x-functions-key",
        "Access-Control-Max-Age": "3600"
    }

# =============================
# Utilidades de texto / UTF-8
# =============================

def _normalize_text(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize('NFC', text)
    corrections = {
        'Ã§': 'ç',  'Ã‡': 'Ç', 'Ã£': 'ã',  'Ã': 'Ã',
        'Ã¡': 'á',  'Ã©': 'é',  'Ã‰': 'É', 'Ã³': 'ó',
        'Ã"': 'Ó', 'Ãº': 'ú',  'Ãš': 'Ú', 'Ã¢': 'â',
        'Ã‚': 'Â', 'Ãª': 'ê',  'ÃŠ': 'Ê', 'Ã´': 'ô',
        'Ã"': 'Ô', 'Ã ': 'à',  'Ã€': 'À', 'Ã¨': 'è',
        'Ãˆ': 'È', 'Ã¬': 'ì',  'ÃŒ': 'Ì', 'Ã²': 'ò',
        'Ã'': 'Ò', 'Ã¹': 'ù',  'Ã™': 'Ù', 'Ã±': 'ñ',
        'Ã'': 'Ñ', 'Ã¼': 'ü',  'Ãœ': 'Ü', 'ÃƒÂ': '',
        'â€™': "'", 'â€œ': '"', 'â€\x9d': '"', 'â€"': '–', 'â€¦': '…',
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')
    return text.strip()

# =============================
# Helpers de Storage / SAS
# =============================

def _blob_service() -> BlobServiceClient:
    account = os.getenv("STORAGE_ACCOUNT_NAME")
    cred = DefaultAzureCredential()
    return BlobServiceClient(account_url=f"https://{account}.blob.core.windows.net", credential=cred)

def _make_upload_sas(container: str, blob_name: str, minutes_valid: int = 120) -> dict:
    """
    Cria SAS de upload (user delegation SAS) para PUT grande direto do front.
    Requer MSI com 'Storage Blob Data Contributor' na conta.
    """
    bsc = _blob_service()
    account = os.getenv("STORAGE_ACCOUNT_NAME")

    # pega user delegation key
    udk = bsc.get_user_delegation_key(
        key_start_time=datetime.now(tz=timezone.utc) - timedelta(minutes=5),
        key_expiry_time=datetime.now(tz=timezone.utc) + timedelta(minutes=minutes_valid)
    )

    sas = generate_blob_sas(
        account_name=account,
        container_name=container,
        blob_name=blob_name,
        user_delegation_key=udk,
        permission=BlobSasPermissions(create=True, write=True, add=True, read=True),
        expiry=datetime.now(tz=timezone.utc) + timedelta(minutes=minutes_valid),
        start=datetime.now(tz=timezone.utc) - timedelta(minutes=5),
        content_type=None
    )
    blob_url = f"https://{account}.blob.core.windows.net/{container}/{quote(blob_name)}"
    put_url = f"{blob_url}?{sas}"
    return {"blob_url": blob_url, "put_url": put_url, "sas_valid_minutes": minutes_valid}

def _upload_to_storage(container: str, blob_name: str, content, content_type: str = "text/plain"):
    """
    Upload com credencial do servidor (para salvar transcript/ata e áudios pequenos).
    """
    try:
        bsc = _blob_service()
        container_client = bsc.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)

        if isinstance(content, bytes):
            data = content
        else:
            normalized = _normalize_text(content) if isinstance(content, str) else content
            data = normalized.encode("utf-8")

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

def _download_to_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

# =============================
# Utilidades diversas
# =============================

def _require_env(keys):
    missing = [k for k in keys if not os.getenv(k)]
    return (len(missing) == 0, f"Variáveis não configuradas: {', '.join(missing)}" if missing else "")

def _format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# =============================
# Speech / Transcrição
# =============================

def _speech_transcribe_from_bytes(audio_bytes: bytes, language: str = "pt-BR") -> dict:
    print(f"_speech_transcribe_from_bytes chamada com {len(audio_bytes)} bytes, language={language}")

    segments_with_time = []
    full_text = ""
    recognition_done = threading.Event()

    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("SPEECH_KEY"),
            region=os.getenv("SPEECH_REGION")
        )
        speech_config.speech_recognition_language = "pt-BR"
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_RecoLanguage, "pt-BR")
        speech_config.output_format = speechsdk.OutputFormat.Detailed
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "5000")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")
        speech_config.request_word_level_timestamps()
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestDetailedResultTrueFalse, "true")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestProfanityFilterTrueFalse, "false")

        # formato generoso, o serviço reamostra se necessário
        wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second=48000, bits_per_sample=16, channels=2)
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
        push_stream.write(audio_bytes)
        push_stream.close()

        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        def recognized_handler(evt):
            nonlocal full_text
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                start_time = evt.result.offset / 10_000_000
                duration = evt.result.duration / 10_000_000
                end_time = start_time + duration
                raw_text = evt.result.text.strip()
                normalized_text = _normalize_text(raw_text)
                segment = {
                    "start_seconds": round(start_time, 2),
                    "end_seconds": round(end_time, 2),
                    "start_time": _format_timestamp(start_time),
                    "end_time": _format_timestamp(end_time),
                    "text": normalized_text,
                    "raw_text": raw_text
                }
                segments_with_time.append(segment)
                full_text += normalized_text + " "

        def session_stopped_handler(_): recognition_done.set()
        def canceled_handler(evt):
            print(f"Reconhecimento cancelado: {evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                print(f"Detalhes do erro: {evt.error_details}")
            recognition_done.set()

        recognizer.recognized.connect(recognized_handler)
        recognizer.session_stopped.connect(session_stopped_handler)
        recognizer.canceled.connect(canceled_handler)

        recognizer.start_continuous_recognition()

        # aguarda até 5 min (áudios grandes vindos do blob)
        timeout_sec = int(os.getenv("SPEECH_RECO_TIMEOUT_SEC", "300"))
        recognition_done.wait(timeout=timeout_sec)
        recognizer.stop_continuous_recognition()

        full_text = _normalize_text(full_text.strip())

        if segments_with_time:
            return {
                "ok": True,
                "text": full_text,
                "segments": segments_with_time,
                "segment_count": len(segments_with_time),
                "reason": "RecognizedSpeechWithTimestamps"
            }
        else:
            return {"ok": True, "text": "", "segments": [], "segment_count": 0, "reason": "NoMatch"}

    except Exception as e:
        logging.exception("Erro na transcrição de áudio")
        raise RuntimeError(f"Erro na transcrição: {str(e)}")

# =============================
# Geração da ATA (OpenAI)
# =============================

def _generate_meeting_minutes_with_timestamps(transcript_data: dict) -> str:
    try:
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-05-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        full_text = transcript_data.get("text", "")
        segments = transcript_data.get("segments", [])

        formatted_transcript = ""
        for s in segments:
            formatted_transcript += f"[{s['start_time']}] {s['text']}\n"

        system_prompt = (
            "Você é um assistente especializado em elaborar ATAs de reunião em português do Brasil. "
            "Você receberá uma transcrição com marcações de tempo (timestamps) no formato [MM:SS]. "
            "Use acentuação correta. Produza uma ATA com: Participantes, Objetivo, Pontos, Decisões, "
            "Pendências/Ações, Próximos passos e Anexo com a Transcrição (timestamps)."
        )
        user_prompt = (
            f"TRANSCRIÇÃO COM TIMESTAMPS:\n{formatted_transcript}\n\n"
            f"TEXTO COMPLETO: {full_text}\n\n"
            f"Gere a ATA no formato solicitado."
        )

        resp = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        ata = resp.choices[0].message.content
        return _normalize_text(ata)
    except Exception as e:
        logging.exception("Erro na geração da ATA")
        raise RuntimeError(f"Erro na geração da ATA: {str(e)}")

# =============================
# Função principal (HttpTrigger)
# =============================

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Responde a preflight OPTIONS (CORS)
        if req.method == "OPTIONS":
            return func.HttpResponse(
                "",
                headers=_cors_headers(),
                status_code=200
            )

        # Ping rápido
        if req.params.get("ping") == "1":
            return func.HttpResponse(
                json.dumps({
                    "ok": True, "status": "pong", "marker": MARKER,
                    "message": "Transcrição com timestamps/UTF-8 + SAS de upload ok",
                    "features": ["audio_transcription","timestamp_tracking","utf8_normalization","ata_generation","blob_sas_upload"]
                }, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                headers=_cors_headers(),
                status_code=200
            )

        # ========= Modo 0: gerar SAS de upload para arquivos grandes =========
        if req.params.get("get_upload_url") == "1":
            filename = req.params.get("filename") or "audio.wav"
            container = os.getenv("STORAGE_CONTAINER_INPUT")
            if not container:
                return func.HttpResponse(
                    json.dumps({"ok":False,"error":"STORAGE_CONTAINER_INPUT ausente"}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=500
                )
            # opcional: prefixo por data
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            blob_name = f"uploads/{ts}/{filename}"
            try:
                sas = _make_upload_sas(container, blob_name, minutes_valid=int(os.getenv("UPLOAD_SAS_MINUTES","120")))
                return func.HttpResponse(
                    json.dumps({"ok":True, "upload": sas, "marker": MARKER}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=200
                )
            except Exception as e:
                return func.HttpResponse(
                    json.dumps({"ok":False,"error":str(e)}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=500
                )

        # ========= Verificação de variáveis obrigatórias =========
        required = [
            "SPEECH_KEY","SPEECH_REGION",
            "AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_KEY","AZURE_OPENAI_DEPLOYMENT",
            "STORAGE_ACCOUNT_NAME","STORAGE_CONTAINER_INPUT","STORAGE_CONTAINER_TRANSCRIPTS","STORAGE_CONTAINER_ATAS"
        ]
        ok_env, env_msg = _require_env(required)
        if not ok_env:
            return func.HttpResponse(
                json.dumps({"ok":False,"error":env_msg}, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                headers=_cors_headers(),
                status_code=500
            )

        # ========= Entrada: JSON (audio_url) ou binário pequeno =========
        content_type = (req.headers.get("content-type") or "").lower()
        max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "50"))  # limite do app para corpo binário
        audio_bytes = None
        audio_filename = None

        if content_type.startswith("application/json"):
            try:
                body = req.get_json()
            except Exception as e:
                return func.HttpResponse(
                    json.dumps({"ok":False,"error":f"JSON inválido: {str(e)}"}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )
            audio_url = (body or {}).get("audio_url")
            if not audio_url:
                return func.HttpResponse(
                    json.dumps({"ok":False,"error":"Campo 'audio_url' é obrigatório"}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )
            try:
                audio_bytes = _download_to_bytes(audio_url)
                audio_filename = os.path.basename(urlparse(audio_url).path) or "audio.wav"
                logger.info(f"Baixado do blob: {len(audio_bytes)} bytes")
            except Exception as e:
                return func.HttpResponse(
                    json.dumps({"ok":False,"error":f"Erro ao baixar áudio: {str(e)}"}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )

        elif content_type.startswith("audio/") or content_type.startswith("application/octet-stream"):
            # Tenta checar Content-Length antes de ler
            try:
                cl = int(req.headers.get("content-length","0"))
                if cl > max_upload_mb * 1024 * 1024:
                    return func.HttpResponse(
                        json.dumps({"ok":False,
                                    "error": f"Arquivo acima de {max_upload_mb}MB. Use fluxo grande: GET ?get_upload_url=1 e depois POST com audio_url."},
                                   ensure_ascii=False),
                        mimetype="application/json; charset=utf-8",
                        headers=_cors_headers(),
                        status_code=413
                    )
            except: pass

            audio_bytes = req.get_body()
            if not audio_bytes:
                return func.HttpResponse(
                    json.dumps({"ok":False,"error":"Corpo da requisição vazio"}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )
            if len(audio_bytes) > max_upload_mb * 1024 * 1024:
                return func.HttpResponse(
                    json.dumps({"ok":False,
                                "error": f"Arquivo acima de {max_upload_mb}MB. Use fluxo grande: GET ?get_upload_url=1 e depois POST com audio_url."},
                               ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=413
                )
            audio_filename = "audio_upload.wav"
            logger.info(f"Recebido binário pequeno: {len(audio_bytes)} bytes")

        else:
            return func.HttpResponse(
                json.dumps({"ok":False,
                            "error":"Content-Type não suportado. Use application/json com audio_url OU envie áudio binário pequeno."},
                           ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                headers=_cors_headers(),
                status_code=400
            )

        # ========= (Opcional) Salva cópia do áudio de entrada =========
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base_name = os.path.splitext(audio_filename)[0]
        try:
            original_audio_url = _upload_to_storage(
                container=os.getenv("STORAGE_CONTAINER_INPUT"),
                blob_name=f"{base_name}-{timestamp}.wav",
                content=audio_bytes,
                content_type="audio/wav"
            )
        except Exception as e:
            logger.warning(f"Erro ao salvar áudio original: {e}")
            original_audio_url = None

        # ========= Transcrição =========
        try:
            tr = _speech_transcribe_from_bytes(audio_bytes)
            transcript_text = tr.get("text","")
            reason = tr.get("reason","")
            segments = tr.get("segments",[])
            seg_count = tr.get("segment_count",0)

            logger.info(f"Transcrição: {len(transcript_text)} chars, {seg_count} segmentos, reason={reason}")
            if not transcript_text.strip():
                return func.HttpResponse(
                    json.dumps({"ok":False,"error":"Nenhum texto foi transcrito do áudio","reason":reason}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    headers=_cors_headers(),
                    status_code=400
                )
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"ok":False,"error":f"Erro na transcrição: {str(e)}"}, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                headers=_cors_headers(),
                status_code=500
            )

        # ========= Persistência da transcrição =========
        transcript_data = {
            "timestamp": timestamp,
            "audio_filename": audio_filename,
            "reason": reason,
            "text": transcript_text,
            "char_count": len(transcript_text),
            "segments": segments,
            "segment_count": seg_count,
            "total_duration": segments[-1]["end_seconds"] if segments else 0,
            "encoding": "utf-8"
        }
        try:
            transcript_url = _upload_to_storage(
                container=os.getenv("STORAGE_CONTAINER_TRANSCRIPTS"),
                blob_name=f"{base_name}-transcript-{timestamp}.json",
                content=json.dumps(transcript_data, ensure_ascii=False, indent=2),
                content_type="application/json"
            )
        except Exception as e:
            logger.warning(f"Erro ao salvar transcrição: {e}")
            transcript_url = None

        # ========= ATA =========
        ata_content, ata_url, ata_error = None, None, None
        try:
            ata_content = _generate_meeting_minutes_with_timestamps(transcript_data)
            ata_url = _upload_to_storage(
                container=os.getenv("STORAGE_CONTAINER_ATAS"),
                blob_name=f"{base_name}-ata-{timestamp}.md",
                content=ata_content,
                content_type="text/markdown"
            )
        except Exception as e:
            ata_error = str(e)
            logger.error(f"Erro na geração/salvamento da ATA: {e}")

        # ========= Resposta =========
        resp = {
            "ok": True,
            "marker": MARKER,
            "timestamp": timestamp,
            "processing": {
                "audio_size_bytes": len(audio_bytes),
                "transcript_chars": len(transcript_text),
                "transcription_reason": reason,
                "segment_count": seg_count,
                "total_duration_seconds": transcript_data.get("total_duration", 0),
                "ata_generated": ata_content is not None,
                "ata_chars": len(ata_content) if ata_content else 0,
                "encoding": "utf-8"
            },
            "preview": {
                "transcript": transcript_text[:300] + ("..." if len(transcript_text) > 300 else ""),
                "first_segments": segments[:3] if segments else [],
                "ata": ata_content[:400] + ("..." if ata_content and len(ata_content) > 400 else "") if ata_content else None
            },
            "urls": {
                "original_audio": original_audio_url,
                "transcript": transcript_url,
                "ata": ata_url
            },
            "errors": {
                "ata_generation": ata_error
            }
        }
        return func.HttpResponse(
            json.dumps(resp, ensure_ascii=False, indent=2),
            mimetype="application/json; charset=utf-8",
            headers=_cors_headers(),
            status_code=200
        )

    except Exception as e:
        logger.exception("Erro não tratado")
        debug_mode = os.getenv("DEBUG") == "1"
        err = {"ok": False, "error": str(e), "marker": MARKER}
        if debug_mode: err["stack_trace"] = traceback.format_exc()
        return func.HttpResponse(
            json.dumps(err, ensure_ascii=False),
            mimetype="application/json; charset=utf-8",
            headers=_cors_headers(),
            status_code=500
        )
