import azure.functions as func
import json, logging, os, traceback, io, wave, math, tempfile
from datetime import datetime
from urllib.parse import urlparse
import requests
import azure.cognitiveservices.speech as speechsdk
import openai
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
import unicodedata
import threading

MARKER = "PRODUCTION_VERSION_V2_TIMESTAMPS_UTF8"

def _normalize_text(text: str) -> str:
    """Normalizar texto para corrigir problemas de codificação"""
    if not text:
        return text
    
    text = unicodedata.normalize('NFC', text)
    corrections = {
        'Ã§': 'ç','Ã‡':'Ç','Ã£':'ã','Ã':'Ã','Ã¡':'á','Ã©':'é','Ã‰':'É','Ã³':'ó','Ã"':'Ó',
        'Ãº':'ú','Ãš':'Ú','Ã¢':'â','Ã‚':'Â','Ãª':'ê','ÃŠ':'Ê','Ã´':'ô','Ã ':'à','Ã€':'À',
        'Ã¨':'è','Ãˆ':'È','Ã¬':'ì','ÃŒ':'Ì','Ã²':'ò',"Ã'":'Ò','Ã¹':'ù','Ã™':'Ù','Ã±':'ñ',
        'Ã¼':'ü','Ãœ':'Ü','ÃªÃ§':'eç','ÃƒÂ':'','â€™':"'",'â€œ':'"','â€':'"','â€"':'–','â€¦':'…',
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')
    return text.strip()

def _download_to_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def _require_env(keys):
    missing = [k for k in keys if not os.getenv(k)]
    return (len(missing) == 0, f"Variáveis não configuradas: {', '.join(missing)}" if missing else "")

def _format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# ---------------------------
# Azure Storage helpers
# ---------------------------
def _blob_service_client():
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    if not storage_account_name:
        raise RuntimeError("STORAGE_ACCOUNT_NAME não configurada")
    credential = DefaultAzureCredential()
    return BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=credential
    )

def _ensure_container(svc: BlobServiceClient, container: str):
    try:
        svc.create_container(container)
    except Exception:
        pass

def _upload_to_storage(container: str, blob_name: str, content, content_type: str = "text/plain"):
    """Upload de conteúdo para Azure Storage com codificação UTF-8"""
    try:
        svc = _blob_service_client()
        _ensure_container(svc, container)
        blob_client = svc.get_blob_client(container, blob_name)

        if isinstance(content, bytes):
            data_to_upload = content
        else:
            normalized_content = _normalize_text(content) if isinstance(content, str) else content
            data_to_upload = normalized_content.encode("utf-8")

        if content_type.startswith("text/") or content_type.startswith("application/json"):
            content_settings = ContentSettings(content_type=f"{content_type}; charset=utf-8")
        else:
            content_settings = ContentSettings(content_type=content_type)

        blob_client.upload_blob(
            data=data_to_upload,
            overwrite=True,
            content_settings=content_settings
        )

        storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
        return f"https://{storage_account_name}.blob.core.windows.net/{container}/{blob_name}"

    except Exception as e:
        logging.exception("Erro no upload para storage")
        raise RuntimeError(f"Erro no upload: {str(e)}")

# ---------------------------
# Speech (usa arquivo temporário; sem propriedades problemáticas)
# ---------------------------
def _wav_duration_seconds(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return 0.0

def _speech_transcribe_from_bytes(audio_bytes: bytes, language: str = "pt-BR") -> dict:
    """
    Transcrever áudio com Azure Speech + timestamps.
    Estratégia: salvar WAV em arquivo temporário e usar AudioConfig(filename=...),
    deixando o SDK ler o cabeçalho e o formato correto.
    """
    logging.info(f"_speech_transcribe_from_bytes: {len(audio_bytes)} bytes, lang={language}")

    segments_with_time = []
    full_text = ""
    recognition_done = threading.Event()

    try:
        speech_key = os.getenv("SPEECH_KEY")
        speech_region = os.getenv("SPEECH_REGION")
        if not speech_key or not speech_region:
            raise RuntimeError("SPEECH_KEY/SPEECH_REGION não configurados")

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.speech_recognition_language = language or "pt-BR"

        # Resultado detalhado + timestamps (maneira suportada no SDK Python)
        speech_config.output_format = speechsdk.OutputFormat.Detailed
        # A linha abaixo é suficiente; a propriedade RequestDetailedResult* gerou erro e foi removida
        speech_config.set_service_property("format", "detailed", speechsdk.ServicePropertyChannel.UriQueryParameter)
        speech_config.request_word_level_timestamps()
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "5000"
        )
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            os.getenv("SPEECH_INITIAL_SILENCE_MS", "10000")
        )

        # Salva em arquivo temporário para o SDK lidar com o cabeçalho WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        def recognized_handler(evt):
            nonlocal full_text
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                start_time = evt.result.offset / 10_000_000  # 100ns -> s
                duration = evt.result.duration / 10_000_000
                end_time = start_time + duration
                raw_text = (evt.result.text or "").strip()
                normalized_text = _normalize_text(raw_text)
                segments_with_time.append({
                    "start_seconds": round(start_time, 2),
                    "end_seconds": round(end_time, 2),
                    "start_time": _format_timestamp(start_time),
                    "end_time": _format_timestamp(end_time),
                    "text": normalized_text,
                    "raw_text": raw_text
                })
                full_text += normalized_text + " "

        def session_stopped_handler(_):
            recognition_done.set()

        def canceled_handler(evt):
            logging.error(f"Reconhecimento cancelado: {evt.reason} | {getattr(evt, 'error_details', '')}")
            recognition_done.set()

        recognizer.recognized.connect(recognized_handler)
        recognizer.session_stopped.connect(session_stopped_handler)
        recognizer.canceled.connect(canceled_handler)

        recognizer.start_continuous_recognition()

        # Timeout proporcional à duração do arquivo (mín 30s, máx 180s)
        dur = _wav_duration_seconds(audio_bytes) or 0.0
        wait_s = min(max(int(math.ceil(dur * 1.5)), 30), 180)
        if recognition_done.wait(timeout=wait_s):
            logging.info("Reconhecimento concluído")
        else:
            logging.warning("Timeout no reconhecimento")

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

# ---------------------------
# OpenAI (ATA)
# ---------------------------
def _generate_meeting_minutes_with_timestamps(transcript_data: dict) -> str:
    """Gerar ATA usando Azure OpenAI com informações de timestamp e correção UTF-8"""
    try:
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        full_text = transcript_data.get("text", "")
        segments = transcript_data.get("segments", [])
        formatted_transcript = ""
        for seg in segments:
            formatted_transcript += f"[{seg['start_time']}] {seg['text']}\n"

        system_prompt = (
            "Você é um assistente especializado em elaborar ATAs de reunião em português do Brasil.\n"
            "Receberá uma transcrição com timestamps [MM:SS]. Gere uma ATA profissional em Markdown com:\n"
            "Participantes, Contexto/Objetivo, Pontos Discutidos, Decisões, Pendências/Ações (com responsáveis quando possível), Próximos Passos, "
            "e Anexo: Transcrição com Timestamps. Seja fiel ao conteúdo, conciso e com acentuação correta."
        )
        user_prompt = (
            f"TRANSCRIÇÃO COM TIMESTAMPS:\n{formatted_transcript}\n\n"
            f"TEXTO COMPLETO: {full_text}\n\n"
            f"Gere a ATA seguindo o formato solicitado."
        )

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        ata_content = response.choices[0].message.content
        return _normalize_text(ata_content)
    except Exception as e:
        logging.exception("Erro na geração da ATA")
        raise RuntimeError(f"Erro na geração da ATA: {str(e)}")

# ---------------------------
# HTTP Trigger
# ---------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        if req.params.get("ping") == "1":
            return func.HttpResponse(
                json.dumps({
                    "ok": True, "status": "pong", "marker": MARKER,
                    "message": "Serviço de transcrição com timestamps e correção UTF-8 funcionando!",
                    "features": ["audio_transcription","timestamp_tracking","utf8_normalization","ata_generation","azure_storage"]
                }, ensure_ascii=False),
                mimetype="application/json; charset=utf-8", status_code=200
            )

        required_vars = [
            "SPEECH_KEY","SPEECH_REGION",
            "AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_KEY","AZURE_OPENAI_DEPLOYMENT",
            "STORAGE_ACCOUNT_NAME","STORAGE_CONTAINER_INPUT","STORAGE_CONTAINER_TRANSCRIPTS","STORAGE_CONTAINER_ATAS"
        ]
        env_ok, env_error = _require_env(required_vars)
        if not env_ok:
            return func.HttpResponse(
                json.dumps({"ok": False, "error": env_error}, ensure_ascii=False),
                mimetype="application/json; charset=utf-8", status_code=500
            )

        content_type = (req.headers.get("content-type") or "").lower()
        audio_bytes, audio_filename = None, None

        if content_type.startswith("application/json"):
            try:
                body = req.get_json()
            except Exception as e:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": f"JSON inválido: {str(e)}"}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8", status_code=400
                )
            audio_url = (body or {}).get("audio_url")
            if not audio_url:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": "Campo 'audio_url' é obrigatório"}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8", status_code=400
                )
            try:
                audio_bytes = _download_to_bytes(audio_url)
                audio_filename = os.path.basename(urlparse(audio_url).path) or "audio.wav"
                logger.info(f"Áudio baixado: {len(audio_bytes)} bytes, nome: {audio_filename}")
            except Exception as e:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": f"Erro ao baixar áudio: {str(e)}"}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8", status_code=400
                )

        elif content_type.startswith("audio/") or content_type.startswith("application/octet-stream"):
            audio_bytes = req.get_body()
            if not audio_bytes:
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": "Corpo da requisição vazio"}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8", status_code=400
                )
            audio_filename = "audio_upload.wav"
            logger.info(f"Áudio recebido: {len(audio_bytes)} bytes")
        else:
            return func.HttpResponse(
                json.dumps({"ok": False, "error": "Content-Type não suportado. Use 'application/json' com audio_url ou envie áudio binário."}, ensure_ascii=False),
                mimetype="application/json; charset=utf-8", status_code=400
            )

        # Salvar áudio original
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base_name = os.path.splitext(audio_filename)[0]
        try:
            original_audio_url = _upload_to_storage(
                container=os.getenv("STORAGE_CONTAINER_INPUT"),
                blob_name=f"{base_name}-{timestamp}.wav",
                content=audio_bytes,
                content_type="audio/wav"
            )
            logger.info(f"Áudio original salvo: {original_audio_url}")
        except Exception as e:
            logger.warning(f"Erro ao salvar áudio original: {e}")
            original_audio_url = None

        # Transcrever
        try:
            transcription_result = _speech_transcribe_from_bytes(audio_bytes, language="pt-BR")
            transcript_text = transcription_result.get("text", "")
            transcription_reason = transcription_result.get("reason", "")
            segments = transcription_result.get("segments", [])
            segment_count = transcription_result.get("segment_count", 0)

            logger.info(f"Transcrição: {len(transcript_text)} chars, {segment_count} segmentos, reason: {transcription_reason}")

            if not transcript_text.strip():
                return func.HttpResponse(
                    json.dumps({"ok": False, "error": "Nenhum texto foi transcrito do áudio", "reason": transcription_reason}, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8", status_code=400
                )
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"ok": False, "error": f"Erro na transcrição: {str(e)}"}, ensure_ascii=False),
                mimetype="application/json; charset=utf-8", status_code=500
            )

        # Salvar transcrição
        transcript_data = {
            "timestamp": timestamp,
            "audio_filename": audio_filename,
            "reason": transcription_reason,
            "text": transcript_text,
            "char_count": len(transcript_text),
            "segments": segments,
            "segment_count": segment_count,
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
            logger.info(f"Transcrição salva: {transcript_url}")
        except Exception as e:
            logger.warning(f"Erro ao salvar transcrição: {e}")
            transcript_url = None

        # Gerar ATA
        ata_content, ata_url, ata_error = None, None, None
        try:
            ata_content = _generate_meeting_minutes_with_timestamps(transcript_data)
            if ata_content:
                ata_url = _upload_to_storage(
                    container=os.getenv("STORAGE_CONTAINER_ATAS"),
                    blob_name=f"{base_name}-ata-{timestamp}.md",
                    content=ata_content,
                    content_type="text/markdown"
                )
                logger.info(f"ATA salva: {ata_url}")
            else:
                ata_error = "Resposta vazia do modelo."
        except Exception as e:
            ata_error = str(e)
            logger.error(f"Erro na geração da ATA: {e}")

        response_data = {
            "ok": True,
            "marker": MARKER,
            "timestamp": timestamp,
            "processing": {
                "audio_size_bytes": len(audio_bytes),
                "transcript_chars": len(transcript_text),
                "transcription_reason": transcription_reason,
                "segment_count": segment_count,
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
        return func.HttpResponse(json.dumps(response_data, ensure_ascii=False, indent=2),
                                 mimetype="application/json; charset=utf-8", status_code=200)

    except Exception as e:
        logger.exception("Erro não tratado na função")
        debug_mode = os.getenv("DEBUG") == "1"
        error_response = {"ok": False, "error": str(e), "marker": MARKER}
        if debug_mode:
            error_response["stack_trace"] = traceback.format_exc()
        return func.HttpResponse(json.dumps(error_response, ensure_ascii=False),
                                 mimetype="application/json; charset=utf-8", status_code=500)
