import azure.functions as func
import json
import logging
import os
import traceback
from datetime import datetime
from urllib.parse import urlparse
import requests
import azure.cognitiveservices.speech as speechsdk
import openai
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
import time
import threading
import unicodedata

MARKER = "PRODUCTION_VERSION_V3_SIMPLE_FIXED"

# Lista simples de tipos MIME aceitos
ACCEPTED_MIME_TYPES = [
    'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/m4a', 'audio/mp4',
    'audio/aac', 'audio/vnd.dlna.adts', 'audio/adts',
    'application/octet-stream'
]

# Configurações
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1GB

def _normalize_text(text: str) -> str:
    """Normalizar texto para corrigir problemas de codificação"""
    if not text:
        return text
    
    text = unicodedata.normalize('NFC', text)
    
    corrections = {
        'Ã§': 'ç', 'Ã©': 'é', 'Ã¡': 'á', 'Ã³': 'ó', 'Ãº': 'ú',
        'Ã¢': 'â', 'Ãª': 'ê', 'Ã´': 'ô', 'Ã ': 'à', 'Ã£': 'ã'
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text.strip()

def _speech_transcribe_from_bytes(audio_bytes: bytes, language: str = "pt-BR") -> dict:
    """Transcrever áudio usando Azure Speech Service"""
    logging.info(f"Iniciando transcrição com {len(audio_bytes)} bytes")
    
    segments_with_time = []
    full_text = ""
    recognition_done = threading.Event()
    
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("SPEECH_KEY"),
            region=os.getenv("SPEECH_REGION")
        )
        
        speech_config.speech_recognition_language = "pt-BR"
        speech_config.output_format = speechsdk.OutputFormat.Detailed
        speech_config.request_word_level_timestamps()
        
        wave_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1
        )
        
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
        push_stream.write(audio_bytes)
        push_stream.close()

        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        def recognized_handler(evt):
            nonlocal full_text
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                start_time = evt.result.offset / 10000000
                duration = evt.result.duration / 10000000
                normalized_text = _normalize_text(evt.result.text.strip())
                
                segment = {
                    "start_seconds": round(start_time, 2),
                    "end_seconds": round(start_time + duration, 2),
                    "text": normalized_text
                }
                
                segments_with_time.append(segment)
                full_text += normalized_text + " "
        
        def session_stopped_handler(evt):
            recognition_done.set()
        
        recognizer.recognized.connect(recognized_handler)
        recognizer.session_stopped.connect(session_stopped_handler)
        
        recognizer.start_continuous_recognition()
        recognition_done.wait(timeout=300)  # 5 minutos
        recognizer.stop_continuous_recognition()
        
        full_text = _normalize_text(full_text.strip())
        
        return {
            "ok": True,
            "text": full_text,
            "segments": segments_with_time,
            "segment_count": len(segments_with_time),
            "reason": "RecognizedSpeechWithTimestamps"
        }
            
    except Exception as e:
        logging.exception("Erro na transcrição")
        raise RuntimeError(f"Erro na transcrição: {str(e)}")

def _generate_meeting_minutes(transcript_data: dict) -> str:
    """Gerar ATA usando Azure OpenAI"""
    try:
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-05-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        
        full_text = transcript_data.get("text", "")
        
        system_prompt = (
            "Você é um assistente especializado em elaborar ATAs de reunião em português do Brasil. "
            "Gere uma ATA profissional com: participantes, contexto, pontos discutidos, decisões, "
            "pendências e próximos passos. Use acentuação correta."
        )
        
        user_prompt = f"Gere uma ATA baseada na transcrição: {full_text}"
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        return _normalize_text(response.choices[0].message.content)
        
    except Exception as e:
        logging.exception("Erro na geração da ATA")
        raise RuntimeError(f"Erro na geração da ATA: {str(e)}")

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Função principal simplificada para resolver erro 500"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Ping test
        if req.params.get("ping") == "1":
            return func.HttpResponse(
                json.dumps({
                    "ok": True, 
                    "status": "pong", 
                    "marker": MARKER,
                    "message": "Backend simplificado funcionando!",
                    "accepted_mime_types": ACCEPTED_MIME_TYPES
                }, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                status_code=200
            )

        # Verificar variáveis essenciais
        required_vars = ["SPEECH_KEY", "SPEECH_REGION", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT"]
        missing = [k for k in required_vars if not os.getenv(k)]
        
        if missing:
            return func.HttpResponse(
                json.dumps({"ok": False, "error": f"Variáveis não configuradas: {', '.join(missing)}"}, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                status_code=500
            )

        # Obter áudio
        content_type = (req.headers.get("content-type") or "").lower()
        audio_bytes = req.get_body()
        
        if not audio_bytes:
            return func.HttpResponse(
                json.dumps({"ok": False, "error": "Corpo da requisição vazio"}, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                status_code=400
            )

        # Validar tamanho
        if len(audio_bytes) > MAX_FILE_SIZE:
            return func.HttpResponse(
                json.dumps({
                    "ok": False,
                    "error": f"Arquivo muito grande: {len(audio_bytes) / (1024**3):.2f}GB. Máximo: 1GB"
                }, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                status_code=400
            )

        logger.info(f"Processando arquivo: {len(audio_bytes)} bytes, Content-Type: {content_type}")

        # Transcrever
        transcription_result = _speech_transcribe_from_bytes(audio_bytes)
        transcript_text = transcription_result.get("text", "")
        segments = transcription_result.get("segments", [])
        
        if not transcript_text.strip():
            return func.HttpResponse(
                json.dumps({
                    "ok": False,
                    "error": "Nenhum texto foi transcrito do áudio"
                }, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                status_code=400
            )

        # Gerar ATA
        ata_content = None
        ata_error = None
        
        try:
            ata_content = _generate_meeting_minutes(transcription_result)
        except Exception as e:
            ata_error = str(e)
            logger.error(f"Erro na ATA: {e}")

        # Resposta
        response_data = {
            "ok": True,
            "marker": MARKER,
            "timestamp": datetime.utcnow().isoformat(),
            "processing": {
                "audio_size_bytes": len(audio_bytes),
                "transcript_chars": len(transcript_text),
                "segment_count": len(segments),
                "ata_generated": ata_content is not None,
                "encoding": "utf-8"
            },
            "preview": {
                "transcript": transcript_text[:300] + ("..." if len(transcript_text) > 300 else ""),
                "ata": ata_content[:400] + ("..." if ata_content and len(ata_content) > 400 else "") if ata_content else None
            },
            "errors": {
                "ata_generation": ata_error
            }
        }

        return func.HttpResponse(
            json.dumps(response_data, ensure_ascii=False, indent=2),
            mimetype="application/json; charset=utf-8",
            status_code=200
        )

    except Exception as e:
        logger.exception("Erro não tratado")
        
        return func.HttpResponse(
            json.dumps({
                "ok": False,
                "error": str(e),
                "marker": MARKER,
                "stack_trace": traceback.format_exc() if os.getenv("DEBUG") == "1" else None
            }, ensure_ascii=False),
            mimetype="application/json; charset=utf-8",
            status_code=500
        )
