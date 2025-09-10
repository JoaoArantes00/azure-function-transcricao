import azure.functions as func
import json, logging, os, traceback
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

MARKER = "PRODUCTION_VERSION_V2_TIMESTAMPS_UTF8"

def _normalize_text(text: str) -> str:
    """Normalizar texto para corrigir problemas de codificação"""
    if not text:
        return text
    
    # Normalizar Unicode para forma canônica
    text = unicodedata.normalize('NFC', text)
    
    # Correções específicas de caracteres mal interpretados comuns em português
    corrections = {
        'Ã§': 'ç',  'Ã‡': 'Ç',
        'Ã£': 'ã',  'Ã': 'Ã',
        'Ã¡': 'á', 
        'Ã©': 'é',  'Ã‰': 'É',
        'Ã³': 'ó',  'Ã"': 'Ó',
        'Ãº': 'ú',  'Ãš': 'Ú',
        'Ã¢': 'â',  'Ã‚': 'Â',
        'Ãª': 'ê',  'ÃŠ': 'Ê',
        'Ã´': 'ô',  'Ã"': 'Ô',
        'Ã ': 'à',  'Ã€': 'À',
        'Ã¨': 'è',  'Ãˆ': 'È',
        'Ã¬': 'ì',  'ÃŒ': 'Ì',
        'Ã²': 'ò',  "Ã'": 'Ò',
        'Ã¹': 'ù',  'Ã™': 'Ù',
        'Ã±': 'ñ',  "Ã'": 'Ñ',
        'Ã¼': 'ü',  'Ãœ': 'Ü',
        'ÃªÃ§': 'eç',
        'ÃƒÂ': '',  # Remove marcadores UTF-8 malformados
        'â€™': "'",  # Apóstrofe
        'â€œ': '"',  # Aspas esquerda
        'â€': '"',   # Aspas direita
        'â€"': '–',  # Travessão
        'â€¦': '…',  # Reticências
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Remover caracteres de controle mantendo apenas os necessários
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
    
    return text.strip()

def _download_to_bytes(url: str) -> bytes:
    """Download arquivo de uma URL"""
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def _require_env(keys):
    """Verificar se todas as variáveis de ambiente necessárias estão configuradas"""
    missing = [k for k in keys if not os.getenv(k)]
    return (len(missing) == 0, f"Variáveis não configuradas: {', '.join(missing)}" if missing else "")

def _format_timestamp(seconds):
    """Converter segundos para formato MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def _speech_transcribe_from_bytes(audio_bytes: bytes, language: str = "pt-BR") -> dict:
    """Transcrever áudio usando Azure Speech Service com timestamps e correção UTF-8"""
    print(f"_speech_transcribe_from_bytes chamada com {len(audio_bytes)} bytes, language={language}")
    
    segments_with_time = []
    full_text = ""
    recognition_done = threading.Event()
    
    try:
        # Configurar Speech Service com configurações otimizadas para português
        print("Configurando Speech Service...")
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("SPEECH_KEY"),
            region=os.getenv("SPEECH_REGION")
        )
        
        # Configurações específicas para português brasileiro
        speech_config.speech_recognition_language = "pt-BR"
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_RecoLanguage, "pt-BR")
        speech_config.output_format = speechsdk.OutputFormat.Detailed
        
        # Configurações para melhor reconhecimento e timestamps
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "5000")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")
        speech_config.request_word_level_timestamps()
        
        # Configurações adicionais para melhor qualidade
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestDetailedResultTrueFalse, "true")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_RequestProfanityFilterTrueFalse, "false")
        
        print(f"Speech config criado - Region: {os.getenv('SPEECH_REGION')}, Language: {language}")

        # Definir formato de áudio explicitamente
        wave_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=48000,
            bits_per_sample=16,
            channels=2
        )
        
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=wave_format)
        push_stream.write(audio_bytes)
        push_stream.close()
        print("Áudio escrito no stream")

        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        print("Recognizer criado")
        
        # Callback para capturar resultados com timestamp e normalização UTF-8
        def recognized_handler(evt):
            nonlocal full_text
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                start_time = evt.result.offset / 10000000  # Converter para segundos
                duration = evt.result.duration / 10000000
                end_time = start_time + duration
                
                # Normalizar texto para corrigir problemas de codificação
                raw_text = evt.result.text.strip()
                normalized_text = _normalize_text(raw_text)
                
                segment = {
                    "start_seconds": round(start_time, 2),
                    "end_seconds": round(end_time, 2),
                    "start_time": _format_timestamp(start_time),
                    "end_time": _format_timestamp(end_time),
                    "text": normalized_text,
                    "raw_text": raw_text  # Manter texto original para debug se necessário
                }
                
                segments_with_time.append(segment)
                full_text += normalized_text + " "
                print(f"Segmento reconhecido: [{segment['start_time']}-{segment['end_time']}] {normalized_text[:50]}...")
        
        def session_stopped_handler(evt):
            print("Sessão de reconhecimento finalizada")
            recognition_done.set()
        
        def canceled_handler(evt):
            print(f"Reconhecimento cancelado: {evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                print(f"Detalhes do erro: {evt.error_details}")
            recognition_done.set()
        
        # Conectar callbacks
        recognizer.recognized.connect(recognized_handler)
        recognizer.session_stopped.connect(session_stopped_handler)
        recognizer.canceled.connect(canceled_handler)
        
        # Iniciar reconhecimento contínuo
        print("Iniciando reconhecimento contínuo...")
        recognizer.start_continuous_recognition()
        
        # Aguardar conclusão (timeout de 60 segundos)
        if recognition_done.wait(timeout=60):
            print("Reconhecimento concluído")
        else:
            print("Timeout no reconhecimento")
        
        recognizer.stop_continuous_recognition()
        
        # Normalizar texto completo
        full_text = _normalize_text(full_text.strip())
        
        if segments_with_time:
            print(f"Transcrição concluída: {len(segments_with_time)} segmentos, {len(full_text)} caracteres")
            return {
                "ok": True,
                "text": full_text,
                "segments": segments_with_time,
                "segment_count": len(segments_with_time),
                "reason": "RecognizedSpeechWithTimestamps"
            }
        else:
            print("Nenhum segmento reconhecido")
            return {
                "ok": True,
                "text": "",
                "segments": [],
                "segment_count": 0,
                "reason": "NoMatch"
            }
            
    except Exception as e:
        print(f"Exceção na transcrição: {str(e)}")
        logging.exception("Erro na transcrição de áudio")
        raise RuntimeError(f"Erro na transcrição: {str(e)}")

def _generate_meeting_minutes_with_timestamps(transcript_data: dict) -> str:
    """Gerar ATA usando Azure OpenAI com informações de timestamp e correção UTF-8"""
    try:
        # Configurar cliente OpenAI com nova sintaxe (versão 1.x)
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-05-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        
        # Construir contexto com timestamps
        full_text = transcript_data.get("text", "")
        segments = transcript_data.get("segments", [])
        
        # Criar transcrição formatada com timestamps
        formatted_transcript = ""
        for segment in segments:
            formatted_transcript += f"[{segment['start_time']}] {segment['text']}\n"
        
        system_prompt = (
            "Você é um assistente especializado em elaborar ATAs de reunião em português do Brasil. "
            "Você receberá uma transcrição com marcações de tempo (timestamps) no formato [MM:SS]. "
            "IMPORTANTE: Use SEMPRE acentuação correta e caracteres especiais do português brasileiro. "
            "Produza uma ATA profissional e bem estruturada com as seguintes seções:\n\n"
            "1. **Participantes** (se identificáveis na transcrição)\n"
            "2. **Contexto/Objetivo da Reunião**\n"
            "3. **Pontos Discutidos** (principais tópicos abordados)\n"
            "4. **Decisões Tomadas** (conclusões e deliberações)\n"
            "5. **Pendências/Ações** (tarefas identificadas, com responsáveis quando possível)\n"
            "6. **Próximos Passos** (encaminhamentos futuros)\n"
            "7. **Anexo: Transcrição Completa com Timestamps**\n\n"
            "INSTRUÇÕES ESPECÍFICAS:\n"
            "- Use SEMPRE a grafia correta em português: não, são, transcrição, reunião, etc.\n"
            "- Quando referenciar pontos específicos da discussão, inclua o timestamp correspondente\n"
            "- Mantenha formatação em Markdown\n"
            "- Use linguagem profissional e concisa\n"
            "- Corrija automaticamente qualquer erro de acentuação que possa estar na transcrição"
        )
        
        user_prompt = (
            f"Com base na transcrição da reunião abaixo, gere uma ATA profissional.\n\n"
            f"TRANSCRIÇÃO COM TIMESTAMPS:\n{formatted_transcript}\n\n"
            f"TEXTO COMPLETO: {full_text}\n\n"
            f"Gere a ATA seguindo o formato solicitado, corrigindo qualquer erro de acentuação."
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
        
        # Normalizar resposta do OpenAI para garantir UTF-8 correto
        ata_content = response.choices[0].message.content
        return _normalize_text(ata_content)
        
    except Exception as e:
        logging.exception("Erro na geração da ATA")
        raise RuntimeError(f"Erro na geração da ATA: {str(e)}")

def _upload_to_storage(container: str, blob_name: str, content, content_type: str = "text/plain"):
    """Upload de conteúdo para Azure Storage com codificação UTF-8"""
    try:
        # Usar DefaultAzureCredential para autenticação
        credential = DefaultAzureCredential()
        
        # Criar cliente do Blob Storage
        storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
        blob_service_client = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=credential
        )
        
        # Determinar como processar o conteúdo
        if isinstance(content, bytes):
            # Para dados binários (áudio)
            data_to_upload = content
        else:
            # Para dados texto (JSON, markdown) - garantir UTF-8
            normalized_content = _normalize_text(content) if isinstance(content, str) else content
            data_to_upload = normalized_content.encode("utf-8")
        
        # Upload do conteúdo
        container_client = blob_service_client.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        
        # Usar ContentSettings com charset UTF-8
        if content_type.startswith("text/"):
            content_settings = ContentSettings(
                content_type=f"{content_type}; charset=utf-8"
            )
        else:
            content_settings = ContentSettings(content_type=content_type)
        
        blob_client.upload_blob(
            data=data_to_upload,
            overwrite=True,
            content_settings=content_settings
        )
        
        # Retornar URL do blob
        return f"https://{storage_account_name}.blob.core.windows.net/{container}/{blob_name}"
        
    except Exception as e:
        logging.exception("Erro no upload para storage")
        raise RuntimeError(f"Erro no upload: {str(e)}")

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Função principal para transcrição de áudio e geração de ATA com timestamps e correção UTF-8"""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Verificar se é ping
        if req.params.get("ping") == "1":
            return func.HttpResponse(
                json.dumps({
                    "ok": True, 
                    "status": "pong", 
                    "marker": MARKER,
                    "message": "Serviço de transcrição com timestamps e correção UTF-8 funcionando!",
                    "features": [
                        "audio_transcription", 
                        "timestamp_tracking", 
                        "utf8_normalization",
                        "ata_generation", 
                        "azure_storage"
                    ]
                }, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                status_code=200
            )

        # Verificar variáveis de ambiente necessárias
        required_vars = [
            "SPEECH_KEY", "SPEECH_REGION",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT",
            "STORAGE_ACCOUNT_NAME", "STORAGE_CONTAINER_INPUT", 
            "STORAGE_CONTAINER_TRANSCRIPTS", "STORAGE_CONTAINER_ATAS"
        ]
        
        env_ok, env_error = _require_env(required_vars)
        if not env_ok:
            return func.HttpResponse(
                json.dumps({"ok": False, "error": env_error}, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                status_code=500
            )

        # Processar requisição
        content_type = (req.headers.get("content-type") or "").lower()
        audio_bytes = None
        audio_filename = None

        if content_type.startswith("application/json"):
            # Modo JSON: espera URL do áudio
            try:
                body = req.get_json()
                if not body:
                    return func.HttpResponse(
                        json.dumps({
                            "ok": False, 
                            "error": "JSON vazio. Envie {'audio_url': 'https://exemplo.com/audio.wav'}"
                        }, ensure_ascii=False),
                        mimetype="application/json; charset=utf-8",
                        status_code=400
                    )
            except Exception as e:
                return func.HttpResponse(
                    json.dumps({
                        "ok": False, 
                        "error": f"JSON inválido: {str(e)}"
                    }, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    status_code=400
                )
            
            audio_url = body.get("audio_url")
            if not audio_url:
                return func.HttpResponse(
                    json.dumps({
                        "ok": False, 
                        "error": "Campo 'audio_url' é obrigatório"
                    }, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    status_code=400
                )
            
            # Download do áudio
            try:
                audio_bytes = _download_to_bytes(audio_url)
                audio_filename = os.path.basename(urlparse(audio_url).path) or "audio.wav"
                logger.info(f"Áudio baixado: {len(audio_bytes)} bytes, nome: {audio_filename}")
            except Exception as e:
                return func.HttpResponse(
                    json.dumps({
                        "ok": False, 
                        "error": f"Erro ao baixar áudio: {str(e)}"
                    }, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    status_code=400
                )

        elif content_type.startswith("audio/") or content_type.startswith("application/octet-stream"):
            # Modo binário: áudio enviado diretamente
            audio_bytes = req.get_body()
            if not audio_bytes:
                return func.HttpResponse(
                    json.dumps({
                        "ok": False, 
                        "error": "Corpo da requisição vazio"
                    }, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    status_code=400
                )
            audio_filename = "audio_upload.wav"
            logger.info(f"Áudio recebido: {len(audio_bytes)} bytes")

        else:
            return func.HttpResponse(
                json.dumps({
                    "ok": False, 
                    "error": "Content-Type não suportado. Use 'application/json' com audio_url ou envie áudio binário."
                }, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                status_code=400
            )

        # Salvar áudio original (opcional)
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

        # Transcrever áudio com timestamps e correção UTF-8
        try:
            transcription_result = _speech_transcribe_from_bytes(audio_bytes)
            transcript_text = transcription_result.get("text", "")
            transcription_reason = transcription_result.get("reason", "")
            segments = transcription_result.get("segments", [])
            segment_count = transcription_result.get("segment_count", 0)
            
            logger.info(f"Transcrição concluída: {len(transcript_text)} caracteres, {segment_count} segmentos, reason: {transcription_reason}")
            
            if not transcript_text.strip():
                return func.HttpResponse(
                    json.dumps({
                        "ok": False,
                        "error": "Nenhum texto foi transcrito do áudio",
                        "reason": transcription_reason
                    }, ensure_ascii=False),
                    mimetype="application/json; charset=utf-8",
                    status_code=400
                )
                
        except Exception as e:
            return func.HttpResponse(
                json.dumps({
                    "ok": False,
                    "error": f"Erro na transcrição: {str(e)}"
                }, ensure_ascii=False),
                mimetype="application/json; charset=utf-8",
                status_code=500
            )

        # Salvar transcrição com timestamps
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

        # Gerar ATA com timestamps e correção UTF-8
        ata_content = None
        ata_url = None
        ata_error = None
        
        try:
            ata_content = _generate_meeting_minutes_with_timestamps(transcript_data)
            logger.info(f"ATA gerada: {len(ata_content)} caracteres")
            
            # Salvar ATA
            ata_url = _upload_to_storage(
                container=os.getenv("STORAGE_CONTAINER_ATAS"),
                blob_name=f"{base_name}-ata-{timestamp}.md",
                content=ata_content,
                content_type="text/markdown"
            )
            logger.info(f"ATA salva: {ata_url}")
            
        except Exception as e:
            ata_error = str(e)
            logger.error(f"Erro na geração da ATA: {e}")

        # Resposta final
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

        return func.HttpResponse(
            json.dumps(response_data, ensure_ascii=False, indent=2),
            mimetype="application/json; charset=utf-8",
            status_code=200
        )

    except Exception as e:
        logger.exception("Erro não tratado na função")
        
        # Em modo debug, incluir stack trace
        debug_mode = os.getenv("DEBUG") == "1"
        error_response = {
            "ok": False,
            "error": str(e),
            "marker": MARKER
        }
        
        if debug_mode:
            error_response["stack_trace"] = traceback.format_exc()
        
        return func.HttpResponse(
            json.dumps(error_response, ensure_ascii=False),
            mimetype="application/json; charset=utf-8",
            status_code=500
        )
