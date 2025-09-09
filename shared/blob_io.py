from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from .config import Settings
import io

_cred = DefaultAzureCredential()  # usa Managed Identity na Azure; local usa CLI/VSCode login
_blob = BlobServiceClient(
    account_url=f"https://{Settings.STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
    credential=_cred
)

def upload_text(container: str, blob_name: str, text: str, overwrite: bool = True):
    container_client = _blob.get_container_client(container)
    container_client.upload_blob(name=blob_name, data=text.encode("utf-8"), overwrite=overwrite)

def upload_bytes(container: str, blob_name: str, data: bytes, overwrite: bool = True):
    container_client = _blob.get_container_client(container)
    container_client.upload_blob(name=blob_name, data=io.BytesIO(data), overwrite=overwrite)

def download_text(container: str, blob_name: str) -> str:
    container_client = _blob.get_container_client(container)
    stream = container_client.download_blob(blob_name)
    return stream.content_as_text()

def blob_url(container: str, blob_name: str) -> str:
    return f"https://{Settings.STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{container}/{blob_name}"
