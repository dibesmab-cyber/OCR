from fastapi import APIRouter, Body, HTTPException, UploadFile, File
from pydantic import BaseModel
import httpx
import os
import fitz  # PyMuPDF — PDF sayfa içeriğini okumak için

router = APIRouter(prefix="/api/v1", tags=["kumru"])

# Pydantic modelleri ile veri doğrulama ve dokümantasyon
class KumruRequest(BaseModel):
    question: str

class KumruResponse(BaseModel):
    kumru_response: str



OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "receptim/kumru-2b")
# Ollama generate API endpoint'i
OLLAMA_GENERATE_URL = f"{OLLAMA_URL}/api/generate"


@router.get("/kumru/health")
async def kumru_health_check():
    return {"status": "Kumru router is healthy"}

@router.post("/llm/kumru/ask", response_model=KumruResponse)
async def ask_kumru(request: KumruRequest):
    """Ask a question to the Kumru-2B model via Ollama."""
    payload = {
        "model": MODEL_NAME,
        "prompt": request.question,
        "stream": False
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OLLAMA_GENERATE_URL, json=payload, timeout=120.0)
            response.raise_for_status()  # HTTP 4xx veya 5xx hatalarında exception fırlatır
            data = response.json()
            return KumruResponse(kumru_response=data.get("response", ""))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Ollama service is unavailable: {repr(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
    

@router.post("/llm/kumru/send_documents", response_model=KumruResponse)
async def send_docs_kumru(file: UploadFile = File(...)):
    """
    Kullanıcıdan PDF dosyası alır, PDF'i sayfa sayfa okur ve
    her sayfayı Kumru modeline gönderir.
    Modelden gelen yanıtları birleştirip döndürür.
    """
    try:
        # PDF içeriğini oku
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        full_response = ""

        async with httpx.AsyncClient() as client:
            for page_num, page in enumerate(pdf_document, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue  # boş sayfaları atla

                prompt = f"Sayfa {page_num} içeriği:\n\n{text}\n\nBu sayfadaki metni çıkar. Kendin hiçbir şey ekleme. Metni olduğu gibi ver. Sakın özetleme sadece tüm metni olduğu gibi çıkar:"
                payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}

                response = await client.post(OLLAMA_GENERATE_URL, json=payload, timeout=120.0)
                response.raise_for_status()
                data = response.json()
                page_summary = data.get("response", "")
                full_response += f"\n\n--- Sayfa {page_num} ---\n{page_summary}"

        return KumruResponse(kumru_response=full_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF işleme hatası: {str(e)}")
    

@router.post("/llm/kumru/send_pdf", response_model=KumruResponse)
async def send_pdf_kumru(file: UploadFile = File(...)):
    """
    PDF dosyasını alır, her sayfayı Kumru modeline gönderir.
    Eğer sayfa metin tabanlıysa doğrudan text'i yollar,
    eğer boşsa yine modeli dener (OCR benzeri şekilde).
    """
    try:
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        full_response = ""

        async with httpx.AsyncClient() as client:
            for page_num, page in enumerate(pdf_document, start=1):
                text = page.get_text("text").strip()

                # Prompt hazırlama (sayfa boş mu dolu mu durumuna göre)
                if text:
                    prompt = f"Bu sayfadaki metni çıkar:\n\n{text}"
                else:
                    prompt = (
                        "Bu sayfadaki metni çıkar. Sayfa görüntü içeriyorsa görüntüden metin çıkar. "
                        "Sakın özetleme sadece bu sayfadan metni olduğu gibi çıkar. "
                        "Eğer sayfa bir görüntü içeriyorsa veya metin okunamıyorsa, "
                    )

                payload = {
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                }

                try:
                    response = await client.post(
                        OLLAMA_GENERATE_URL,
                        json=payload,
                        timeout=120.0
                    )
                    response.raise_for_status()
                    data = response.json()
                    page_text = data.get("response", "").strip()

                except Exception as model_err:
                    page_text = f"(Sayfa {page_num} için model hatası: {model_err})"

                full_response += f"\n\n--- Sayfa {page_num} ---\n{page_text}"

        return KumruResponse(kumru_response=full_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF işleme hatası: {str(e)}")
