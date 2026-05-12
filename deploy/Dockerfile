FROM vllm/vllm-openai:latest

WORKDIR /app

ARG LLM_MODEL=Qwen/Qwen3-8B

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_lg

COPY . .

# Pre-download GLiNER NER model — no network needed at runtime
RUN python -c "\
from gliner import GLiNER; \
GLiNER.from_pretrained('urchade/gliner_large-v2.1'); \
print('GLiNER cached')"

# Pre-download LLM model — eliminates first-run HuggingFace download on RunPod
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${LLM_MODEL}'); \
print('LLM cached: ${LLM_MODEL}')"

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000 8001

ENTRYPOINT ["/start.sh"]
