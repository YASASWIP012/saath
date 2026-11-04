FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf PIP_NO_CACHE_DIR=1 SAATH_DEVICE=cpu TOKENIZERS_PARALLELISM=false
WORKDIR /app

# CPU-only torch first, so the pinned requirements don't pull CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the two encoders into the image so the first request is not a download.
RUN python -c "from sentence_transformers import SentenceTransformer as S; \
[S(m, device='cpu') for m in ['intfloat/multilingual-e5-base', 'sentence-transformers/all-mpnet-base-v2']]"

COPY saath ./saath
COPY app ./app
COPY artifacts ./artifacts
COPY results/report.json ./results/report.json

RUN useradd -m -u 1000 user && chown -R user /app /tmp/hf
USER user
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
