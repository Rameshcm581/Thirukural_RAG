# Backend image for Hugging Face Spaces (Docker SDK). Also runs anywhere else:
#   docker build -t thirukural-api . && docker run -p 7860:7860 --env-file .env thirukural-api
FROM python:3.12-slim

# HF Spaces run the container as UID 1000. Create that user first so the pip
# packages, model cache, and app files are all owned by it.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR $HOME/app

# CPU-only torch first: the default wheel pulls ~2.5 GB of CUDA libs that a
# CPU host can't use. sentence-transformers then sees torch as satisfied.
RUN pip install --no-cache-dir --user --timeout 300 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --timeout 300 --retries 5 -r requirements.txt

# Bake the embedding model into the image so a cold start doesn't re-download ~1 GB.
ARG EMBEDDING_MODEL_NAME=paraphrase-multilingual-mpnet-base-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL_NAME}')"

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
