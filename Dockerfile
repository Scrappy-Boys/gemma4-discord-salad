# Gemma 4 E2B fine-tune for SaladCloud (RTX 3090, 24 GB).
# Build:  docker build -t YOURNAME/gemma4-discord-salad:1 .
# Push :  docker push YOURNAME/gemma4-discord-salad:1
#
# No secrets are baked in; all keys arrive as environment variables from the Salad portal.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# build-essential: Triton (used by Unsloth) compiles small helpers at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Unsloth pulls in a matching torch/transformers/trl. Rebuild the image to pick up newer versions.
RUN pip install -U pip \
 && pip install -U unsloth unsloth_zoo \
 && pip install -r requirements.txt

COPY cloud.py entrypoint.py convert_dataset.py train_gemma4_e2b.py ./
RUN mkdir -p /workspace

CMD ["python", "-u", "entrypoint.py"]
