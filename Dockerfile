# syntax=docker/dockerfile:1

# Two stages so the model weights are downloaded once at build time and copied
# into the final image. Fetching them at startup instead costs ~75s of downtime
# on every restart and makes the container depend on huggingface.co being up.
FROM python:3.12-slim AS weights

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

# This stage's only job is to fill the HuggingFace cache, and downloading files
# does not need the library that later reads them: huggingface_hub is a few MB,
# where installing sentence-transformers to call its constructor for the
# download side effect would drag in torch (~475MB unpacked) to produce ~92MB
# of weights. That mattered in practice - the torch download is what kept
# failing on a flaky connection.
RUN pip install "huggingface_hub>=0.24"

# Must match qdrant_loader.MODEL_NAME: the query embedding has to come from the
# same model as the stored vectors, or the two are not comparable.
ARG MODEL_NAME=all-MiniLM-L6-v2
ENV HF_HOME=/opt/hf

# The repo ships the same weights several times over: ONNX, OpenVINO and
# TensorFlow copies the PyTorch path never loads, plus pytorch_model.bin, which
# is the .safetensors file in the older format - transformers prefers
# safetensors when both are present, so the .bin is 91MB of dead weight.
# Excluding all of them takes the cache from ~182MB to ~91MB. Everything else
# is allowed through rather than named explicitly, so a config file added
# upstream is still fetched instead of silently missing.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='sentence-transformers/${MODEL_NAME}', \
    ignore_patterns=['*.onnx', '*openvino*', '*.h5', '*tf_model*', '*rust_model*', '*.msgpack', '*.ot', '*.bin'], \
)"


FROM python:3.12-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1

# Requirements first, so a source edit doesn't invalidate the dependency layer.
WORKDIR /app
COPY requirements.txt .

# CPU-only torch: the default PyPI wheel bundles CUDA and is ~2.5GB, while
# nothing here touches a GPU - this is one 384-dimension MiniLM embedding per
# query. It is installed from PyTorch's own index first, then *pinned by a
# constraints file* for the second install. Without that pin the second command
# resolves against PyPI, where a higher-versioned CUDA build would satisfy
# sentence-transformers' `torch` requirement and silently replace this one.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip freeze | grep -i "^torch==" > /tmp/torch-constraint.txt \
 && pip install --constraint /tmp/torch-constraint.txt -r requirements.txt \
 && rm /tmp/torch-constraint.txt

# Ownership is set during the copy. A later `chown -R` would rewrite every
# file, and rewritten files land in a new layer - storing the ~92MB of weights
# twice in the image for nothing.
RUN useradd --create-home --uid 10001 okf
COPY --from=weights --chown=okf:okf /opt/hf /opt/hf
COPY --chown=okf:okf *.py ./
COPY --chown=okf:okf static/ ./static/

USER okf

EXPOSE 8000

# The compose healthcheck hits this too; it reports whether the model finished
# loading, which is the slow part of startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "\
import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4).status==200 else 1)"

# 0.0.0.0 because the port is published by the container runtime, which is what
# decides who can actually reach it. There is no authentication in front of
# this endpoint - publish it only on a network you trust, or put a proxy that
# authenticates in front of it.
CMD ["python", "webui.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
