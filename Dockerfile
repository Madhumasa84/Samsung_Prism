# Digest captured during the verified x86_64 build; update deliberately when
# changing the base image rather than inheriting a moving tag.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.lock.txt pyproject.toml README.md ./
COPY src ./src

# The lock file pins the core offline runtime. The dense extra is intentionally
# not installed in this small image; install it separately when model weights
# and the required CPU runtime are available.
RUN python -m pip install --no-cache-dir -r requirements.lock.txt \
    && python -m pip install --no-cache-dir --no-deps .

COPY data/synthetic ./data/synthetic
COPY data/evaluation ./data/evaluation
COPY examples ./examples
COPY docs ./docs
COPY PHASE1_CHECKLIST.md ./
COPY .env.example ./

ENTRYPOINT ["python", "-m", "flowcontext"]
