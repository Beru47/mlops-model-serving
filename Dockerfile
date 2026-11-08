FROM python:3.11-slim
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
# registry (sqlite + artifacts) lives on a shared volume
WORKDIR /data
