FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY config.yaml.cloud /app/config.yaml
EXPOSE 8000

ENTRYPOINT ["ai-free-swap"]
CMD ["--config", "/app/config.yaml"]
