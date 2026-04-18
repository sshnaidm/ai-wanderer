FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml requirements.txt config.yaml.example ./
COPY src/ src/
RUN pip install --no-cache-dir -r requirements.txt && cp config.yaml.example config.yaml

EXPOSE 8000

ENTRYPOINT ["ai-free-swap"]
CMD ["--config", "/app/config.yaml"]
