FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml config.yaml.example ./
COPY src/ src/
RUN pip install --no-cache-dir .

EXPOSE 8000

ENTRYPOINT ["ai-free-swap"]
CMD ["--config", "/app/config.yaml"]
