FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

ENTRYPOINT ["ai-free-swap"]
CMD ["--config", "/app/config.yaml"]
