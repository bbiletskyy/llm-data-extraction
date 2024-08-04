FROM python:3.10-slim
WORKDIR /app

COPY . .

COPY llm_data_extraction/ ./llm_data_extraction/
COPY setup.cfg .
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENV OPENAI_API_KEY=""

CMD ["uvicorn", "llm_data_extraction.main:app", "--host", "0.0.0.0", "--port", "8000"]