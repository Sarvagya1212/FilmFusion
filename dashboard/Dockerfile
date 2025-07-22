
FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

RUN pip install --upgrade pip wheel setuptools

COPY dashboard/requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt



FROM python:3.9-slim 

WORKDIR /app

COPY --from=builder /app/wheels /wheels

COPY dashboard/ ./dashboard
COPY src/ ./src         

RUN pip install --no-cache-dir /wheels/*

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app1.py", "--server.port=8501", "--server.address=0.0.0.0"]
