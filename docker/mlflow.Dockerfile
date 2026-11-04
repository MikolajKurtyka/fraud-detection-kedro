# MLflow tracking server with Postgres support.
FROM python:3.12-slim
RUN pip install --no-cache-dir "mlflow>=3.5,<4" "psycopg[binary]>=3.2"
EXPOSE 5000
