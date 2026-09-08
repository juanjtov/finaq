# FINAQ — single image for both Azure Container Apps (see docs §fabric):
#   telegram-bot : python -m scripts.run_telegram_bot   (default CMD)
#   streamlit    : streamlit run ui/app.py              (command override)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "scripts.run_telegram_bot"]
