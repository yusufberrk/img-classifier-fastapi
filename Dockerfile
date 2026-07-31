# Hafif bir python imajı
FROM python:3.10-slim

# Çalışma dizinini belirle
WORKDIR /code

# Gereksinimleri kopyala ve yükle
COPY ./requirements.txt /code/requirements.txt
RUN pip install --default-timeout=1000 --no-cache-dir --upgrade -r /code/requirements.txt

# Uygulama kodlarını kopyala
COPY ./app /code/app

# FastAPI sunucusunu 7001 portunda başlat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7001"]