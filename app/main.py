from fastapi import FastAPI, UploadFile, File
from app.model import predict_image

app = FastAPI(title="Fruit Object Detection API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Yüklenen dosyayı byte olarak oku ve modele gönder
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return result