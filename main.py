# main.py
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, confloat
import httpx
from geopy.distance import geodesic
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Model
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool

# =========================
# Konfigurasi (Bagian Lama)
# =========================
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
DEFAULT_RADIUS_M = int(os.getenv("DEFAULT_RADIUS_M", "5000"))
MAX_RADIUS_M = int(os.getenv("MAX_RADIUS_M", "20000"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "15"))

# =========================
# Konfigurasi ML (Bagian Baru)
# =========================
MODEL_PATH = "models/model_05102025_225139.h5"
ML_WINDOW_SIZE = 12  # Ukuran window (12 sampel) dari script Anda
ml_models = {} # Dictionary untuk menyimpan model yang sudah di-load

# =========================
# FastAPI Lifespan (Untuk Load Model saat Startup)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Aksi yang dijalankan saat server startup
    print("Memuat model Keras...")
    try:
        model = load_model(MODEL_PATH)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        ml_models['afib_model'] = model
        print(f"Model {MODEL_PATH} berhasil dimuat.")
    except Exception as e:
        print(f"ERROR: Gagal memuat model di {MODEL_PATH}. Detail: {e}")
        ml_models['afib_model'] = None
    
    yield
    
    # Aksi yang dijalankan saat server shutdown
    print("Membersihkan resource...")
    ml_models.clear()

app = FastAPI(
    title="Arterythm Backend", 
    version="1.1.0",
    lifespan=lifespan # Daftarkan lifespan event
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Skema Request/Response (Rumah Sakit)
# =========================
class LocationData(BaseModel):
    lat: confloat(ge=-90, le=90)
    lon: confloat(ge=-180, le=180)
    radius_m: int = Field(DEFAULT_RADIUS_M, ge=100, le=MAX_RADIUS_M)

class Hospital(BaseModel):
    id: int
    name: str
    latitude: float
    longitude: float
    distance_km: float

class HospitalsResponse(BaseModel):
    count: int
    radius_m: int
    hospitals: List[Hospital]

# =========================
# Skema Request/Response (Prediksi ML) (BARU)
# =========================
class PpgPredictionRequest(BaseModel):
    # Aplikasi Flutter akan mengirim data mentah ppg sebagai list angka
    ppg_data: List[float] = Field(..., example=[102.0, 103.0, 105.0, ...])

class PpgPredictionResponse(BaseModel):
    total_windows: int
    afib_window_count: int
    mean_probability: float
    diagnosis: str # "Normal" atau "Kemungkinan Atrial Fibrilasi"

# =========================
# Logika Prediksi ML (BARU)
# =========================
def _run_prediction_pipeline(raw_ppg: List[float]) -> Dict:
    """
    Fungsi Synchronous yang menjalankan pipeline ML.
    Ini akan dijalankan di thread pool terpisah.
    """
    
    # 1. Pastikan model sudah di-load
    model = ml_models.get('afib_model')
    if model is None:
        raise HTTPException(status_code=500, detail="Model prediksi tidak berhasil dimuat di server.")
        
    # 2. Replikasi Preprocessing dari script Anda
    df = pd.DataFrame(raw_ppg, columns=['ppg'])
    
    # Scaling (Sesuai script Anda, scaler di-fit ke data input)
    # Catatan: Ini adalah praktik yang tidak umum, 
    # biasanya scaler di-fit pada data training dan disimpan.
    # Namun, kami mereplikasi logika script Anda dengan tepat.
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    ppg_signal = scaled_data.flatten() # Ambil sebagai array numpy

    # 3. Windowing (Sesuai script Anda)
    n_windows = len(ppg_signal) // ML_WINDOW_SIZE
    if n_windows == 0:
        raise HTTPException(status_code=400, detail=f"Data PPG tidak cukup. Diperlukan setidaknya {ML_WINDOW_SIZE} sampel.")

    windows_list = []
    for i in range(n_windows):
        start = i * ML_WINDOW_SIZE
        end = start + ML_WINDOW_SIZE
        window = ppg_signal[start:end]
        windows_list.append(window)
    
    x = np.array(windows_list)
    # Pastikan input shape sesuai dengan model (jika model Anda LSTM/CNN)
    # Jika model Anda mengharapkan shape (n_windows, window_size, 1)
    # x = x.reshape(x.shape[0], x.shape[1], 1)

    # 4. Predict
    y_pred_prob = model.predict(x).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # 5. Buat Hasil Agregat
    total_windows = len(y_pred)
    afib_windows = int(np.sum(y_pred))
    mean_prob = float(np.mean(y_pred_prob))
    
    if afib_windows > 0:
        diagnosis = "Kemungkinan Atrial Fibrilasi"
    else:
        diagnosis = "Normal"
        
    return {
        "total_windows": total_windows,
        "afib_window_count": afib_windows,
        "mean_probability": mean_prob,
        "diagnosis": diagnosis
    }

# =========================
# Util Overpass (Tidak Berubah)
# =========================
def _build_overpass_query(lat: float, lon: float, radius_m: int) -> str:
    # ... (kode Anda tidak berubah)
    return f"""
[out:json][timeout:25];
(
  node["amenity"="hospital"](around:{radius_m},{lat},{lon});
  way["amenity"="hospital"](around:{radius_m},{lat},{lon});
  relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
  node["healthcare"="hospital"](around:{radius_m},{lat},{lon});
  way["healthcare"="hospital"](around:{radius_m},{lat},{lon});
  relation["healthcare"="hospital"](around:{radius_m},{lat},{lon});
);
out center;
"""

async def _fetch_overpass(lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    # ... (kode Anda tidak berubah)
    headers = {
        "User-Agent": "ArterythmBackend/1.1 (+dev)",
        "Content-Type": "text/plain; charset=utf-8",
    }
    query = _build_overpass_query(lat, lon, radius_m)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, headers=headers, http2=True) as client:
        r = await client.post(OVERPASS_URL, content=query)
        r.raise_for_status()
        return r.json()

def _to_hospitals(data: Dict[str, Any], user_lat: float, user_lon: float) -> List[Hospital]:
    # ... (kode Anda tidak berubah)
    user_loc = (user_lat, user_lon)
    hospitals: List[Hospital] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        name = tags.get("name")
        if not name:
            continue
        if "center" in el and el["center"]:
            lat, lon = el["center"]["lat"], el_center["lon"]
        else:
            lat, lon = el.get("lat"), el.get("lon")
        if lat is None or lon is None:
            continue
        dist_km = geodesic(user_loc, (lat, lon)).kilometers
        hospitals.append(
            Hospital(
                id=int(el.get("id", 0)),
                name=name,
                latitude=lat,
                longitude=lon,
                distance_km=round(float(dist_km), 3),
            )
        )
    hospitals.sort(key=lambda x: x.distance_km)
    return hospitals

# =========================
# Endpoints (Rumah Sakit - Tidak Berubah)
# =========================
@app.get("/", tags=["meta"])
async def root():
    return {"status": "Arterythm Backend is running!", "version": app.version}

@app.get("/health", tags=["meta"])
async def health():
    return {"ok": True}

@app.get("/api/v1/nearby-hospitals", response_model=HospitalsResponse, tags=["hospitals"])
async def get_nearby_hospitals(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_m: int = Query(DEFAULT_RADIUS_M, ge=100, le=MAX_RADIUS_M),
):
    try:
        data = await _fetch_overpass(lat, lon, radius_m)
        hospitals = _to_hospitals(data, lat, lon)
        return HospitalsResponse(count=len(hospitals), radius_m=radius_m, hospitals=hospitals)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Gagal menghubungi layanan peta: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Overpass membalas {e.response.status_code}: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan di server: {e}")

@app.post("/api/v1/nearby-hospitals", response_model=HospitalsResponse, tags=["hospitals"])
async def post_nearby_hospitals(body: LocationData):
    return await get_nearby_hospitals(lat=body.lat, lon=body.lon, radius_m=body.radius_m)

# =========================
# Endpoint Prediksi ML (BARU)
# =========================
@app.post("/api/v1/predict-afib", response_model=PpgPredictionResponse, tags=["ml-prediction"])
async def predict_afib(body: PpgPredictionRequest):
    """
    Menerima data mentah PPG, menjalankan pipeline preprocessing dan 
    model prediksi, lalu mengembalikan hasil diagnosis.
    """
    if ml_models.get('afib_model') is None:
        raise HTTPException(status_code=503, detail="Model prediksi sedang tidak tersedia. Coba lagi nanti.")
    
    try:
        # Jalankan fungsi ML yang berat di thread pool terpisah
        result = await run_in_threadpool(
            _run_prediction_pipeline, 
            body.ppg_data
        )
        return PpgPredictionResponse(**result)
    except Exception as e:
        # Tangani error yang mungkin terjadi selama pipeline
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {e}")

# =========================
# Run Server (Tidak Berubah)
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)