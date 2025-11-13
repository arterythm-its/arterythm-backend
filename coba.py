# main.py
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, confloat
import httpx
from geopy.distance import geodesic
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool

# --- Impor untuk ML ---
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import MinMaxScaler

# =========================
# Konfigurasi
# =========================
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
DEFAULT_RADIUS_M = int(os.getenv("DEFAULT_RADIUS_M", "5000"))
MAX_RADIUS_M = int(os.getenv("MAX_RADIUS_M", "20000"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "15"))

# --- Konfigurasi ML ---
MODEL_PATH = "models/model_05102025_225139.h5" #
ML_WINDOW_SIZE = 12 # Diambil dari 'sampling_rate' di model.py
ml_models = {} # Dictionary untuk menyimpan model yang sudah di-load

# =========================
# Lifespan (Load Model saat Startup)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Aksi yang dijalankan saat server startup
    print("Memuat model Keras...")
    try:
        model = load_model(MODEL_PATH) #
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #
        ml_models['afib_model'] = model
        print(f"Model {MODEL_PATH} berhasil dimuat.")
    except Exception as e:
        print(f"ERROR: Gagal memuat model di {MODEL_PATH}. Detail: {e}")
        ml_models['afib_model'] = None
    
    yield
    
    # Aksi yang dijalankan saat server shutdown
    print("Membersihkan resource model...")
    ml_models.clear()

app = FastAPI(
    title="Arterythm Backend (Combined)", 
    version="1.1.0",
    lifespan=lifespan # Daftarkan lifespan event
)

# CORS (Sama seperti sebelumnya)
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
class PpgData(BaseModel):
    """Menerima data PPG mentah dari Flutter"""
    data: List[float] = Field(..., example=[102.0, 103.0, 105.0, ...])

class PredictionResponse(BaseModel):
    """Mengirim hasil prediksi kembali ke Flutter"""
    diagnosis: str
    probability: float
    afib_windows: int
    total_windows: int

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
    user_loc = (user_lat, user_lon)
    hospitals: List[Hospital] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        name = tags.get("name")
        if not name:
            continue

        # Ambil titik pusat
        if "center" in el and el["center"]:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
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

    # Urutkan terdekat
    hospitals.sort(key=lambda x: x.distance_km)
    return hospitals

# =========================
# Util Prediksi ML (BARU)
# =========================
def run_prediction_pipeline(ppg_signal_raw: List[float]) -> Dict:
    """
    Fungsi Synchronous yang menjalankan pipeline ML.
    Mengambil logika dari model.py dan menerapkannya pada data input.
    """
    
    print(f"Menerima {len(ppg_signal_raw)} sampel PPG untuk diprediksi.")
    
    # 1. Dapatkan model yang sudah di-load
    model = ml_models.get('afib_model')
    if model is None:
        raise HTTPException(status_code=503, detail="Model prediksi sedang tidak tersedia.")
        
    # 2. Preprocessing (diadaptasi dari model.py)
    # Ubah list input menjadi DataFrame
    df = pd.DataFrame(ppg_signal_raw, columns=['ppg']) #
    df.dropna(axis=1, inplace=True) #
    
    # Scaling (Replikasi logika model.py, meskipun fit di data baru tidak ideal)
    scaler = MinMaxScaler() #
    scaler.fit(df) #
    df_scaled = scaler.transform(df) #
    df_scaled = pd.DataFrame(df_scaled) #
    
    ppg_signal = df_scaled.iloc[:, 0].values #

    # 3. Windowing (diadaptasi dari model.py)
    window_size = ML_WINDOW_SIZE #
    n_windows = len(ppg_signal) // window_size #
    
    if n_windows == 0:
        raise Exception(f"Data tidak cukup untuk membuat window. Diperlukan setidaknya {ML_WINDOW_SIZE} sampel.")

    windows_list = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window = ppg_signal[start:end]
        windows_list.append(window) #
    
    # Konversi ke format yang diterima model (DataFrame di skrip Anda)
    x_predict = pd.DataFrame(windows_list) #

    # 4. Predict
    print(f"Menjalankan prediksi pada {n_windows} window...")
    y_pred_prob = model.predict(x_predict).ravel() #
    y_pred = (y_pred_prob >= 0.5).astype(int) #
    
    # 5. Buat Hasil Agregat
    total_windows = len(y_pred)
    afib_windows = int(np.sum(y_pred))
    mean_prob = float(np.mean(y_pred_prob))
    
    if afib_windows > (total_windows * 0.1): # Contoh: Jika > 10% window adalah AFib
        diagnosis = "Beresiko AFib"
        print(f"Hasil: {diagnosis} silahkan konsultasi dengan dokter")
    else:
        diagnosis = "Tidak Beresiko AFib"
        print(f"Hasil: {diagnosis}")
        
    # print(f"Hasil: {diagnosis} (AFib Windows: {afib_windows}/{total_windows})")
    

    
    return {
        "diagnosis": diagnosis,
        "probability": mean_prob,
        "afib_windows": afib_windows,
        "total_windows": total_windows
    }

# =========================
# Endpoints (Rumah Sakit - Tidak Berubah)
# =========================
@app.get("/", tags=["meta"])
async def root():
    return {"status": "Arterythm Backend (Combined) is running!", "version": app.version}

@app.get("/health", tags=["meta"])
async def health():
    return {"ok": True}

@app.get("/api/v1/nearby-hospitals", response_model=HospitalsResponse, tags=["hospitals"])
async def get_nearby_hospitals(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_m: int = Query(DEFAULT_RADIUS_M, ge=100, le=MAX_RADIUS_M),
):
    # ... (kode Anda tidak berubah)
    try:
        data = await _fetch_overpass(lat, lon, radius_m)
        hospitals = _to_hospitals(data, lat, lon)
        return HospitalsResponse(count=len(hospitals), radius_m=radius_m, hospitals=hospitals)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {e}")


@app.post("/api/v1/nearby-hospitals", response_model=HospitalsResponse, tags=["hospitals"])
async def post_nearby_hospitals(body: LocationData):
    return await get_nearby_hospitals(lat=body.lat, lon=body.lon, radius_m=body.radius_m) #

# =========================
# Endpoint Prediksi ML (BARU)
# =========================
@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["ml-prediction"])
async def predict_afib(request: PpgData):
    """
    Menerima data mentah PPG, menjalankan pipeline preprocessing dan 
    model prediksi, lalu mengembalikan hasil diagnosis.
    """
    if ml_models.get('afib_model') is None:
        raise HTTPException(status_code=503, detail="Model prediksi sedang tidak tersedia. Coba lagi nanti.")
    
    try:
        # Jalankan fungsi ML yang berat di thread pool terpisah
        result_dict = await run_in_threadpool(
            run_prediction_pipeline, 
            request.data
        )
        return PredictionResponse(**result_dict)
    except Exception as e:
        # Tangani error dari pipeline (misal data tidak cukup)
        raise HTTPException(status_code=400, detail=f"Error selama prediksi: {e}")

# =========================
# Run Server (Tidak Berubah)
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
