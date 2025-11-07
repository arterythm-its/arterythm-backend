# main.py
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, confloat
import httpx
from geopy.distance import geodesic

# =========================
# Konfigurasi (env override)
# =========================
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
DEFAULT_RADIUS_M = int(os.getenv("DEFAULT_RADIUS_M", "5000"))
MAX_RADIUS_M = int(os.getenv("MAX_RADIUS_M", "20000"))
HTTP_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "15"))

app = FastAPI(title="Arterythm Backend", version="1.1.0")

# CORS dev-friendly (batasi di production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti dengan daftar origin di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Skema request/response
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
# Util Overpass
# =========================
def _build_overpass_query(lat: float, lon: float, radius_m: int) -> str:
    # Tambah "healthcare=hospital" selain "amenity=hospital" agar cakupan lebih luas
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
# Endpoints
# =========================

@app.get("/", tags=["meta"])
async def root():
    return {"status": "Arterythm Backend is running!", "version": app.version}

@app.get("/health", tags=["meta"])
async def health():
    return {"ok": True}

# GET: /api/v1/nearby-hospitals?lat=...&lon=...&radius_m=...
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

# POST: kompatibel dengan versi lama (body JSON)
@app.post("/api/v1/nearby-hospitals", response_model=HospitalsResponse, tags=["hospitals"])
async def post_nearby_hospitals(body: LocationData):
    return await get_nearby_hospitals(lat=body.lat, lon=body.lon, radius_m=body.radius_m)

# Jalankan langsung: bind ke 0.0.0.0 agar device fisik bisa akses via IP LAN
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
