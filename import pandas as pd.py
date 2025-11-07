import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
import os

# Aktifkan tqdm di pandas
tqdm.pandas()

# ======== 1. BACA FILE INPUT ========
df = pd.read_excel("C:/Users/USER1/Downloads/Data Latitude dan Longitude Lampung.xlsx", header=None)
df.columns = ['latitude', 'longitude']


# ======== 2. KONFIGURASI GEOPY ========
geolocator = Nominatim(user_agent="alamat_batch_geocoder_2025 (emailkamu@gmail.com)")
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=2, error_wait_seconds=2.0)


# ======== 3. FUNGSI PENGAMBILAN ALAMAT ========
def get_location(lat, lon):
    try:
        location = geocode((lat, lon), language='id')
        if location and location.raw and 'address' in location.raw:
            address = location.raw['address']
            return pd.Series({
                'kelurahan': address.get('village') or address.get('suburb') or "data tidak valid",
                'kecamatan': address.get('county') or address.get('district') or "data tidak valid",
                'kota_kabupaten': address.get('city') or address.get('town') or address.get('municipality') or address.get('regency') or "data tidak valid",
                'provinsi': address.get('state') or "data tidak valid"
            })
        else:
            return pd.Series({
                'kelurahan': "data tidak valid",
                'kecamatan': "data tidak valid",
                'kota_kabupaten': "data tidak valid",
                'provinsi': "data tidak valid"
            })
    except Exception as e:
        print(f"⚠️ Gagal geocode untuk ({lat}, {lon}): {e}")
        return pd.Series({
            'kelurahan': "data tidak valid",
            'kecamatan': "data tidak valid",
            'kota_kabupaten': "data tidak valid",
            'provinsi': "data tidak valid"
        })


# ======== 4. PROSES DENGAN PROGRESS BAR ========
location_data = df.progress_apply(lambda row: get_location(row['latitude'], row['longitude']), axis=1)


# ======== 5. SIMPAN HASIL KE EXCEL ========
# Pastikan direktori output ada
output_dir = r"D:\data latitude dan longitude"
os.makedirs(output_dir, exist_ok=True)

# Path hasil akhir
output_path = os.path.join(output_dir, "hasil_geocoding.xlsx")

# Gabungkan data dan simpan
result_df = pd.concat([df, location_data], axis=1)
result_df.to_excel(output_path, index=False)

print(f"\n✅ Proses selesai! File hasil disimpan di:\n{output_path}")