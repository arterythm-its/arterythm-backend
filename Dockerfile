# Langkah 1: Tentukan Fondasi (Base Image)
# Kita gunakan image Python 3.9 versi "slim" yang ringan
FROM python:3.9-slim

# Langkah 2: Tentukan Folder Kerja di Dalam Kontainer
# Semua perintah selanjutnya akan dijalankan dari folder /app
WORKDIR /app

# Langkah 3: Salin "Daftar Belanjaan" Library
# Ini disalin terpisah untuk optimasi cache Docker
COPY requirements.txt requirements.txt

# Langkah 4: Instal Semua Library
# --no-cache-dir membuat image lebih kecil
RUN pip install --no-cache-dir -r requirements.txt

# Langkah 5: Salin Semua Sisa Kode Anda
# Ini termasuk main.py, folder models/, dll.
COPY . .

# Langkah 6: Tentukan Perintah untuk Menjalankan Server
# Render secara otomatis menyediakan variabel $PORT
# Kita TIDAK menggunakan --reload di produksi
CMD ["uvicorn", "coba:app", "--host", "0.0.0.0", "--port", "$PORT"]