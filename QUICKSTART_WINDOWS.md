# Panduan Cepat Windows

## 1. Siapkan input

Letakkan input yang diperlukan di dalam struktur paket. Jalur analisis yang memanfaatkan hasil lama membutuhkan `data/trajectory_botsort.jsonl` atau `output/detections.csv`. Pelacakan dari awal membutuhkan `weights/best.pt` dan video pada lokasi yang ditentukan di `config.yaml`. Audit dataset memerlukan folder dataset dan `data.yaml` yang menunjuk ke lokasi tersebut.

## 2. Buat environment dan pasang pustaka

Dari folder proyek (direktori yang memuat `run_pipeline.py`), jalankan:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika memakai Windows PowerShell, aktivasi environment menggunakan `.\.venv\Scripts\Activate.ps1`. Untuk GPU NVIDIA, pastikan instalasi PyTorch sesuai perangkat.

## 3. Jalankan analisis

```bat
python run_pipeline.py --config config.yaml
```

## 4. Jalankan pemeriksaan data dan keluaran

```bat
python validation\run_validation_suite.py --config validation\config.yaml
```

## 5. Jalankan tes otomatis

```bat
python -m pytest -q tests validation\tests
```

## 6. Pelacakan ulang opsional

```bat
python run_pipeline.py --config config.yaml --force-track
```

Jalankan pelacakan ulang hanya ketika memang diperlukan. Jika trajectory/detections yang valid sudah tersedia, gunakan analisis downstream agar eksperimen tidak terulang tanpa alasan.
