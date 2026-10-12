# Program Deteksi, Pelacakan, dan Penilaian Aktivitas Siswa Berbasis YOLO26

Paket ini berisi source code dan konfigurasi untuk pemrosesan aktivitas di ruang kelas. Alur utama menggabungkan deteksi objek, pelacakan multiobjek, asosiasi identitas berbasis posisi kursi tetap, agregasi temporal, serta pengambilan keputusan menggunakan Fuzzy AHP dan Weighted Product.

## Fitur utama

1. Deteksi dan pelacakan aktivitas dengan model YOLO26-M dan konfigurasi BoT-SORT/ReID.
2. Asosiasi fragmen trajectory ke identitas siswa berdasarkan posisi kursi tetap.
3. Agregasi aktivitas per siswa menggunakan durasi atau frekuensi.
4. Perhitungan bobot Fuzzy AHP, skor Weighted Product, pemeringkatan, dan analisis sensitivitas.
5. Audit distribusi kelas, evaluasi detector, rekonsiliasi identitas, serta pemeriksaan konsistensi posisi kursi.
6. Laporan keluaran dan informasi reproducibility.

## Prinsip identitas

`track_id` merupakan identitas trajectory yang diberikan tracker, bukan identitas siswa. Pemeringkatan akhir dilakukan setelah trajectory dihubungkan dengan student-level identity melalui asosiasi posisi kursi tetap dan rekonsiliasi fragmen. Konfigurasi proyek menargetkan 21 entitas siswa untuk rekaman/ruang kelas yang dipakai dalam eksperimen ini.

## Struktur paket

- `run_pipeline.py`: entry point pemrosesan utama.
- `src/detect_track.py`: inferensi dan pelacakan opsional dari video.
- `src/identity.py`: asosiasi identitas siswa dan penggabungan fragmen trajectory.
- `src/analytics.py`: agregasi temporal, Fuzzy AHP, Weighted Product, sensitivitas, dan laporan.
- `validation/`: modul pemeriksaan distribusi data, evaluasi detector, rekonsiliasi identitas, dan konsistensi posisi kursi.
- `tests/`: pengujian fungsi inti dan pengujian modul validasi.
- `config.yaml`, `data.yaml`, `botsort_reid.yaml`: konfigurasi program dan dataset.
- `data/student_seats.csv`: peta posisi kursi yang disesuaikan untuk ruang/rekaman tertentu.

## Persyaratan

Python 3.10 atau 3.11 direkomendasikan. Dependency dasar tercantum di `requirements.txt`. Untuk pelacakan menggunakan GPU NVIDIA, pasang build PyTorch yang sesuai dengan versi driver/CUDA perangkat sebelum menjalankan program.

## Instalasi universal

Ekstrak ZIP, buka terminal pada direktori proyek (folder yang berisi `run_pipeline.py`), lalu jalankan:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Windows CMD:        .venv\Scripts\activate.bat
# Linux/macOS:        source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Menjalankan program

Analisis downstream dari trajectory/detections yang sudah tersedia:

```bash
python run_pipeline.py --config config.yaml
```

Untuk mengutamakan trajectory yang tersedia tanpa meminta tracking ulang:

```bash
python run_pipeline.py --config config.yaml --prefer-trajectory
```

Pelacakan ulang dari video hanya bila memang dibutuhkan:

```bash
python run_pipeline.py --config config.yaml --force-track
```

Pemeriksaan data dan keluaran program:

```bash
python validation/run_validation_suite.py --config validation/config.yaml
```

Jalankan tes otomatis:

```bash
python -m pytest -q tests validation/tests
```

Windows memiliki launcher `.bat` yang setara. Perintah Python di atas dapat dipakai lintas platform selama dependency tersedia.

## Input eksternal

Untuk menjalankan seluruh fungsi, beberapa berkas harus disediakan secara terpisah karena ukuran dan sifatnya yang spesifik terhadap eksperimen: checkpoint model `weights/best.pt`, video, dataset YOLO, dan/atau `data/trajectory_botsort.jsonl` atau `output/detections.csv`. Gunakan path relatif pada konfigurasi untuk memudahkan pemindahan proyek.

## Keluaran

Keluaran disimpan di `output/` dan umumnya mencakup pemetaan identitas, ringkasan trajectory, matriks aktivitas siswa, peringkat Weighted Product, hasil analisis sensitivitas, log validasi, dan laporan Excel. Daftar tepatnya bergantung pada mode pemrosesan dan input yang tersedia.

## Batas penggunaan

- Audit distribusi/manifest oversampling tidak berarti dataset sudah diubah atau model dilatih ulang.
- Evaluasi detector menggunakan checkpoint yang sudah ada; pemanggilan evaluasi tidak sama dengan training.
- Metrik MOT seperti HOTA, IDF1, dan MOTA hanya dapat dihitung jika tersedia ground truth trajectory yang sesuai.
- Peta kursi bersifat spesifik terhadap geometri kamera/ruang. Kalibrasikan ulang sebelum dipakai pada lokasi lain.
- Jangan memasukkan credential, kata sandi, token, video, atau data siswa yang tidak diperlukan ke dalam arsip publik.
