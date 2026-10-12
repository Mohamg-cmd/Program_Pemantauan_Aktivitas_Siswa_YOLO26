# Daftar Isi Paket Program

## Source code inti
- `run_pipeline.py` - pengendali alur utama.
- `src/detect_track.py` - deteksi dan pelacakan opsional dari video.
- `src/identity.py` - asosiasi fixed-seat identity dan rekonsiliasi fragmen trajectory.
- `src/analytics.py` - agregasi temporal, Fuzzy AHP, Weighted Product, analisis sensitivitas, dan laporan.

## Validasi dan tes
- `validation/run_validation_suite.py` - menjalankan rangkaian pemeriksaan.
- `validation/src/class_imbalance.py` - audit distribusi kelas dan pembuatan manifest sampling.
- `validation/src/detector_eval.py` - inventaris dan/atau evaluasi detector dari checkpoint tersedia.
- `validation/src/identity_reconcile.py` - rekonsiliasi assignment identity.
- `validation/src/seat_purity.py` - pemeriksaan konsistensi seat identity dan assignment coverage.
- `tests/` dan `validation/tests/` - tes otomatis.

## Konfigurasi dan launcher
- `config.yaml` - konfigurasi pipeline.
- `data.yaml` - konfigurasi dataset YOLO.
- `botsort_reid.yaml` - pengaturan pelacakan dan ReID.
- `validation/config.yaml` - konfigurasi rangkaian pemeriksaan.
- `RUN_PIPELINE.bat`, `RUN_DOWNSTREAM.bat`, `RUN_RETRACK.bat`, `RUN_VALIDATION_SUITE.bat`, `RUN_TESTS.bat` - launcher Windows.

## Input yang tidak disertakan
ZIP source code ini tidak menyertakan checkpoint besar, rekaman video, dataset penuh, atau hasil deteksi/trajectory berukuran besar. Berkas tersebut perlu disediakan secara terpisah sesuai konfigurasi. `data/student_seats.csv` adalah contoh peta kursi yang terkait dengan tata letak eksperimen dan perlu diperiksa sebelum digunakan pada ruang lain.

## Catatan metodologis
- `track_id` adalah ID trajectory, bukan ID siswa.
- Analisis utama mengutamakan entitas siswa setelah fixed-seat identity association.
- Manifest oversampling yang dihasilkan dalam mode audit tidak berarti retraining dilakukan.
- Metrik MOT berbasis ground truth tidak boleh diklaim tanpa ground truth trajectory yang sesuai.
