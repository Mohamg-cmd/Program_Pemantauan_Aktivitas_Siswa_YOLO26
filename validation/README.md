# Pemeriksaan Kualitas Program

Direktori ini berisi pemeriksaan data dan keluaran yang melengkapi pipeline utama.

- `run_validation_suite.py`: menjalankan rangkaian pemeriksaan secara berurutan.
- `src/class_imbalance.py`: menghitung distribusi kelas dan menyiapkan manifest sampling bila dikonfigurasi.
- `src/detector_eval.py`: memeriksa artefak evaluasi yang tersedia atau menjalankan evaluasi checkpoint secara eksplisit.
- `src/identity_reconcile.py`: memeriksa kelengkapan dan status assignment identitas siswa.
- `src/seat_purity.py`: menghitung konsistensi posisi kursi, assignment coverage, dan metrik terkait.
- `tests/test_validation_suite.py`: menguji perhitungan rekonsiliasi identitas dan seat purity pada data uji terkontrol.

Jalankan dari direktori root proyek dengan:

```bash
python validation/run_validation_suite.py --config validation/config.yaml
```

Rangkaian ini tidak melakukan training ulang. Evaluasi detector baru dijalankan apabila diminta secara eksplisit. Untuk semua metrik pelacakan berbasis ground truth, sediakan ground truth trajectory yang sesuai.
