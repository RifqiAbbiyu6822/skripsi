# PTB-XL Preprocessing Pipeline (On-the-fly, Multi-Label)

Pipeline preprocessing untuk klasifikasi multi-label ECG memakai PTB-XL, dengan 5
target diagnostic superclass:

```
NORM, MI, STTC, CD, HYP
```

Tahap ini **hanya preprocessing + dataloader**. Model SE-ConvNeXt1D dan training
loop menyusul di tahap berikutnya sesuai permintaan.

## 1. Struktur folder yang dibutuhkan

Download dataset resmi dari PhysioNet lalu extract, sehingga strukturnya:

```
ptb-xl/
├── ptbxl_database.csv
├── scp_statements.csv
├── records100/            # sinyal 100 Hz (10 detik -> 1000 sample/lead)
│   ├── 00000/
│   ├── 01000/
│   └── ...
└── records500/            # sinyal 500 Hz (10 detik -> 5000 sample/lead)
    ├── 00000/
    └── ...
```

`ptbxl_root` di `config.py` diarahkan ke folder `ptb-xl/` ini.

## 2. Install dependency

```bash
pip install wfdb pandas numpy scipy torch scikit-learn
```

## 3. Struktur kode

```
ptbxl_pipeline/
├── config.py          # semua parameter (sampling rate, filter, augmentasi, split)
├── labels.py          # load ptbxl_database.csv + scp_statements.csv -> multi-hot label
├── signal_ops.py       # bandpass filter, z-score normalize, fix-length, augmentasi
├── dataset.py          # PyTorch Dataset (on-the-fly) + WeightedRandomSampler + DataLoader
├── utils.py            # set_seed, validate_dataset (sanity check)
├── example_usage.py    # contoh menjalankan pipeline end-to-end
└── __init__.py
```

## 4. Cara pakai cepat

```bash
python example_usage.py --ptbxl_root /path/ke/ptb-xl --sampling_rate 100 --batch_size 32
```

Output akan menampilkan:
- distribusi kelas di train/val/test
- validasi shape & NaN
- shape 1 batch (`signals`, `labels`)

Atau dipakai langsung dari kode:

```python
from ptbxl_pipeline import PTBXLConfig, load_ptbxl_metadata, split_by_fold, build_dataloaders, set_seed

config = PTBXLConfig(ptbxl_root="/path/ke/ptb-xl", sampling_rate=100)
set_seed(config.seed)

df = load_ptbxl_metadata(config.ptbxl_root, config.target_classes)
train_df, val_df, test_df = split_by_fold(df, config.test_fold, config.val_fold)

train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, config)

for signals, labels in train_loader:
    # signals: (batch, 12, target_length)
    # labels : (batch, 5)  -- multi-hot, boleh lebih dari satu 1 per baris
    break
```

## 5. Kenapa disebut "on-the-fly"

Tidak ada file hasil preprocessing (filtered / normalized / augmented) yang
ditulis ke disk. Setiap kali `DataLoader` mengambil sample:

1. Baca sinyal mentah `.dat/.hea` lewat `wfdb.rdsamp`
2. Bandpass filter 0.5–40 Hz (buang baseline wander & noise)
3. Crop/pad ke panjang tetap (`target_length`, default 1000 sample @100Hz)
4. Z-score normalize per lead
5. (khusus train) augmentasi acak: gaussian noise, random scaling, random time-shift
6. Sanitasi NaN/Inf terakhir sebelum dikembalikan sebagai tensor

Artinya dataset di disk tetap dataset mentah PTB-XL apa adanya, hemat storage,
dan setiap epoch training melihat variasi augmentasi yang berbeda-beda.

## 6. Bagaimana "stabilitas" dijaga

| Sumber ketidakstabilan | Solusi di pipeline ini |
|---|---|
| Panjang sinyal berbeda-beda / rekaman gagal load penuh | `fix_length()` memaksa semua sample punya shape sama persis |
| Skala amplitudo berbeda antar pasien/alat | `zscore_normalize()` per lead |
| NaN/Inf pada beberapa rekaman PTB-XL | `sanitize()` dipanggil sebelum & sesudah filter/augmentasi |
| Kelas timpang (NORM jauh lebih banyak dari HYP) | `WeightedRandomSampler` via `compute_sample_weights()`, bobot per sample dari inverse frekuensi tiap label aktif |
| Split data bocor/tidak representatif | Split memakai `strat_fold` resmi PTB-XL (1-8 train, 9 val, 10 test) yang sudah di-stratifikasi oleh pembuat dataset |
| Reproduksibilitas run-to-run | `set_seed()` dipanggil di awal |
| Data rusak lolos tanpa terdeteksi | `validate_dataset()` sanity-check shape/NaN/label kosong sebelum training dimulai |

## 7. Catatan multi-label

PTB-XL secara alami multi-label: satu rekaman ECG bisa punya lebih dari satu
diagnostic superclass sekaligus (mis. `MI` + `STTC` bersamaan). Karena itu:

- Label disimpan sebagai vektor multi-hot 5 dimensi, BUKAN one-hot / softmax class.
- Rekaman yang sama sekali tidak punya satupun dari 5 kelas target **dibuang**
  otomatis di `load_ptbxl_metadata()` supaya tidak ada label all-zero yang
  membingungkan loss function nanti (BCEWithLogitsLoss saat training).

## 8. Model: SE-ConvNeXt1D

File tambahan untuk model & training:

```
model.py     # SEBlock1D, LayerNorm1d, DropPath, ConvNeXt1DBlock, ConvNeXt1DSE, build_model()
metrics.py   # macro/micro F1, macro AUROC, F1 per kelas (multi-label)
engine.py    # train_one_epoch(), evaluate()
train.py     # script orkestrasi: data -> model -> training loop -> checkpoint -> test
```

Arsitektur `ConvNeXt1DSE`:

```
Input (batch, 12, 1000)
  -> Stem: Conv1d (kernel besar, stride 4) + LayerNorm      -> downsampling awal
  -> Stage 1..4: N x [ConvNeXt1DBlock + SEBlock1D]           -> tiap stage diikuti
       (depthwise conv -> LayerNorm -> Linear 4x -> GELU     downsampling ke stage
        -> Linear -> layer scale -> SE -> residual)           berikutnya
  -> Global Average Pooling -> LayerNorm -> Linear(dim, 5)
  -> logits (batch, 5)   -- pakai sigmoid + BCEWithLogitsLoss saat training
```

3 preset ukuran (parameter di `build_model(variant=...)`):

| variant | depths | dims | ~jumlah parameter |
|---|---|---|---|
| `nano` | (2,2,4,2) | (32,64,128,256) | ~1.8 juta |
| `tiny` | (3,3,9,3) | (64,128,256,512) | ~12.1 juta |
| `small`| (3,3,27,3)| (96,192,384,768) | lebih besar, untuk data lebih banyak / GPU kuat |

Sudah diuji forward + backward pass dengan data dummy — output shape `(batch, 5)` dan gradien mengalir normal di ketiga preset.

## 9. Training

```bash
python train.py --ptbxl_root "C:\path\ke\ptb-xl" --sampling_rate 100 \
    --model_variant tiny --epochs 50 --batch_size 32 --lr 3e-4 \
    --output_dir ./checkpoints
```

Argumen penting:

| Argumen | Default | Keterangan |
|---|---|---|
| `--model_variant` | tiny | nano / tiny / small |
| `--epochs` | 50 | jumlah epoch maksimum |
| `--lr` | 3e-4 | learning rate awal (AdamW + cosine annealing) |
| `--weight_decay` | 0.05 | regularisasi AdamW |
| `--drop_path_rate`| 0.1 | stochastic depth, makin besar makin kuat regularisasinya |
| `--patience` | 10 | early stopping: berhenti jika macro-F1 val tidak naik N epoch |
| `--threshold` | 0.5 | ambang sigmoid untuk konversi probabilitas -> label biner |
| `--output_dir` | ./checkpoints | tempat simpan `best_model.pt`, `last_model.pt`, `history.json`, `test_metrics.json` |

Yang dilakukan `train.py`:

1. Load data lewat `ptbxl_pipeline` (on-the-fly preprocessing, sudah dibahas di atas)
2. Bangun model `ConvNeXt1DSE` sesuai `--model_variant`
3. Loss `BCEWithLogitsLoss` (cocok untuk multi-label) + optimizer `AdamW` + scheduler cosine annealing
4. Mixed precision (AMP) otomatis aktif kalau ada GPU CUDA, mempercepat training tanpa mengubah kode
5. Tiap epoch: training -> evaluasi di val set -> hitung macro-F1, micro-F1, macro-AUROC, F1 per kelas
6. Simpan checkpoint terbaik berdasarkan macro-F1 validasi (`best_model.pt`) dan checkpoint terakhir (`last_model.pt`)
7. Early stopping kalau macro-F1 val stagnan selama `--patience` epoch
8. Setelah training selesai, load `best_model.pt` dan evaluasi final di test set, hasil disimpan ke `test_metrics.json`

Beri tahu saya kalau mau lanjut ke tahap inferensi (load checkpoint untuk prediksi ECG baru) atau tuning hyperparameter.

## 10. (Opsional) Menambah data kelas minoritas dari Chapman-Shaoxing

Kalau kelas tertentu (mis. HYP) tetap sulit walau sudah pakai `pos_weight`,
kamu bisa menambah data training dari dataset eksternal **Chapman-Shaoxing**
(format sinyal `.mat` + `.hea`, umum dipakai di PhysioNet Challenge).

File tambahan:

```
chapman_labels.py    # load dx_record_details.csv -> multi-hot label (format sama dgn PTB-XL)
chapman_dataset.py    # baca .mat/.hea, parse header manual, resample fs, samakan urutan lead
```

**Penting — supaya evaluasi tetap adil:**
- Data Chapman **hanya masuk ke train set**, tidak pernah ke val/test.
- val/test tetap 100% dari PTB-XL, sehingga hasil `test_metrics.json` tetap
  bisa dibandingkan apple-to-apple dengan run-run sebelumnya.
- Preprocessing (bandpass filter, z-score normalize, fix-length) memakai
  fungsi yang **sama persis** dengan PTB-XL, plus satu langkah `resample_signal()`
  supaya sampling rate (mis. Chapman 500Hz) disamakan ke `config.sampling_rate`
  PTB-XL (mis. 100Hz). Urutan lead direorder otomatis ke urutan standar
  `I, II, III, aVR, aVL, aVF, V1..V6` supaya channel tidak tertukar walau
  urutan di file aslinya berbeda.

Cara pakai — tinggal tambah 4 argumen di `train.py`:

```bash
python train.py --ptbxl_root "." --sampling_rate 100 \
    --chapman_root "C:\path\ke\folder\chapman" \
    --chapman_csv "C:\path\ke\dx_record_details.csv" \
    --boost_class HYP \
    --model_variant tiny --epochs 50 --batch_size 32 \
    --output_dir ./checkpoints_boosted
```

`--boost_target` opsional — kalau tidak diisi, dihitung **otomatis** supaya
jumlah `--boost_class` di train set sejajar dengan rata-rata jumlah kelas
lain (dibatasi oleh jumlah data yang benar-benar tersedia di Chapman).
Bisa diisi manual kalau mau kontrol jumlah pasti, mis. `--boost_target 2000`.

Sudah diuji end-to-end (loader header, reorder lead, resample 500Hz->100Hz,
penggabungan dataset, training loop) memakai data sintetis — semua berjalan
tanpa error sebelum dipakai ke data asli.
