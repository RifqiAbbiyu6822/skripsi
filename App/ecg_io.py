"""Pembacaan berkas EKG: header .hea (wajib), .dat (WFDB), dan .mat. Semua sinyal dikonversi ke mV."""
import os
import shutil
import tempfile

import numpy as np

from checks import EcgInputError, make_check as _chk

UNIT_TO_MV = {"mv": 1.0, "uv": 1e-3, "µv": 1e-3, "μv": 1e-3, "microv": 1e-3, "v": 1e3, "nv": 1e-6}


def unit_factor_mv(unit):
    """Faktor pengali ke mV. None bila satuan tidak dikenali; 1.0 bila satuan tidak ditulis (default WFDB = mV)."""
    if unit is None:
        return 1.0
    return UNIT_TO_MV.get(unit.strip().lower().replace(" ", ""))


def _parse_gain_field(field):
    """'gain(baseline)/unit' -> (gain, baseline, unit); elemen yang tidak ada -> None."""
    gain = baseline = unit = None
    f = field
    if "/" in f:
        f, unit = f.split("/", 1)
        unit = unit.strip() or None
    if "(" in f and f.endswith(")"):
        g, b = f[:-1].split("(", 1)
        gain = float(g) if g else None
        baseline = float(b) if b else None
    else:
        gain = float(f) if f else None
    return gain, baseline, unit


def parse_hea_meta(hea_text):
    """Parse header WFDB. Tidak pernah melempar error; cek meta['header_ok'] dan jalankan check_header()."""
    lines = [l.strip() for l in hea_text.replace("\r", "").split("\n")
             if l.strip() and not l.strip().startswith("#")]
    meta = {"record": "", "n_sig": None, "fs": None, "n_samples": None, "sigs": [], "header_ok": False}
    if not lines:
        return meta
    tok = lines[0].split()
    try:
        meta["record"] = os.path.basename(tok[0].split("/")[0])
        meta["n_sig"] = int(tok[1])
        meta["fs"] = float(tok[2].split("/")[0])
        if len(tok) > 3:
            try:
                meta["n_samples"] = int(tok[3])
            except ValueError:
                pass
        meta["header_ok"] = meta["n_sig"] > 0 and meta["fs"] > 0
    except (IndexError, ValueError):
        return meta

    for i, line in enumerate(lines[1:1 + meta["n_sig"]]):
        t = line.split()
        s = {"file": t[0] if t else "", "fmt": t[1] if len(t) > 1 else "",
             "gain": 200.0, "gain_given": False, "baseline": 0.0,
             "unit": "mV", "unit_given": False, "factor": 1.0,
             "name": f"Lead {i + 1}"}
        adc_zero = None
        if len(t) > 4:
            try:
                adc_zero = float(t[4])
            except ValueError:
                pass
        g = b = u = None
        if len(t) > 2:
            try:
                g, b, u = _parse_gain_field(t[2])
            except ValueError:
                pass
        if g is not None and g > 0:
            s["gain"], s["gain_given"] = g, True
        s["baseline"] = b if b is not None else (adc_zero if adc_zero is not None else 0.0)
        if u is not None:
            s["unit"], s["unit_given"] = u, True
            s["factor"] = unit_factor_mv(u)      # None bila tidak dikenali
        if len(t) >= 9:
            s["name"] = " ".join(t[8:])
        meta["sigs"].append(s)
    return meta


def check_header(meta):
    """Pemeriksaan isi .hea. Status 'fail' = berkas tidak dapat dipakai."""
    checks = []
    if not meta["header_ok"]:
        checks.append(_chk(
            "Berkas .hea valid", "fail",
            "Tidak ada / tidak terbaca: baris pertama header harus berisi 'nama_rekaman jumlah_sinyal "
            "frekuensi_sampling jumlah_sampel'. Tanpa itu, satuan dan skala voltase tidak dapat ditentukan."))
        return checks
    extra = f", {meta['n_samples']} sampel" if meta["n_samples"] else ""
    checks.append(_chk("Berkas .hea valid", "pass",
                       f"rekaman '{meta['record']}', {meta['n_sig']} sinyal, {meta['fs']:g} Hz{extra}."))

    n_lines = len(meta["sigs"])
    if n_lines != meta["n_sig"]:
        checks.append(_chk("Baris sinyal di .hea lengkap", "fail",
                           f"Header menyatakan {meta['n_sig']} sinyal tetapi hanya {n_lines} baris deskripsi sinyal ditemukan."))
    else:
        checks.append(_chk("Baris sinyal di .hea lengkap", "pass", f"{n_lines} baris deskripsi sinyal."))

    no_gain = [s["name"] for s in meta["sigs"] if not s["gain_given"]]
    if no_gain:
        checks.append(_chk("Gain per sinyal tercantum di .hea", "warn",
                           f"Gain tidak tertulis/ tidak valid untuk: {', '.join(no_gain[:6])}"
                           f"{'…' if len(no_gain) > 6 else ''}. Dipakai default WFDB (200 ADU/satuan) — "
                           "voltase bisa meleset bila rekaman aslinya memakai gain lain."))
    else:
        checks.append(_chk("Gain per sinyal tercantum di .hea", "pass", "Gain terbaca untuk semua sinyal."))

    bad_units = sorted({s["unit"] for s in meta["sigs"] if s["factor"] is None})
    no_unit = [s for s in meta["sigs"] if not s["unit_given"]]
    if bad_units:
        checks.append(_chk("Satuan amplitudo adalah tegangan (mV/µV/V)", "fail",
                           f"Satuan tidak dikenali: {', '.join(bad_units)}. Sinyal EKG harus bertegangan "
                           "(mV, µV, atau V); satuan lain berarti ini bukan sinyal EKG atau header-nya keliru."))
    elif no_unit:
        checks.append(_chk("Satuan amplitudo adalah tegangan (mV/µV/V)", "warn",
                           f"Satuan tidak tertulis untuk {len(no_unit)} sinyal; dipakai default WFDB (mV)."))
    else:
        units = sorted({s["unit"] for s in meta["sigs"]})
        checks.append(_chk("Satuan amplitudo adalah tegangan (mV/µV/V)", "pass",
                           f"Satuan di header: {', '.join(units)} (dikonversi ke mV)."))
    return checks


def read_wfdb_record(hea_bytes, dat_bytes, meta):
    """Baca .hea + .dat via wfdb. Return (signal_mV (N, n_sig), fs, sig_names, checks)."""
    import wfdb

    files = sorted({os.path.basename(s["file"]) for s in meta["sigs"] if s["file"]})
    if len(files) != 1:
        raise EcgInputError([_chk(
            "Berkas sinyal .dat sesuai header", "fail",
            f"Header merujuk ke {len(files)} berkas sinyal ({', '.join(files) or '—'}); "
            "aplikasi ini hanya menerima rekaman dengan satu berkas .dat.")])

    true_name = meta["record"] or "record"
    tmp_dir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp_dir, true_name + ".hea"), "wb") as f:
            f.write(hea_bytes)
        with open(os.path.join(tmp_dir, files[0]), "wb") as f:     # nama .dat disamakan dengan yang ada di header
            f.write(dat_bytes)
        try:
            record = wfdb.rdrecord(os.path.join(tmp_dir, true_name))
        except Exception as e:
            raise EcgInputError([_chk(
                "Berkas .dat dapat dibaca WFDB", "fail",
                f"WFDB gagal membaca pasangan .hea/.dat: {e}. Pastikan .dat berasal dari rekaman yang sama "
                "dengan .hea dan formatnya sesuai header.")])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if record.p_signal is None:
        raise EcgInputError([_chk("Berkas .dat dapat dikonversi ke satuan fisik", "fail",
                                  "WFDB tidak menghasilkan sinyal fisik (format campuran/ tidak didukung).")])

    units = list(record.units) if getattr(record, "units", None) else []
    factors = []
    for i in range(record.p_signal.shape[1]):
        u = units[i] if i < len(units) else None
        f = unit_factor_mv(u)
        factors.append(1.0 if f is None else f)
    signal = record.p_signal.astype(np.float32) * np.asarray(factors, dtype=np.float32)[None, :]

    checks = [_chk("Berkas .dat dapat dibaca WFDB", "pass",
                   f"{signal.shape[0]} sampel × {signal.shape[1]} sinyal; dikonversi ke mV memakai gain/baseline/satuan dari .hea.")]
    return signal, float(record.fs), list(record.sig_name), checks


def read_mat_record(mat_bytes, meta):
    """Baca .mat memakai header .hea (wajib). Nilai .mat dianggap ADC mentah standar WFDB:
        mV = (nilai - baseline) / gain x faktor_satuan
    Return (signal_mV (N, n_sig), fs, sig_names, checks)."""
    import io as _io
    from scipy.io import loadmat

    try:
        mat_dict = loadmat(_io.BytesIO(mat_bytes))
    except Exception as e:
        raise EcgInputError([_chk(
            "Berkas .mat dapat dibuka", "fail",
            f"scipy gagal membuka berkas: {e}. Berkas mungkin bukan MATLAB .mat, rusak, atau berformat v7.3 (HDF5) — "
            "simpan ulang dengan opsi -v7 atau -v7.3 → -v7.")])

    checks = [_chk("Berkas .mat dapat dibuka", "pass", "Format MATLAB terbaca.")]

    def _describe(k, v):
        if isinstance(v, np.ndarray):
            return f"{k}: {v.dtype} {tuple(v.shape)}"
        return f"{k}: {type(v).__name__}"

    keys = [k for k in mat_dict if not k.startswith("__")]
    numeric = {k: v for k, v in mat_dict.items()
               if not k.startswith("__") and isinstance(v, np.ndarray)
               and v.dtype.kind in "iuf" and v.ndim == 2 and min(v.shape) > 1}
    if not numeric:
        raise EcgInputError(checks + [_chk(
            "Matriks sinyal numerik 2-D di dalam .mat", "fail",
            "Tidak ada: tidak ditemukan matriks numerik 2 dimensi (leads × sampel). "
            f"Isi berkas: {', '.join(_describe(k, mat_dict[k]) for k in keys) or '(kosong)'}.")])

    n_sig = meta["n_sig"]
    preferred = [k for k in ("val", "ECG", "ecg", "data", "signal", "Data", "sig") if k in numeric]
    ordered = preferred + sorted((k for k in numeric if k not in preferred),
                                 key=lambda k: -numeric[k].size)
    chosen, sig = None, None
    for k in ordered:
        arr = numeric[k]
        if arr.shape[0] == n_sig:
            chosen, sig = k, arr.T
            break
        if arr.shape[1] == n_sig:
            chosen, sig = k, arr
            break
    if sig is None:
        raise EcgInputError(checks + [_chk(
            "Dimensi matriks sesuai jumlah sinyal di .hea", "fail",
            f"Header menyatakan {n_sig} sinyal, tetapi tidak ada matriks yang salah satu dimensinya bernilai {n_sig}. "
            f"Matriks ditemukan: {', '.join(_describe(k, numeric[k]) for k in numeric)}. "
            ".hea kemungkinan bukan pasangan dari .mat ini.")])
    checks.append(_chk("Matriks sinyal sesuai jumlah sinyal di .hea", "pass",
                       f"Kunci '{chosen}' ukuran {tuple(numeric[chosen].shape)} → {sig.shape[0]} sampel × {sig.shape[1]} sinyal."))

    if meta["n_samples"] and meta["n_samples"] != sig.shape[0]:
        checks.append(_chk("Jumlah sampel sesuai .hea", "warn",
                           f"Header: {meta['n_samples']} sampel, .mat: {sig.shape[0]} sampel."))
    else:
        checks.append(_chk("Jumlah sampel sesuai .hea", "pass", f"{sig.shape[0]} sampel."))

    integer_like = bool(np.allclose(sig, np.round(sig)))
    if integer_like:
        checks.append(_chk("Nilai .mat berupa ADC mentah (bilangan bulat)", "pass",
                           "Konversi memakai gain & baseline dari .hea."))
    else:
        checks.append(_chk("Nilai .mat berupa ADC mentah (bilangan bulat)", "warn",
                           "Nilai berupa pecahan — kemungkinan sudah dalam satuan fisik. Konversi gain tetap diterapkan "
                           "sesuai .hea; periksa kewajaran amplitudo pada grafik sinyal."))

    out = np.empty(sig.shape, dtype=np.float32)
    for i, s in enumerate(meta["sigs"]):
        out[:, i] = ((sig[:, i].astype(np.float64) - s["baseline"]) / s["gain"] * (s["factor"] or 1.0))
    return out, float(meta["fs"]), [s["name"] for s in meta["sigs"]], checks
