"""Validasi: apakah sinyal ini benar-benar EKG 12-lead?"""
import numpy as np
from scipy.signal import find_peaks, peak_widths

from checks import make_check as _chk
from constants import LEAD_NAMES, TARGET_LEADS
from signal_ops import bandpass_filter


def _detect_peaks(ref, fs):
    dist = max(1, int(0.4 * fs))
    for cand in (ref, np.abs(ref)):
        hi, med = np.percentile(cand, 99), np.median(cand)
        prom = 0.4 * (hi - med)
        if prom <= 0:
            continue
        peaks, _ = find_peaks(cand, distance=dist, height=np.percentile(cand, 90), prominence=prom)
        if len(peaks) >= 3:
            return peaks
    return np.array([], dtype=int)


def validate_signal(sig, fs, names):
    """sig: (N, n_leads) dalam mV. Return list pemeriksaan; ada 'fail' -> bukan EKG yang valid."""
    checks = []
    sig = np.asarray(sig, dtype=np.float64)
    n, nl = sig.shape
    names = [str(x) for x in names][:nl] + [f"Lead {i + 1}" for i in range(len(names), nl)]
    norm = [x.upper().replace(" ", "") for x in names]
    dur = n / float(fs)

    fin = float(np.isfinite(sig).mean())
    if fin < 0.95:
        checks.append(_chk("Nilai sinyal valid (bukan NaN/Inf)", "fail",
                           f"Hanya {fin * 100:.1f}% sampel bernilai valid; sisanya NaN/Inf."))
    elif fin < 1.0:
        checks.append(_chk("Nilai sinyal valid (bukan NaN/Inf)", "warn",
                           f"{(1 - fin) * 100:.2f}% sampel NaN/Inf (diganti 0)."))
    else:
        checks.append(_chk("Nilai sinyal valid (bukan NaN/Inf)", "pass", "Semua sampel valid."))
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    if fs < 100 or fs > 5000:
        checks.append(_chk("Frekuensi sampling wajar untuk EKG", "fail",
                           f"{fs:g} Hz di luar rentang 100–5000 Hz. EKG diagnostik umumnya 100–1000 Hz."))
    else:
        checks.append(_chk("Frekuensi sampling wajar untuk EKG", "pass", f"{fs:g} Hz."))

    if dur < 3:
        checks.append(_chk("Durasi rekaman cukup", "fail",
                           f"{dur:.1f} detik — terlalu pendek untuk memuat ≥ 3 denyut jantung."))
    elif dur < 8:
        checks.append(_chk("Durasi rekaman cukup", "warn",
                           f"{dur:.1f} detik; model dilatih pada jendela ± 10 detik (sisanya di-zero-pad)."))
    else:
        checks.append(_chk("Durasi rekaman cukup", "pass", f"{dur:.1f} detik."))

    missing = [l for l in LEAD_NAMES if l.upper() not in norm]
    n_recog = TARGET_LEADS - len(missing)
    if not missing:
        checks.append(_chk("Lead standar 12-lead lengkap", "pass", "I, II, III, aVR, aVL, aVF, V1–V6 ditemukan."))
    elif n_recog >= 8:
        checks.append(_chk("Lead standar 12-lead lengkap", "warn",
                           f"Tidak ada: {', '.join(missing)} — akan diisi nol (akurasi model menurun)."))
    elif nl == TARGET_LEADS:
        checks.append(_chk("Lead standar 12-lead lengkap", "warn",
                           "Nama lead tidak standar; urutan kanal diasumsikan I, II, III, aVR, aVL, aVF, V1–V6."))
    else:
        checks.append(_chk("Lead standar 12-lead lengkap", "fail",
                           f"Hanya {n_recog} dari 12 lead standar dikenali (tidak ada: {', '.join(missing)}). "
                           f"Berkas memiliki {nl} kanal — bukan rekaman EKG 12-lead."))

    ptp = np.percentile(sig, 99, axis=0) - np.percentile(sig, 1, axis=0)
    flat = [i for i in range(nl) if ptp[i] < 0.01]
    if len(flat) == nl:
        checks.append(_chk("Sinyal tidak datar", "fail",
                           "Semua kanal datar/ konstan (variasi < 0.01 mV) — tidak ada aktivitas listrik jantung."))
        return checks
    elif flat:
        checks.append(_chk("Sinyal tidak datar", "warn",
                           f"Kanal datar (lead mati): {', '.join(names[i] for i in flat)}."))
    else:
        checks.append(_chk("Sinyal tidak datar", "pass", "Semua kanal memiliki variasi amplitudo."))
    active = [i for i in range(nl) if i not in flat]

    med_pp = float(np.median(ptp[active]))
    if 0.1 <= med_pp <= 10:
        st_amp, tail = "pass", ""
    elif 0.03 <= med_pp < 0.1 or 10 < med_pp <= 30:
        st_amp, tail = "warn", " Di luar rentang umum — periksa satuan/gain di .hea."
    else:
        st_amp, tail = "fail", (" Tidak masuk akal untuk EKG setelah konversi satuan dari .hea — "
                                "bukan sinyal EKG, atau gain/satuan di .hea tidak sesuai dengan .mat/.dat.")
    checks.append(_chk("Amplitudo dalam rentang EKG (mV)", st_amp,
                       f"Median peak-to-peak lintas lead ≈ {med_pp:.2f} mV (wajar ≈ 0.1–10 mV).{tail}"))

    if "II" in norm and norm.index("II") in active:
        ref_i = norm.index("II")
    else:
        ref_i = max(active, key=lambda i: ptp[i])
    x = sig[:, ref_i]

    ratio = float(np.std(np.diff(x)) / (np.std(x) + 1e-12))
    if ratio > 0.9:
        st_n, tail = "fail", " Terlalu acak seperti noise (white noise ≈ 1.4); EKG memiliki bentuk gelombang yang halus."
    elif ratio > 0.6:
        st_n, tail = "warn", " Cukup berisik."
    else:
        st_n, tail = "pass", ""
    checks.append(_chk("Bukan noise acak", st_n,
                       f"Rasio variasi sampel-ke-sampel di lead {names[ref_i]} = {ratio:.2f}.{tail}"))

    try:
        hi = min(40.0, 0.45 * fs)
        xf = bandpass_filter(x[None, :].astype(np.float32), fs, 0.5, hi, 4)[0].astype(np.float64)
    except Exception:
        xf = x - np.median(x)

    peaks = _detect_peaks(xf, fs)
    if len(peaks) < 3:
        checks.append(_chk(
            "Kompleks QRS berulang terdeteksi", "fail",
            f"Tidak ada: kurang dari 3 R-peak terdeteksi di lead {names[ref_i]}. EKG memiliki puncak QRS tajam yang "
            "berulang tiap ≈ 0.3–2 detik; sinyal ini tidak menunjukkan pola tersebut."))
        return checks

    rr = np.diff(peaks) / float(fs)
    med_rr = float(np.median(rr))
    hr = 60.0 / med_rr
    if 30 <= hr <= 200:
        st_hr, tail = "pass", ""
    elif 25 <= hr < 30 or 200 < hr <= 240:
        st_hr, tail = "warn", " Laju ekstrem."
    else:
        st_hr, tail = "fail", " Di luar rentang fisiologis; ini bukan irama jantung."
    checks.append(_chk("Laju jantung fisiologis", st_hr,
                       f"≈ {hr:.0f} bpm dari {len(peaks)} R-peak (lead {names[ref_i]}).{tail}"))

    cv = float(rr.std() / rr.mean())
    checks.append(_chk("Keteraturan interval R–R", "warn" if cv > 0.5 else "pass",
                       f"Koefisien variasi R–R = {cv:.2f}."
                       + (" Sangat tidak teratur — bisa aritmia berat, artefak, atau bukan EKG." if cv > 0.5 else "")))

    try:
        s_pol = xf if np.median(xf[peaks]) >= 0 else -xf
        w = peak_widths(s_pol, peaks, rel_height=0.5)[0] / float(fs)
        med_w = float(np.median(w))
        wr = med_w / med_rr
        if wr > 0.35 or med_w > 0.25:
            st_w, tail = "fail", " Puncak terlalu lebar (mis. gelombang sinus) — bukan kompleks QRS yang tajam."
        elif wr > 0.25 or med_w > 0.15:
            st_w, tail = "warn", " Puncak agak lebar."
        else:
            st_w, tail = "pass", ""
        checks.append(_chk("Lebar puncak QRS sesuai EKG", st_w,
                           f"Lebar R-peak ≈ {med_w * 1000:.0f} ms ({wr * 100:.0f}% dari interval R–R).{tail}"))
    except Exception as e:
        checks.append(_chk("Lebar puncak QRS sesuai EKG", "warn", f"Tidak dapat diukur: {e}"))

    a, b = int(0.25 * fs), int(0.45 * fs)
    segs = [xf[p - a:p + b] for p in peaks if p - a >= 0 and p + b <= len(xf)]
    if len(segs) >= 3:
        S = np.stack(segs)
        tpl = np.median(S, axis=0)
        if tpl.std() > 1e-9:
            cors = [np.corrcoef(s, tpl)[0, 1] for s in S if s.std() > 1e-9]
            mc = float(np.nanmean(cors)) if cors else 0.0
            if mc >= 0.6:
                st_c, tail = "pass", ""
            elif mc >= 0.4:
                st_c, tail = "warn", " Morfologi antar-beat agak bervariasi."
            else:
                st_c, tail = "fail", " Bentuk antar-denyut tidak konsisten — bukan pola EKG yang berulang."
            checks.append(_chk("Morfologi beat konsisten", st_c,
                               f"Korelasi rata-rata beat vs template = {mc:.2f} ({len(segs)} beat).{tail}"))
        else:
            checks.append(_chk("Morfologi beat konsisten", "warn", "Template beat datar; tidak dapat dinilai."))
    else:
        checks.append(_chk("Morfologi beat konsisten", "warn", "Kurang dari 3 beat utuh; tidak dapat dinilai."))
    return checks
