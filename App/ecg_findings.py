"""Analisis morfologi 12-lead berbasis aturan: lokasi (bounding box) temuan MI, STTC, CD,
dan kriteria voltase HYP (Sokolow-Lyon, Cornell).

Catatan penting
  - Semua amplitudo dalam mV. Sinyal yang masuk ke modul ini SUDAH dikonversi dari satuan
    ADC memakai gain / baseline / satuan di berkas .hea (lihat ecg_io.py), jadi ambang klinis
    (mis. Sokolow-Lyon 3.5 mV) bisa dibandingkan langsung.
  - Kotak BUKAN atensi model. Model tidak mengeluarkan lokalisasi; kotak dibuat dari pengukuran
    morfologi (Q, ST, T, lebar QRS) pada lead & denyut yang sesuai dengan kelas yang diprediksi.
"""
import warnings

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

try:
    from constants import LEAD_NAMES
except Exception:  # pragma: no cover - fallback agar modul bisa diuji sendiri
    LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def bandpass_filter(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


LI ={n: i for i, n in enumerate(LEAD_NAMES)}

# --- ambang (nilai umum literatur; semuanya dalam mV / detik) ---------------------------
Q_LEADS = ["I", "II", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6"]   # Q patologis dinilai di sini
ST_LEADS = [l for l in LEAD_NAMES if l != "aVR"]
T_INV_LEADS = [l for l in LEAD_NAMES if l not in ("aVR", "V1", "III")]
Q_DUR_S, Q_MIN_MV, Q_RATIO = 0.040, 0.10, 0.25
ST_ELEV_MV, ST_DEP_MV = 0.10, -0.05
T_INV_MV = -0.10
QRS_WIDE_S = 0.120
SOKOLOW_LYON_MV = 3.5
CORNELL_MALE_MV, CORNELL_FEMALE_MV = 2.8, 2.0
HYP_LEADS = ["V1", "V3", "aVL", "V5", "V6"]

FINDING_NOTES = {
    "MI": "Biasanya terlihat dari perubahan segmen ST dan gelombang Q yang muncul di kelompok lead tertentu.",
    "STTC": "Perubahan segmen ST/gelombang T (iskemia, dll). Pola ini sifatnya morfologi gelombang, "
            "jadi dibantu kernel depthwise.",
    "CD": "Gangguan konduksi (misalnya blok cabang berkas). Cirinya QRS melebar atau bentuknya berubah.",
    "HYP": "Kriteria klinisnya membandingkan amplitudo antar-lead (Sokolow-Lyon, Cornell).",
}


def lead_column_map(sig_names):
    """{nama lead standar: indeks kolom} — aturan yang sama dengan preprocessing.prepare_leads."""
    name_map = {str(n).upper().replace(" ", ""): i for i, n in enumerate(sig_names)}
    m = {l: name_map[l.upper()] for l in LEAD_NAMES if l.upper() in name_map}
    if not m:
        m = {LEAD_NAMES[i]: i for i in range(min(len(sig_names), len(LEAD_NAMES)))}
    return m


def _bp(x, fs, hi):
    try:
        return bandpass_filter(x.astype(np.float32), fs, 0.5, min(hi, 0.45 * fs), 4).astype(np.float64)
    except Exception:
        return x - np.median(x, axis=1, keepdims=True)


def _nanmed(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=0)


def _find_qrs(x_det, fs):
    """Pusat QRS dari kecepatan spasial (norma turunan lintas lead) — tidak bergantung polaritas lead."""
    d = np.diff(x_det, axis=1, prepend=x_det[:, :1])
    sv = np.sqrt((d ** 2).sum(axis=0))
    w = max(1, int(round(0.02 * fs)))
    sv_s = np.convolve(sv, np.ones(w) / w, mode="same")
    p99 = float(np.percentile(sv_s, 99))
    if p99 <= 0:
        return sv_s, np.array([], dtype=int)
    peaks, _ = find_peaks(sv_s, distance=max(1, int(0.25 * fs)),
                          prominence=0.25 * p99, height=0.2 * p99)
    return sv_s, peaks


def _delineate(sv_s, c, fs):
    """Onset & offset QRS: sv turun di bawah 10% puncak (bertahan k sampel)."""
    thr = 0.10 * sv_s[c]
    k = max(1, int(round(0.012 * fs)))
    n = len(sv_s)
    lo, hi = max(0, c - int(0.10 * fs)), min(n - 1, c + int(0.14 * fs))

    on, run = c, 0
    while on > lo:
        run = run + 1 if sv_s[on] < thr else 0
        if run >= k:
            on += k - 1
            break
        on -= 1
    off, run = c, 0
    while off < hi:
        run = run + 1 if sv_s[off] < thr else 0
        if run >= k:
            off -= k - 1
            break
        off += 1
    return on, off


def analyze_findings(leads_12, fs, found_leads=None):
    """leads_12: (N, 12) mV urutan LEAD_NAMES. found_leads: nama lead yang benar-benar ada (sisanya nol-padded).
    Return dict hasil analisis, atau {'ok': False, 'reason': ...}."""
    fs = float(fs)
    x = np.nan_to_num(np.asarray(leads_12, dtype=np.float64)).T          # (12, N)
    n = x.shape[1]
    avail = [i for i, l in enumerate(LEAD_NAMES) if found_leads is None or l in found_leads]
    if len(avail) < 4 or n < int(3 * fs):
        return {"ok": False, "reason": "Lead atau durasi tidak cukup untuk analisis morfologi."}

    x_det = _bp(x, fs, 40.0)
    x_amp = _bp(x, fs, 100.0)            # filter lebih longgar agar amplitudo R/S tidak teredam
    sv_s, centers = _find_qrs(x_det[avail], fs)
    if len(centers) < 3:
        return {"ok": False, "reason": "Kurang dari 3 kompleks QRS terdeteksi."}

    nb = len(centers)
    shp = (nb, 12)
    R, S, QA, QD, ST, TA = (np.full(shp, np.nan) for _ in range(6))
    on_t, off_t, tend_t = np.zeros(nb), np.zeros(nb), np.zeros(nb)

    for b, c in enumerate(centers):
        on, off = _delineate(sv_s, int(c), fs)
        off = max(off, on + 2)
        on_t[b], off_t[b] = on / fs, off / fs
        nxt = centers[b + 1] if b + 1 < nb else None
        pre_a, pre_b = int(0.04 * fs), int(0.01 * fs)
        t0 = off + int(round(0.06 * fs))
        t1 = off + int(round(0.40 * fs))
        if nxt is not None:
            t1 = min(t1, int(nxt - 0.20 * fs))
        tend_t[b] = (off + int(round(0.20 * fs))) / fs

        for i in avail:
            xa, xd = x_amp[i], x_det[i]
            lo = max(0, on - pre_a)
            hi = max(lo + 1, on - pre_b)
            base = float(np.median(xd[lo:hi]))
            seg = xa[on:off + 1] - base
            if len(seg) < 3:
                continue
            rmax, smin = float(seg.max()), float(seg.min())
            R[b, i], S[b, i] = max(rmax, 0.0), max(-smin, 0.0)

            rpos = int(np.argmax(seg))
            if rmax < 0.05 * abs(smin):                       # kompleks QS (tanpa R)
                QA[b, i], QD[b, i] = -smin, (off - on) / fs
            elif rpos > 0:
                pre = seg[:rpos + 1]
                if pre.min() < 0:
                    qpos = int(np.argmin(pre))
                    back = np.where(pre[qpos:] >= 0)[0]
                    qend = qpos + (int(back[0]) if len(back) else rpos - qpos)
                    QA[b, i], QD[b, i] = -float(pre.min()), qend / fs
                else:
                    QA[b, i], QD[b, i] = 0.0, 0.0
            else:
                QA[b, i], QD[b, i] = 0.0, 0.0

            s0, s1 = off + int(round(0.06 * fs)), off + int(round(0.08 * fs))
            if s1 < n:
                ST[b, i] = float(np.mean(xd[s0:max(s0 + 1, s1 + 1)])) - base
            if t1 - t0 >= 3 and t1 < n:
                tseg = xd[t0:t1] - base
                TA[b, i] = float(tseg[int(np.argmax(np.abs(tseg)))])

    r_m, s_m, qa_m, qd_m, st_m, ta_m = (_nanmed(a) for a in (R, S, QA, QD, ST, TA))
    qrs_ms = float(np.median(off_t - on_t) * 1000.0)
    rr = np.diff(centers) / fs
    hr = float(60.0 / np.median(rr)) if len(rr) else float("nan")

    # --- kriteria per lead -------------------------------------------------------------
    q_path, st_abn, t_inv = {}, {}, {}
    for l in LEAD_NAMES:
        i = LI[l]
        if i not in avail:
            continue
        qa, qd, r = qa_m[i], qd_m[i], r_m[i]
        q_path[l] = bool(l in Q_LEADS and qa >= Q_MIN_MV and
                         (qd >= Q_DUR_S or qa >= Q_RATIO * max(r, 1e-6)))
        st_abn[l] = bool(l in ST_LEADS and (st_m[i] >= ST_ELEV_MV or st_m[i] <= ST_DEP_MV))
        t_inv[l] = bool(l in T_INV_LEADS and ta_m[i] <= T_INV_MV)

    lead_table = [{
        "Lead": l,
        "Q (ms)": round(float(qd_m[LI[l]]) * 1000, 0), "Q (mV)": round(float(qa_m[LI[l]]), 2),
        "R (mV)": round(float(r_m[LI[l]]), 2), "S (mV)": round(float(s_m[LI[l]]), 2),
        "ST (mV)": round(float(st_m[LI[l]]), 2), "T (mV)": round(float(ta_m[LI[l]]), 2),
    } for l in LEAD_NAMES if LI[l] in avail]

    def top3(score):
        ranked = sorted((l for l in score if not np.isnan(score[l])), key=lambda l: -score[l])
        return ranked[:3]

    def mk(cls, lead, t0, t1, kind, text, meets):
        return {"cls": cls, "lead": lead, "t0": float(t0), "t1": float(t1),
                "kind": kind, "text": text, "meets": bool(meets)}

    boxes = {"MI": [], "STTC": [], "CD": [], "HYP": []}
    sel = {}

    # MI: gelombang Q + segmen ST
    mi_leads = [l for l in ST_LEADS if l in q_path and (q_path[l] or st_abn[l])]
    mi_ok = bool(mi_leads)
    if not mi_ok:
        mi_leads = top3({l: abs(st_m[LI[l]]) for l in ST_LEADS if l in q_path})
    sel["MI"] = {"leads": mi_leads, "meets": mi_ok}
    for l in mi_leads:
        i = LI[l]
        txt = (f"<b>{l}</b> — Q {qd_m[i] * 1000:.0f} ms / {qa_m[i]:.2f} mV, ST {st_m[i]:+.2f} mV"
               + ("" if mi_ok else "<br><i>di bawah kriteria klasik</i>"))
        for b in range(nb):
            if q_path.get(l):
                boxes["MI"].append(mk("MI", l, on_t[b], on_t[b] + max(qd_m[i], 0.03), "Q", txt, mi_ok))
            boxes["MI"].append(mk("MI", l, off_t[b], off_t[b] + 0.12, "ST", txt, mi_ok))

    # STTC: segmen ST sampai puncak gelombang T
    st_leads = [l for l in ST_LEADS if l in st_abn and (st_abn[l] or t_inv.get(l, False))]
    st_ok = bool(st_leads)
    if not st_ok:
        st_leads = top3({l: max(abs(st_m[LI[l]]), -ta_m[LI[l]] if l in T_INV_LEADS else 0.0)
                         for l in ST_LEADS if l in st_abn})
    sel["STTC"] = {"leads": st_leads, "meets": st_ok}
    for l in st_leads:
        i = LI[l]
        txt = (f"<b>{l}</b> — ST {st_m[i]:+.2f} mV, T {ta_m[i]:+.2f} mV"
               + ("" if st_ok else "<br><i>di bawah kriteria klasik</i>"))
        for b in range(nb):
            boxes["STTC"].append(mk("STTC", l, off_t[b], tend_t[b], "ST-T", txt, st_ok))

    # CD: lebar QRS (lead dengan QRS yang tampak jelas)
    amp = {l: r_m[LI[l]] + s_m[LI[l]] for l in LEAD_NAMES if LI[l] in avail}
    amax = max(amp.values()) if amp else 0.0
    cd_leads = sorted((l for l, a in amp.items() if a >= 0.4 * amax), key=lambda l: -amp[l])[:4]
    cd_ok = qrs_ms >= QRS_WIDE_S * 1000
    sel["CD"] = {"leads": cd_leads, "meets": cd_ok}
    for l in cd_leads:
        txt = (f"<b>{l}</b> — lebar QRS ≈ {qrs_ms:.0f} ms"
               + ("" if cd_ok else "<br><i>di bawah 120 ms</i>"))
        for b in range(nb):
            boxes["CD"].append(mk("CD", l, on_t[b], off_t[b], "QRS", txt, cd_ok))

    # HYP: kriteria voltase (mV, dari .hea)
    def have(*ls):
        return all(LI[l] in avail for l in ls)

    hyp = {"available": False, "missing": [l for l in HYP_LEADS if LI[l] not in avail]}
    if have("V1") and (have("V5") or have("V6")):
        r56 = max(r_m[LI[l]] for l in ("V5", "V6") if LI[l] in avail)
        hyp.update(S_V1=float(s_m[LI["V1"]]), R_V56=float(r56),
                   R_V5=float(r_m[LI["V5"]]) if have("V5") else None,
                   R_V6=float(r_m[LI["V6"]]) if have("V6") else None)
        hyp["SL"] = hyp["S_V1"] + hyp["R_V56"]
        hyp["SL_met"] = hyp["SL"] > SOKOLOW_LYON_MV
        hyp["available"] = True
    if have("aVL", "V3"):
        hyp.update(R_aVL=float(r_m[LI["aVL"]]), S_V3=float(s_m[LI["V3"]]))
        hyp["Cornell"] = hyp["R_aVL"] + hyp["S_V3"]
        hyp["Cornell_male"] = hyp["Cornell"] > CORNELL_MALE_MV
        hyp["Cornell_female"] = hyp["Cornell"] > CORNELL_FEMALE_MV
        hyp["available"] = True
    hyp_leads = [l for l in HYP_LEADS if LI[l] in avail]
    sel["HYP"] = {"leads": hyp_leads, "meets": bool(hyp.get("SL_met") or hyp.get("Cornell_male"))}
    for l in hyp_leads:
        i = LI[l]
        txt = f"<b>{l}</b> — R {r_m[i]:.2f} mV, S {s_m[i]:.2f} mV"
        for b in range(nb):
            boxes["HYP"].append(mk("HYP", l, on_t[b], off_t[b], "QRS", txt, sel["HYP"]["meets"]))

    return {"ok": True, "n_beats": nb, "hr": hr, "qrs_ms": qrs_ms, "lead_table": lead_table,
            "boxes": boxes, "sel": sel, "hyp": hyp}
    
    def hyp_decision(hyp, sex=None):
        """Keputusan HYP dari kriteria voltase. sex: 'male', 'female', atau None.
    Return None kalau kriteria tidak bisa dihitung (lead kurang)."""
    if not hyp or not hyp.get("available"):
        return None
    sl_met = bool(hyp.get("SL_met"))
    c = hyp.get("Cornell")
    c_male = c is not None and c > CORNELL_MALE_MV
    c_female = c is not None and c > CORNELL_FEMALE_MV
    if sex == "male":
        c_met = c_male
    elif sex == "female":
        c_met = c_female
    else:  # sex tidak diketahui: salah satu ambang terpenuhi sudah cukup
        c_met = c_male or c_female
    return {"sex": sex, "sl_met": sl_met, "c_male": c_male, "c_female": c_female,
            "cornell_met": c_met, "active": sl_met or c_met}