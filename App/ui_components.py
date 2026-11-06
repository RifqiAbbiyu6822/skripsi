"""Komponen tampilan (HTML/Streamlit) yang dipakai halaman utama."""
import html

import streamlit as st

from checks import failed, warned
from constants import (
    CLASSES, CLASS_COLORS, CLASS_LABELS_LONG, HERO_SUBTITLE, HERO_TITLE,
)

# Satu kompleks EKG, diulang. Dibuat sekali saat halaman dimuat.
_ECG_PERIOD = ("l 24 0 l 6 -5 l 6 5 l 16 0 l 4 6 l 4 -46 l 4 40 "
               "l 4 10 l 4 -10 l 10 0 l 8 -8 l 8 8 l 102 0 ")
_ECG_PATH_D = "M 0 40 " + (_ECG_PERIOD * 6)


def render_ecg_trace():
    return (
        '<div class="ecg-trace-wrap">'
        '<svg viewBox="0 -8 1200 60" class="ecg-trace-svg" aria-hidden="true">'
        f'<path d="{_ECG_PATH_D}" pathLength="1" class="ecg-trace-path" fill="none"></path>'
        '</svg></div>'
    )


def render_hero():
    st.markdown(
        f'<header class="hero">'
        f'<p class="hero-eyebrow">ConvNeXt-1D · Squeeze-and-Excitation · PTB-XL</p>'
        f'<h1>{HERO_TITLE}</h1>'
        f'<p class="hero-sub">{HERO_SUBTITLE}</p>'
        f'{render_ecg_trace()}'
        f'</header>',
        unsafe_allow_html=True,
    )


def render_progress(placeholder, label, pct):
    placeholder.markdown(
        f'<div class="scan-banner"><div class="scan-track">'
        f'<div class="scan-fill" style="transform:scaleX({pct / 100:.2f});"></div></div>'
        f'<div class="scan-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Peringatan input
# ---------------------------------------------------------------------------

def abort_with_checks(progress_ph, checks, headline):
    """Tampilkan peringatan singkat (hanya kriteria yang tidak terpenuhi), lalu hentikan skrip."""
    progress_ph.empty()
    st.session_state.pop("result", None)
    lines = [f"**{headline}**", ""]
    for c in failed(checks):
        lines.append(f"- **{c['name']}** — {c['detail']}")
    st.warning("\n".join(lines))
    st.stop()


def show_input_warnings(checks):
    """Peringatan ringan untuk input yang masih bisa diproses tetapi kurang ideal."""
    items = warned(checks)
    if items:
        st.warning("Input kurang ideal:\n\n" + "\n".join(
            f"- **{c['name']}** — {c['detail']}" for c in items))


# ---------------------------------------------------------------------------
# Hasil klasifikasi
# ---------------------------------------------------------------------------

def render_diagnosis_row(index, prob, threshold, active):
    """Satu baris per kelas: nama, bar probabilitas dengan tanda ambang, persen, status."""
    code = CLASSES[index]
    color = CLASS_COLORS[code]
    pct = max(0.0, min(1.0, prob)) * 100
    thr = max(0.0, min(1.0, threshold)) * 100
    state = "Terdeteksi" if active else "Di bawah ambang"
    delay = 0.06 * index   # stagger antar baris; bar dan tick menyusul lewat CSS
    return (
        f'<div class="dx-row{" active" if active else ""}" style="--c:{color}; --d:{delay:.2f}s;">'
        f'<div class="dx-name"><span class="dx-code"><i></i>{code}</span>'
        f'<span class="dx-long">{CLASS_LABELS_LONG[code]}</span></div>'
        f'<div class="dx-bar"><div class="dx-fill" style="width:{pct:.1f}%;"></div>'
        f'<div class="dx-tick" style="left:{thr:.1f}%;"></div></div>'
        f'<div class="dx-pct">{pct:.0f}%</div>'
        f'<div class="dx-state"><b>{state}</b><span>ambang {threshold:.2f}</span></div>'
        f'</div>'
    )


def render_verdict(active_labels):
    if active_labels:
        items = "".join(
            f'<span class="verdict-item"><i style="background:{CLASS_COLORS[c]}"></i>{c}</span>'
            for c in active_labels
        )
        value = f'<p class="verdict-value">{items}</p>'
    else:
        value = '<p class="verdict-value none">Tidak ada</p>'
    return f'<div class="verdict"><p class="verdict-label">Label aktif</p>{value}</div>'


def render_meta(fs, n_samples, n_found, device):
    rows = [
        ("Frekuensi sampling", f"{fs:g} Hz"),
        ("Jumlah sampel", f"{n_samples:,}".replace(",", ".")),
        ("Lead terbaca", f"{n_found} dari 12"),
        ("Perangkat", str(device)),
    ]
    inner = "".join(f"<div><dt>{k}</dt><dd>{html.escape(v)}</dd></div>" for k, v in rows)
    return f'<dl class="meta">{inner}</dl>'


def render_classification_tab(active_labels, probs, thresholds, preds):
    st.markdown(render_verdict(active_labels), unsafe_allow_html=True)
    rows = "".join(
        render_diagnosis_row(i, float(probs[i]), thresholds[c], preds[c])
        for i, c in enumerate(CLASSES)
    )
    st.markdown(f'<div class="dx-list">{rows}</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="dx-note">Garis tegak pada setiap batang menandai ambang kelas tersebut. '
        'Hasil ini adalah keluaran model, bukan diagnosis klinis.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Halaman awal (belum ada hasil)
# ---------------------------------------------------------------------------

def render_empty_state():
    steps = [
        ("1", "Unggah", "File .hea wajib, ditambah .dat atau .mat. Berkas diperiksa apakah benar sinyal EKG."),
        ("2", "Praproses", "Sinyal dikonversi ke mV sesuai .hea, dipetakan ke 12 lead, di-resample, difilter, dan dinormalisasi seperti saat training."),
        ("3", "Klasifikasi", "ConvNeXt-1D dengan Squeeze-and-Excitation menghitung probabilitas untuk tiap kelas."),
        ("4", "Hasil", "Ambang batas menentukan label aktif. Probabilitas tiap kelas dan sinyal 12-lead ikut ditampilkan."),
    ]
    steps_html = "".join(
        f'<div class="step" style="--d:{0.06 * i:.2f}s;"><div class="step-num">{n}</div>'
        f'<div class="step-title">{t}</div><p class="step-desc">{d}</p></div>'
        for i, (n, t, d) in enumerate(steps)
    )
    st.markdown(f'<div class="steps">{steps_html}</div>', unsafe_allow_html=True)

    legend_items = "".join(
        f'<span class="legend-item"><i style="background:{CLASS_COLORS[c]}"></i><b>{c}</b>{CLASS_LABELS_LONG[c]}</span>'
        for c in CLASSES
    )
    st.markdown(
        f'<div class="legend"><p class="legend-title">Kelas yang diprediksi</p>'
        f'<div class="legend-grid">{legend_items}</div></div>'
        '<p class="disclaimer">Alat bantu penelitian. Keluaran model bukan diagnosis klinis '
        'dan tidak menggantikan penilaian tenaga medis.</p>',
        unsafe_allow_html=True,
    )