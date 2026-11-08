import html
import time
import streamlit as st
from checks import failed, warned
from constants import CLASSES, CLASS_COLORS, CLASS_LABELS_LONG, HERO_SUBTITLE, HERO_TITLE
_ECG_PERIOD = 'l 24 0 l 6 -5 l 6 5 l 16 0 l 4 6 l 4 -46 l 4 40 l 4 10 l 4 -10 l 10 0 l 8 -8 l 8 8 l 102 0 '
_ECG_PATH_D = 'M 0 40 ' + _ECG_PERIOD * 6

def render_ecg_trace():
    return f'<div class="ecg-trace-wrap"><svg viewBox="0 -8 1200 60" class="ecg-trace-svg" aria-hidden="true"><path d="{_ECG_PATH_D}" pathLength="1" class="ecg-trace-path" fill="none"></path></svg></div>'

def render_hero():
    st.markdown(f'<header class="hero"><div class="hero-copy"><p class="hero-eyebrow">ConvNeXt-1D · Squeeze-and-Excitation · PTB-XL</p><h1>{HERO_TITLE}</h1><p class="hero-sub">{HERO_SUBTITLE}</p></div>{render_ecg_trace()}</header>', unsafe_allow_html=True)

class DebouncedProgress:
    DELAY = 0.1

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.t0 = time.monotonic()

    def show(self, label, pct):
        if time.monotonic() - self.t0 >= self.DELAY or pct >= 56:
            render_progress(self.placeholder, label, pct)

    def clear(self):
        self.placeholder.empty()

def render_progress(placeholder, label, pct):
    placeholder.markdown(f'<div class="scan-banner" role="status" aria-live="polite"><div class="scan-track"><div class="scan-fill" style="transform:scaleX({pct / 100:.2f});"></div></div><div class="scan-label">{label}</div></div>', unsafe_allow_html=True)

def abort_with_checks(progress_ph, checks, headline):
    progress_ph.empty()
    st.session_state.pop('result', None)
    lines = [f'**{headline}**', '']
    for c in failed(checks):
        lines.append(f"- **{c['name']}** — {c['detail']}")
    st.warning('\n'.join(lines))
    st.stop()

def show_input_warnings(checks):
    items = warned(checks)
    if items:
        st.warning('Input kurang ideal:\n\n' + '\n'.join((f"- **{c['name']}** — {c['detail']}" for c in items)))

def render_diagnosis_row(index, prob, threshold, active, gated=False):
    code = CLASSES[index]
    color = CLASS_COLORS[code]
    pct = max(0.0, min(1.0, prob)) * 100
    thr = max(0.0, min(1.0, threshold)) * 100
    if active:
        state = 'Terdeteksi'
    elif gated:
        state = 'Tidak lolos kriteria voltase'
    else:
        state = 'Di bawah ambang'
    delay = 0.04 * index
    return f"""<div class="dx-row{(' active' if active else '')}" style="--c:{color}; --d:{delay:.2f}s;"><div class="dx-name"><span class="dx-code"><i></i>{code}</span><span class="dx-long">{CLASS_LABELS_LONG[code]}</span></div><div class="dx-bar"><div class="dx-fill" style="width:{pct:.1f}%;"></div><div class="dx-tick" style="left:{thr:.1f}%;"></div></div><div class="dx-pct">{pct:.0f}%</div><div class="dx-state"><b>{state}</b><span>ambang {threshold:.2f}</span></div></div>"""

def render_verdict(active_labels):
    if active_labels:
        items = ''.join((f'<span class="verdict-item"><i style="background:{CLASS_COLORS[c]}"></i>{c}</span>' for c in active_labels))
        value = f'<p class="verdict-value">{items}</p>'
    else:
        value = '<p class="verdict-value none">Tidak ada</p>'
    return f'<div class="verdict"><p class="verdict-label">Label aktif</p>{value}</div>'

def render_meta(fs, n_samples, n_found, device):
    rows = [('Frekuensi sampling', f'{fs:g} Hz'), ('Jumlah sampel', f'{n_samples:,}'.replace(',', '.')), ('Lead terbaca', f'{n_found} dari 12'), ('Perangkat', str(device))]
    inner = ''.join((f'<div><dt>{k}</dt><dd>{html.escape(v)}</dd></div>' for k, v in rows))
    return f'<dl class="meta">{inner}</dl>'

def render_classification_tab(active_labels, probs, thresholds, preds, gated=None):
    gated = gated or set()
    left, right = st.columns([2, 5])
    with left:
        st.markdown(render_verdict(active_labels), unsafe_allow_html=True)
        st.markdown('<p class="dx-note">Garis tegak pada setiap batang menandai ambang kelas tersebut. Hasil ini adalah keluaran model, bukan diagnosis klinis.</p>', unsafe_allow_html=True)
        if 'HYP' in gated:
            st.markdown('<p class="dx-note">HYP dibatalkan: probabilitas model melewati ambang, tetapi kriteria voltase Sokolow-Lyon maupun Cornell tidak terpenuhi pada rekaman ini. Detail pengukurannya ada di tab Sinyal 12-lead.</p>', unsafe_allow_html=True)
    with right:
        rows = ''.join((render_diagnosis_row(i, float(probs[i]), thresholds[c], preds[c], gated=c in gated) for i, c in enumerate(CLASSES)))
        st.markdown(f'<div class="dx-list">{rows}</div>', unsafe_allow_html=True)

def render_results_skeleton(placeholder):
    meta = ''.join(('<div><span class="sk sk-line s"></span><span class="sk sk-line l"></span></div>' for _ in range(4)))
    rows = ''.join(('<div class="sk-row"><div><span class="sk sk-line l"></span></div><span class="sk sk-bar"></span><span class="sk sk-line l"></span><span class="sk sk-line l"></span></div>' for _ in CLASSES))
    placeholder.markdown(f'<div aria-busy="true" aria-label="Memuat hasil"><div class="sk-meta">{meta}</div><div class="sk-body"><div class="sk-verdict"><span class="sk sk-line s"></span><span class="sk sk-line xl"></span></div><div class="sk-list">{rows}</div></div></div>', unsafe_allow_html=True)

def render_empty_state():
    steps = [('1', 'Unggah', 'File .hea wajib, ditambah .dat atau .mat. Berkas diperiksa apakah benar sinyal EKG.'), ('2', 'Praproses', 'Sinyal dikonversi ke mV sesuai .hea, dipetakan ke 12 lead, di-resample, difilter, dan dinormalisasi seperti saat training.'), ('3', 'Klasifikasi', 'ConvNeXt-1D dengan Squeeze-and-Excitation menghitung probabilitas untuk tiap kelas.'), ('4', 'Hasil', 'Ambang batas menentukan label aktif. Khusus HYP, label hanya aktif kalau kriteria voltase (Sokolow-Lyon atau Cornell) juga terpenuhi.')]
    steps_html = ''.join((f'<div class="step" style="--d:{0.04 * i:.2f}s;"><div class="step-num">{n}</div><div class="step-title">{t}</div><p class="step-desc">{d}</p></div>' for i, (n, t, d) in enumerate(steps)))
    st.markdown(f'<div class="steps">{steps_html}</div>', unsafe_allow_html=True)
    legend_items = ''.join((f'<span class="legend-item"><i style="background:{CLASS_COLORS[c]}"></i><b>{c}</b>{CLASS_LABELS_LONG[c]}</span>' for c in CLASSES))
    st.markdown(f'<div class="legend"><p class="legend-title">Kelas yang diprediksi</p><div class="legend-grid">{legend_items}</div></div><p class="disclaimer">Alat bantu penelitian. Keluaran model bukan diagnosis klinis dan tidak menggantikan penilaian tenaga medis.</p>', unsafe_allow_html=True)
