import os
import sys
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_APP_DIR, os.path.dirname(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import streamlit as st
import torch
from checks import EcgInputError, failed, warned
from constants import CLASSES, FMT_MAT, FMT_WFDB, LEAD_NAMES, MODEL_PATH, PAGE_TITLE, THRESHOLDS_PATH
from model_flow_tab import render_model_flow_tab
from ecg_io import check_header, parse_hea_meta, read_mat_record, read_wfdb_record
from ecg_findings import analyze_findings, hyp_decision
from ecg_validation import validate_signal
from inference import load_model_and_config, load_thresholds, run_inference
from preprocessing import build_model_input, prepare_leads
from signal_plot import render_signal_tab
from styles import inject_css
from ui_components import DebouncedProgress, abort_with_checks, render_classification_tab, render_empty_state, render_hero, render_meta, render_results_skeleton, show_input_warnings
st.set_page_config(page_title=PAGE_TITLE, page_icon=':anatomical_heart:', layout='wide', initial_sidebar_state='collapsed')
inject_css()
render_hero()
st.markdown('<p class="section-title">Unggah rekaman</p>', unsafe_allow_html=True)
st.markdown('<p class="section-hint">File <b>.hea</b> wajib untuk kedua format. Isinya dipakai untuk menentukan satuan dan skala voltase sinyal, serta jenis kelamin pasien (kalau tercantum) untuk kriteria Cornell.</p>', unsafe_allow_html=True)
input_format = st.radio('Format file', [FMT_WFDB, FMT_MAT], horizontal=True, label_visibility='collapsed')
c1, c2, c3 = st.columns([5, 5, 3], vertical_alignment='bottom')
if input_format == FMT_WFDB:
    with c1:
        hea_file = st.file_uploader('.hea (header, wajib)', type=['hea'])
    with c2:
        dat_file = st.file_uploader('.dat (sinyal)', type=['dat'])
    mat_file = None
else:
    with c1:
        mat_file = st.file_uploader('.mat (sinyal)', type=['mat'])
    with c2:
        hea_file = st.file_uploader('.hea (header, wajib)', type=['hea'])
    dat_file = None
with c3:
    run_btn = st.button('Jalankan klasifikasi', type='primary', use_container_width=True)
skeleton_ph = st.empty()
if run_btn:
    if hea_file is None:
        st.warning('File .hea wajib diunggah. File ini dibaca untuk menentukan satuan dan skala voltase sinyal.')
        st.stop()
    if input_format == FMT_WFDB:
        if dat_file is None:
            st.warning('Unggah file .dat untuk melanjutkan.')
            st.stop()
    elif mat_file is None:
        st.warning('Unggah file .mat untuk melanjutkan.')
        st.stop()
    if not os.path.exists(MODEL_PATH):
        st.error(f'File model tidak ditemukan: {MODEL_PATH}')
        st.stop()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    progress_ph = st.empty()
    progress = DebouncedProgress(progress_ph)
    show_progress = progress.show
    show_progress('Membaca header .hea (satuan & skala voltase)…', 8)
    hea_bytes = hea_file.getvalue()
    meta = parse_hea_meta(hea_bytes.decode('utf-8', errors='ignore'))
    header_checks = check_header(meta)
    if failed(header_checks):
        abort_with_checks(progress_ph, header_checks, 'File .hea tidak dapat dipakai, sehingga rekaman tidak bisa dibaca dengan benar.')
    show_progress('Membaca rekaman…', 18)
    try:
        if input_format == FMT_WFDB:
            raw_signal, fs, sig_names, load_checks = read_wfdb_record(hea_bytes, dat_file.getvalue(), meta)
        else:
            raw_signal, fs, sig_names, load_checks = read_mat_record(mat_file.getvalue(), meta)
    except EcgInputError as e:
        abort_with_checks(progress_ph, header_checks + e.checks, 'Berkas tidak dapat dibaca sebagai rekaman EKG.')
    except Exception as e:
        progress_ph.empty()
        st.error(f'Gagal membaca rekaman: {e}')
        st.stop()
    show_progress('Memvalidasi: apakah ini sinyal EKG 12-lead?…', 30)
    signal_checks = validate_signal(raw_signal, fs, sig_names)
    all_checks = header_checks + load_checks + signal_checks
    if failed(all_checks):
        abort_with_checks(progress_ph, all_checks, 'Berkas ini tidak dikenali sebagai sinyal EKG 12-lead yang valid, sehingga analisis dihentikan.')
    show_progress('Memetakan tata letak 12-lead…', 42)
    leads_12, found_leads = prepare_leads(raw_signal, sig_names)
    missing = [l for l in LEAD_NAMES if l not in found_leads]
    show_progress('Memuat bobot model…', 56)
    render_results_skeleton(skeleton_ph)
    try:
        model, ptb_config = load_model_and_config(MODEL_PATH, device)
    except Exception as e:
        progress_ph.empty()
        skeleton_ph.empty()
        st.error(f'Gagal memuat model: {e}')
        st.stop()
    show_progress('Resampling, filtering & normalisasi sinyal…', 72)
    model_input, resampled = build_model_input(leads_12, fs, ptb_config)
    show_progress('Menjalankan inferensi ConvNeXt-1D + SE…', 88)
    probs = run_inference(model, model_input, device)
    st.session_state['result'] = {'raw_signal': raw_signal, 'fs': fs, 'sig_names': sig_names, 'found_leads': found_leads, 'missing': missing, 'resampled': resampled, 'sr': ptb_config.sampling_rate, 'model_input': model_input, 'ptb_config': ptb_config, 'leads_12': leads_12, 'sex': meta.get('sex'), 'hea_units': [{'name': s['name'], 'unit': s['unit'], 'unit_given': s['unit_given'], 'gain': s['gain'], 'gain_given': s['gain_given'], 'baseline': s['baseline'], 'factor': s['factor']} for s in meta['sigs']], 'probs': probs, 'input_warnings': warned(all_checks), 'device': device}
    show_progress('Selesai', 100)
    progress_ph.empty()
    skeleton_ph.empty()
    st.toast('Klasifikasi selesai')
result = st.session_state.get('result')
if result is not None:
    raw_signal, fs, sig_names = (result['raw_signal'], result['fs'], result['sig_names'])
    found_leads, missing = (result['found_leads'], result['missing'])
    resampled, sr = (result['resampled'], result['sr'])
    probs = result['probs']
    model_input = result['model_input']
    ptb_config = result['ptb_config']
    sex = result.get('sex')
    thresholds = load_thresholds(THRESHOLDS_PATH)
    preds = {c: bool(probs[i] >= thresholds[c]) for i, c in enumerate(CLASSES)}
    gated = set()
    if preds['HYP'] and result.get('leads_12') is not None:
        try:
            std_found = [l for l in found_leads if l in LEAD_NAMES] or None
            hyp_findings = analyze_findings(result['leads_12'], fs, std_found)
            dec = hyp_decision(hyp_findings['hyp'], sex) if hyp_findings.get('ok') else None
        except Exception:
            dec = None
        if dec is not None and (not dec['active']):
            preds['HYP'] = False
            gated.add('HYP')
    active_labels = [c for c in CLASSES if preds[c]]
    model, _ = load_model_and_config(MODEL_PATH, result['device'])
    st.write('')
    st.markdown(render_meta(fs, raw_signal.shape[0], len(found_leads), result['device']), unsafe_allow_html=True)
    if missing:
        st.warning(f"Lead yang hilang diisi nol: {', '.join(missing)}")
    show_input_warnings(result.get('input_warnings', []))
    tab_dx, tab_signal, tab_flow = st.tabs(['Klasifikasi', 'Sinyal 12-lead', 'Alur Model'])
    with tab_dx:
        render_classification_tab(active_labels, probs, thresholds, preds, gated)
    with tab_signal:
        render_signal_tab(raw_signal, fs, sig_names, resampled, sr, preds=preds, leads_12=result.get('leads_12'), found_leads=found_leads, hea_units=result.get('hea_units'), sex=sex)
    with tab_flow:
        render_model_flow_tab(model, ptb_config, CLASSES, model_input, probs, thresholds, preds)
else:
    render_empty_state()
