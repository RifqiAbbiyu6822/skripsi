"""Tab 'Sinyal 12-lead': plot EKG bergaya kertas grid dengan panel yang bisa digeser."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from constants import CLASS_COLORS, LEAD_NAMES
from ecg_findings import (
    CORNELL_FEMALE_MV, CORNELL_MALE_MV, FINDING_NOTES, SOKOLOW_LYON_MV, analyze_findings,
)

BOX_CLASSES = ("MI", "STTC", "CD")          # kelas yang divisualisasikan sebagai bounding box
BOX_TITLES = {"MI": "Infark miokard", "STTC": "Perubahan ST/T", "CD": "Gangguan konduksi"}
ECG_PAPER_BG = "#141518"
ECG_MINOR_GRID = "rgba(233,231,226,0.05)"
ECG_MAJOR_GRID = "rgba(233,231,226,0.13)"
ECG_TRACE_COLOR = "#E9E7E2"
ECG_TEXT_COLOR = "#9A9892"


def _hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _lead_rows(plot_leads):
    """{nama lead standar: nomor baris subplot (1-based)} — plot mentah memakai urutan kolom berkas."""
    key = {str(n).upper().replace(" ", ""): i + 1 for i, n in enumerate(plot_leads)}
    return {l: key[l.upper()] for l in LEAD_NAMES if l.upper() in key}


def _add_boxes(fig, plot_signal, plot_fs, plot_leads, boxes, lead_y_ranges):
    """Bounding box per temuan. Posisi horizontal = waktu temuan; tinggi mengikuti amplitudo sinyal di jendela itu."""
    rows = _lead_rows(plot_leads)
    n = plot_signal.shape[0]
    shapes = []
    labeled = set()
    for cls_idx, cls in enumerate(BOX_CLASSES):
        color = CLASS_COLORS[cls]
        for bx in boxes.get(cls, []):
            row = rows.get(bx["lead"])
            if row is None:
                continue
            i0 = max(0, int(bx["t0"] * plot_fs) - 1)
            i1 = min(n, int(bx["t1"] * plot_fs) + 2)
            if i1 - i0 < 2:
                continue
            seg = plot_signal[i0:i1, row - 1]
            y_lo, y_hi = float(seg.min()), float(seg.max())
            axis_lo, axis_hi = lead_y_ranges[row - 1]
            pad = max((y_hi - y_lo) * 0.18, (axis_hi - axis_lo) * 0.07)
            y_lo, y_hi = max(axis_lo, y_lo - pad), min(axis_hi, y_hi + pad)
            ax = "" if row == 1 else str(row)
            shapes.append(dict(
                type="rect", xref=f"x{ax}", yref=f"y{ax}",
                x0=bx["t0"], x1=bx["t1"], y0=y_lo, y1=y_hi,
                line=dict(color=color, width=1.6, dash="solid" if bx["meets"] else "dot"),
                fillcolor=_hex_rgba(color, 0.14 if bx["meets"] else 0.07), layer="above",
            ))
            tag = (cls, bx["lead"], bx["kind"])
            if tag not in labeled:                      # satu label teks per (kelas, lead, jenis)
                labeled.add(tag)
                fig.add_annotation(
                    x=bx["t0"], y=y_hi, xref=f"x{ax}", yref=f"y{ax}", text=f"{cls}·{bx['kind']}",
                    showarrow=False, xanchor="left", yanchor="bottom", yshift=cls_idx * 11,
                    font=dict(size=10, color=color, family="Instrument Sans, sans-serif"),
                )
    if shapes:
        fig.update_layout(shapes=list(fig.layout.shapes) + shapes)
        by_row = {}      # satu trace hover tak terlihat per (baris, kelas)
        for cls in BOX_CLASSES:
            for bx in boxes.get(cls, []):
                row = rows.get(bx["lead"])
                if row is None:
                    continue
                by_row.setdefault((row, cls), ([], [], []))
                xs, ys, ts = by_row[(row, cls)]
                xs.append((bx["t0"] + bx["t1"]) / 2)
                i0 = max(0, int(bx["t0"] * plot_fs) - 1)
                i1 = min(n, int(bx["t1"] * plot_fs) + 2)
                ys.append(float(plot_signal[i0:max(i0 + 1, i1), row - 1].max()))
                ts.append(f"<b>{cls}</b> · {bx['kind']}<br>{bx['text']}")
        for (row, cls), (xs, ys, ts) in by_row.items():
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)"),
                hoverinfo="text", hovertext=ts, showlegend=False,
                hoverlabel=dict(bgcolor="#1B1C21", bordercolor=CLASS_COLORS[cls],
                                font=dict(color="#E9E7E2", size=12)),
            ), row=row, col=1)
    return len(shapes)


def _build_figure(plot_signal, plot_leads, plot_fs, zoom_factor, boxes=None):
    t_axis = np.arange(plot_signal.shape[0]) / float(plot_fs)
    t_min, t_max = float(t_axis[0]), float(t_axis[-1])
    n_leads_plot = plot_signal.shape[1]

    fig = make_subplots(
        rows=n_leads_plot, cols=1, shared_xaxes=True,
        subplot_titles=[str(l) for l in plot_leads],
        vertical_spacing=0.012,
    )

    lead_y_ranges = []
    for lead_idx in range(n_leads_plot):
        lead_signal = plot_signal[:, lead_idx]
        lead_min = float(lead_signal.min())
        lead_max = float(lead_signal.max())
        pad = (lead_max - lead_min) * 0.08 or 1.0
        lead_y_ranges.append((lead_min - pad, lead_max + pad))

    for i in range(n_leads_plot):
        fig.add_trace(
            go.Scatter(x=t_axis, y=plot_signal[:, i], mode="lines",
                       line=dict(width=1.2, color=ECG_TRACE_COLOR), showlegend=False),
            row=i + 1, col=1,
        )

    fig.update_layout(
        height=88 * n_leads_plot,
        width=int(1380 * zoom_factor),
        margin=dict(l=44, r=24, t=30, b=30),
        template="plotly_dark",
        plot_bgcolor=ECG_PAPER_BG,
        paper_bgcolor=ECG_PAPER_BG,
        font=dict(family="Instrument Sans, sans-serif", color=ECG_TEXT_COLOR),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(family="Instrument Sans, sans-serif", color=ECG_TEXT_COLOR, size=12)

    fig.update_xaxes(title_text="detik", row=n_leads_plot, col=1)
    fig.update_xaxes(
        showgrid=True, gridcolor=ECG_MAJOR_GRID, gridwidth=1,
        griddash="solid", dtick=0.2, range=[t_min, t_max],
        minor=dict(showgrid=True, gridcolor=ECG_MINOR_GRID, gridwidth=0.6, dtick=0.04),
        zeroline=False, linecolor=ECG_MAJOR_GRID,
    )

    for lead_idx in range(n_leads_plot):
        y0, y1 = lead_y_ranges[lead_idx]
        span = (y1 - y0) or 1.0
        y_major_dtick = span / 6.0
        y_minor_dtick = y_major_dtick / 5.0
        fig.update_yaxes(
            showgrid=True, gridcolor=ECG_MAJOR_GRID, gridwidth=1,
            griddash="solid", dtick=y_major_dtick,
            minor=dict(showgrid=True, gridcolor=ECG_MINOR_GRID, gridwidth=0.6, dtick=y_minor_dtick),
            zeroline=False, linecolor=ECG_MAJOR_GRID, showticklabels=False,
            range=[y0, y1],
            row=lead_idx + 1, col=1,
        )
    if boxes:
        _add_boxes(fig, plot_signal, plot_fs, plot_leads, boxes, lead_y_ranges)
    return fig


def _fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def _criterion_card(title, formula, value, threshold_txt, met):
    color = CLASS_COLORS["HYP"]
    badge = ("memenuhi kriteria", "var(--bad)") if met else ("di bawah ambang", "var(--good)")
    return (
        f'<div style="flex:1 1 280px;background:var(--surface);border:1px solid var(--line);'
        f'border-top:2px solid {color};border-radius:var(--r-md);padding:var(--s-4);">'
        f'<div style="font-size:var(--text-meta);letter-spacing:.08em;text-transform:uppercase;'
        f'color:var(--faint);font-weight:500;">{title}</div>'
        f'<div style="font-size:var(--text-secondary);color:var(--muted);margin:var(--s-2) 0;">{formula}</div>'
        f'<div style="font-size:1.75rem;font-weight:600;color:var(--text);line-height:1.2;">{value} '
        f'<span style="font-size:var(--text-secondary);color:var(--muted);font-weight:400;">mV</span></div>'
        f'<div style="font-size:var(--text-secondary);color:var(--muted);margin-top:var(--s-2);">'
        f'ambang {threshold_txt} · <span style="color:{badge[1]};font-weight:500;">{badge[0]}</span></div>'
        f'</div>'
    )


def _render_hyp_panel(findings, hea_units):
    st.markdown('<p class="section-title" style="margin-top:var(--s-5);">HYP — kriteria voltase klinis</p>',
                unsafe_allow_html=True)
    st.caption(FINDING_NOTES["HYP"] + " Amplitudo diambil dari sinyal dalam mV (setelah gain, baseline, "
               "dan satuan di berkas .hea diterapkan), bukan dari input model yang sudah dinormalisasi.")

    if hea_units:
        with st.expander("Penyesuaian satuan dari berkas .hea", expanded=False):
            rows = []
            for u in hea_units:
                f = u["factor"]
                rows.append({
                    "Kanal": u["name"],
                    "Satuan .hea": u["unit"] + ("" if u["unit_given"] else " (default)"),
                    "Gain (adu/satuan)": u["gain"] if u["gain_given"] else f"{u['gain']:g} (default)",
                    "Baseline (adu)": u["baseline"],
                    "Faktor → mV": "tidak dikenali" if f is None else f"{f:g}",
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.caption("mV = (nilai ADC − baseline) ÷ gain × faktor satuan.")

    if not findings.get("ok"):
        st.info(findings.get("reason", "Analisis kriteria tidak dapat dihitung."))
        return
    hyp = findings["hyp"]
    if not hyp.get("available"):
        st.info("Lead yang dibutuhkan tidak tersedia: " + ", ".join(hyp["missing"]) + ".")
        return

    sex = st.radio("Jenis kelamin (untuk ambang Cornell)", ["Laki-laki", "Perempuan"], horizontal=True,
                   key="hyp_sex")
    cards = []
    if "SL" in hyp:
        cards.append(_criterion_card(
            "Sokolow-Lyon", f"S V1 ({_fmt(hyp['S_V1'])}) + R V5/V6 maks ({_fmt(hyp['R_V56'])})",
            _fmt(hyp["SL"]), f"&gt; {SOKOLOW_LYON_MV:g} mV", hyp["SL_met"]))
    if "Cornell" in hyp:
        thr = CORNELL_MALE_MV if sex == "Laki-laki" else CORNELL_FEMALE_MV
        cards.append(_criterion_card(
            "Cornell voltage", f"R aVL ({_fmt(hyp['R_aVL'])}) + S V3 ({_fmt(hyp['S_V3'])})",
            _fmt(hyp["Cornell"]), f"&gt; {thr:g} mV ({sex.lower()})", hyp["Cornell"] > thr))
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:var(--s-4);margin:var(--s-3) 0;">{"".join(cards)}</div>',
                unsafe_allow_html=True)
    if hyp["missing"]:
        st.caption("Lead tidak ada di rekaman: " + ", ".join(hyp["missing"]) + " — kriteria terkait tidak dihitung.")


def render_signal_tab(raw_signal, fs, sig_names, resampled, sr,
                      preds=None, leads_12=None, found_leads=None, hea_units=None):
    preds = preds or {}
    st.write("")
    c_show, c_zoom = st.columns(2)
    with c_show:
        show_raw = st.toggle("Sinyal mentah (panjang asli)", value=True)
    with c_zoom:
        plot_zoom = st.select_slider(
            "Lebar grafik", options=["1x", "1.5x", "2x", "2.5x", "3x"], value="2x",
        )
    zoom_factor = float(plot_zoom.replace("x", ""))
    st.caption("Amplitudo dikonversi ke mV berdasarkan gain, baseline, dan satuan di berkas .hea.")

    plot_signal = raw_signal if show_raw else resampled
    plot_leads = sig_names if show_raw else LEAD_NAMES
    plot_fs = fs if show_raw else sr
    n_leads_plot = plot_signal.shape[1]

    # --- temuan morfologi (mV). Mode mentah: dari sinyal asli; mode resampled: dari sinyal yang ditampilkan,
    #     supaya waktu kotak selalu selaras dengan sumbu-x grafik (jendela 10 detik bisa terpotong).
    std_found = [l for l in (found_leads or []) if l in LEAD_NAMES] or None
    analysis_in = (leads_12 if (show_raw and leads_12 is not None) else resampled)
    analysis_fs = fs if (show_raw and leads_12 is not None) else sr
    try:
        findings = analyze_findings(analysis_in, analysis_fs, std_found)
    except Exception as e:  # analisis tidak boleh merusak tampilan sinyal
        findings = {"ok": False, "reason": f"Analisis morfologi gagal: {e}"}
    hyp_findings = findings
    if not show_raw and leads_12 is not None:
        try:
            hyp_findings = analyze_findings(leads_12, fs, std_found)
        except Exception as e:
            hyp_findings = {"ok": False, "reason": f"Analisis morfologi gagal: {e}"}

    active_box_classes = [c for c in BOX_CLASSES if preds.get(c)]
    boxes = None
    if findings.get("ok") and active_box_classes:
        boxes = {c: findings["boxes"][c] for c in active_box_classes}

    fig = _build_figure(plot_signal, plot_leads, plot_fs, zoom_factor, boxes=boxes)

    # --- ringkasan hasil klasifikasi -> kotak
    st.markdown('<p class="section-title">Temuan pada sinyal</p>', unsafe_allow_html=True)
    if active_box_classes:
        chips = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:6px;margin:0 var(--s-3) var(--s-2) 0;'
            f'font-size:var(--text-secondary);color:var(--text);">'
            f'<span style="width:12px;height:12px;border:1.6px solid {CLASS_COLORS[c]};'
            f'background:{_hex_rgba(CLASS_COLORS[c], 0.18)};border-radius:2px;display:inline-block;"></span>'
            f'<b style="font-weight:600;">{c}</b> {BOX_TITLES[c]}</span>'
            for c in active_box_classes)
        st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
        if findings.get("ok"):
            for c in active_box_classes:
                s = findings["sel"][c]
                lead_txt = ", ".join(s["leads"]) or "—"
                tail = "" if s["meets"] else " · tidak ada lead yang melewati kriteria klasik (kotak putus-putus: lead dengan penyimpangan terbesar)"
                st.caption(f"**{c}** — {FINDING_NOTES[c]} Lead: {lead_txt}{tail}")
            st.caption(f"{findings['n_beats']} denyut terdeteksi · HR ≈ {findings['hr']:.0f} bpm · "
                       f"lebar QRS median ≈ {findings['qrs_ms']:.0f} ms. Kotak menandai gelombang yang diukur "
                       f"(Q, segmen ST–T, atau kompleks QRS) pada setiap denyut; arahkan kursor ke kotak untuk nilai ukur.")
            st.caption("Kotak dibuat dari pengukuran morfologi pada lead yang relevan dengan kelas yang diprediksi model — "
                       "model tidak mengeluarkan lokasi, sehingga ini bukan peta atensi dan bukan diagnosis.")
        else:
            st.info(findings.get("reason", "Kotak tidak dapat dibuat untuk rekaman ini."))
    else:
        st.caption("Model tidak memprediksi MI, STTC, maupun CD pada rekaman ini, sehingga tidak ada kotak yang digambar.")

    st.markdown(
        f'<p class="wide-plot-hint">Ditampilkan pada lebar {plot_zoom}. Geser ke samping di dalam panel '
        f'untuk melihat semua {n_leads_plot} lead.</p>',
        unsafe_allow_html=True,
    )

    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displaylogo": False})
    stage_html = f"""
    <style>
      html, body {{ margin: 0; background: transparent; }}
     .stage {{ overflow-x: auto; overscroll-behavior-x: contain; max-width: 100%;
          border-radius: 12px; border: 1px solid rgba(233,231,226,0.1);
          background: {ECG_PAPER_BG}; padding: 6px 6px 2px 6px; }}
      .stage::-webkit-scrollbar {{ height: 8px; }}
      .stage::-webkit-scrollbar-track {{ background: transparent; }}
      .stage::-webkit-scrollbar-thumb {{ background: rgba(233,231,226,0.2); border-radius: 6px; }}
        @media (hover: hover) {{
  .stage::-webkit-scrollbar-thumb:hover {{ background: rgba(233,231,226,0.4); }}
}}
    </style>
    <div class="stage">
        <div style="width:{int(1380 * zoom_factor)}px;">{plot_html}</div>
    </div>
    """
    components.html(stage_html, height=88 * n_leads_plot + 48, scrolling=False)

    if preds.get("HYP"):
        _render_hyp_panel(hyp_findings, hea_units)
    else:
        st.caption("HYP tidak diprediksi oleh model; kriteria Sokolow-Lyon dan Cornell hanya ditampilkan saat HYP terdeteksi.")