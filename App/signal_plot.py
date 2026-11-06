"""Tab 'Sinyal 12-lead': plot EKG bergaya kertas grid dengan panel yang bisa digeser."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from constants import LEAD_NAMES

ECG_PAPER_BG = "#141518"
ECG_MINOR_GRID = "rgba(233,231,226,0.05)"
ECG_MAJOR_GRID = "rgba(233,231,226,0.13)"
ECG_TRACE_COLOR = "#E9E7E2"
ECG_TEXT_COLOR = "#9A9892"


def _build_figure(plot_signal, plot_leads, plot_fs, zoom_factor):
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
    return fig


def render_signal_tab(raw_signal, fs, sig_names, resampled, sr):
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

    fig = _build_figure(plot_signal, plot_leads, plot_fs, zoom_factor)

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
