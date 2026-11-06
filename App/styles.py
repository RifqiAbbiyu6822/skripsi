"""CSS kustom aplikasi.

Design tokens
  Surfaces   #0F1012 canvas (60%) / #15161A surface + #1B1C21 raised (30%)
  Aksen      bone #E9E7E2 (10%): CTA, fokus, tab aktif. Warna kelas hanya untuk data.
  Lines      bone @ 8% dan 16%; panel = border 1px + highlight inset tipis
  Spacing    grid 8pt (mikro 4pt); prosa maks 70ch
  Type       Instrument Sans, 3 bobot (400 / 500 / 600)
  Viewport   narrow 375 / medium 768 / wide 1280 (satu breakpoint: < 768)
  Motion     150 / 200 / 320ms, hanya opacity dan transform
"""
import streamlit as st

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
  color-scheme: dark;
  --bg: #0F1012; --surface: #15161A; --raised: #1B1C21;
  --line: rgba(233,231,226,0.08); --line-strong: rgba(233,231,226,0.16);
  --highlight: inset 0 1px 0 rgba(255,255,255,0.04);
  --text: #E9E7E2; --muted: #A09E97; --faint: #8A8981;
  --good: #7FCB9A; --bad: #EE7B73; --warn: #E3B25A;
  --r-sm: 8px; --r-md: 12px; --r-lg: 16px;

  /* spacing (8pt, mikro 4pt) */
  --s-1: 4px; --s-2: 8px; --s-3: 12px; --s-4: 16px; --s-5: 24px;
  --s-6: 32px; --s-7: 48px; --s-8: 64px; --s-9: 80px;
  --prose: 70ch;

  /* type */
  --font-regular: 400; --font-medium: 500; --font-semibold: 600;
  --text-meta: 0.75rem; --text-secondary: 0.875rem; --text-body: 1rem;
  --text-h3: 1.25rem; --text-h1: clamp(1.75rem, 3.6vw, 2.5rem);
  --leading-tight: 1.2; --leading-body: 1.55; --leading-meta: 1.4;

  /* viewport */
  --vp-narrow: 375px; --vp-medium: 768px; --vp-wide: 1280px;
  --hit: 44px;
  --gutter: clamp(16px, 5vw, 72px);

  /* motion */
  --motion-150: 150ms; --motion-200: 200ms; --motion-320: 320ms;
  --ease: cubic-bezier(0.2, 0.7, 0.2, 1);
}

html, body, [class*="css"], .stApp {
  font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-variant-numeric: tabular-nums;
  font-size: var(--text-body);
  font-weight: var(--font-regular);
  -webkit-font-smoothing: antialiased;
}

#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
section[data-testid="stSidebar"] { display: none !important; }

.stApp { background: var(--bg); color: var(--text); }
.stApp p, .stApp label, .stApp li, .stApp span { color: inherit; }

.block-container {
  width: 100%;
  max-width: min(var(--vp-wide), 100%) !important;
  padding: var(--s-7) var(--gutter) var(--s-9) var(--gutter) !important;
}
div.element-container { margin-bottom: 0 !important; }

::selection { background: rgba(233,231,226,0.22); color: var(--text); }
:focus-visible { outline: 2px solid var(--text) !important; outline-offset: 2px; }
hr { border-color: var(--line) !important; }

/* --- Type -------------------------------------------------------------- */
h1, h2, h3, h4, h5 { color: var(--text) !important; font-family: inherit !important; line-height: var(--leading-tight) !important; }
h1 { font-weight: var(--font-semibold) !important; letter-spacing: -0.02em !important;
  font-size: var(--text-h1) !important; margin: 0 !important; padding: 0 !important; max-width: 20em; }
h3, h5 { font-weight: var(--font-medium) !important; font-size: var(--text-h3) !important; }
p { line-height: var(--leading-body); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; font-size: var(--text-secondary) !important; }

/* --- Hero ---------------------------------------------------------------- */
.hero { padding: 0; margin-bottom: var(--s-7); animation: riseIn var(--motion-320) var(--ease) both; }
.hero-eyebrow { margin: 0 0 var(--s-4) 0; font-size: var(--text-meta); line-height: var(--leading-meta);
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--faint); font-weight: var(--font-medium); }
.hero-sub { color: var(--muted); font-size: var(--text-body); margin: var(--s-4) 0 0 0; max-width: var(--prose); line-height: var(--leading-body); }

.ecg-trace-wrap { width: 100%; margin: var(--s-6) 0 0 0;
  -webkit-mask-image: linear-gradient(90deg, #000 70%, transparent);
  mask-image: linear-gradient(90deg, #000 70%, transparent); }
.ecg-trace-svg { width: 100%; height: auto; display: block; overflow: visible; }
.ecg-trace-path { stroke: var(--text); stroke-opacity: 0.7; stroke-width: 1.4; stroke-linejoin: round;
  stroke-dasharray: 1; stroke-dashoffset: 1; animation: ecgDraw 1.4s var(--ease) 0.15s forwards; }

/* --- Section headings ------------------------------------------------------ */
.section-title { font-size: var(--text-h3); font-weight: var(--font-medium); line-height: var(--leading-tight);
  color: var(--text); margin: 0 0 var(--s-2) 0; }
.section-hint { font-size: var(--text-secondary); color: var(--muted); margin: 0 0 var(--s-5) 0; max-width: var(--prose); }

/* --- Controls (target tap >= 44px) ----------------------------------------- */
.stButton > button, button[data-testid="stBaseButton-secondary"] {
  min-height: var(--hit); border-radius: var(--r-sm); font-size: var(--text-body); font-weight: var(--font-medium);
  border: 1px solid var(--line-strong); background: transparent; color: var(--text);
  transition: background var(--motion-150) var(--ease), border-color var(--motion-150) var(--ease),
              color var(--motion-150) var(--ease), transform var(--motion-150) var(--ease); }

.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  min-height: var(--hit); background: var(--text); color: #0F1012; border: 1px solid var(--text);
  font-weight: var(--font-semibold); padding: 12px 24px; }
.stButton > button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p { color: #0F1012; }
.stButton > button:active { transform: scale(0.98); }

@media (hover: hover) {
  .stButton > button:hover { background: var(--raised); border-color: rgba(233,231,226,0.32); color: var(--text); }
  .stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: #FFFFFF; border-color: #FFFFFF; color: #0F1012; }
  div[data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(233,231,226,0.4) !important; background: var(--raised) !important; }
  div[data-testid="stTabs"] button[role="tab"]:hover { color: var(--text); background: transparent; }
  .dx-row:hover { background: rgba(233,231,226,0.025); }
  .step:hover { border-color: rgba(233,231,226,0.28); }
}

.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
  min-height: var(--hit); border-radius: var(--r-sm) !important; background: var(--surface) !important;
  border: 1px solid var(--line) !important; }

div[data-testid="stFileUploader"] label p { color: var(--muted); font-size: var(--text-secondary); }
div[data-testid="stFileUploaderDropzone"] { min-height: var(--hit); background: var(--surface) !important;
  border: 1px dashed var(--line-strong) !important; border-radius: var(--r-md) !important;
  transition: border-color var(--motion-150) var(--ease), background var(--motion-150) var(--ease); }
div[data-testid="stFileUploaderDropzone"] small, div[data-testid="stFileUploaderDropzone"] span { color: var(--muted) !important; font-size: var(--text-secondary); }
div[data-testid="stFileUploaderFile"] { color: var(--text); }

div[role="radiogroup"] label { min-height: var(--hit); display: inline-flex; align-items: center;
  font-size: var(--text-body); color: var(--muted); transition: color var(--motion-150) var(--ease); }
div[role="radiogroup"] label:has(input:checked) { color: var(--text); }
div[data-testid="stToggle"] label { min-height: var(--hit); display: inline-flex; align-items: center; }
div[data-testid="stToggle"] label p { font-size: var(--text-secondary); color: var(--muted); }
div[data-testid="stSlider"] label p { font-size: var(--text-secondary); color: var(--muted); }

div[data-testid="stAlert"] { border-radius: var(--r-md); border: 1px solid var(--line-strong); background: var(--surface) !important;
  box-shadow: var(--highlight); animation: riseIn var(--motion-200) var(--ease) both; }
div[data-testid="stAlert"] > div { background: transparent !important; }
div[data-testid="stAlert"] p { color: var(--text); font-size: var(--text-secondary); max-width: var(--prose); }
div[data-testid="stToast"] { background: var(--raised); border: 1px solid var(--line-strong); }

/* --- Tabs ----------------------------------------------------------------- */
div[data-testid="stTabs"] { margin-top: var(--s-2); }
div[data-testid="stTabs"] button[role="tab"] { min-height: var(--hit); font-weight: var(--font-medium); color: var(--muted);
  font-size: var(--text-body); padding: 0 var(--s-1); margin-right: var(--s-5); transition: color var(--motion-150) var(--ease); }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--text); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--text) !important; height: 2px !important;
  transition: transform var(--motion-200) var(--ease), width var(--motion-200) var(--ease) !important; }
div[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: var(--line) !important; }
div[data-testid="stTabs"] [role="tablist"] { gap: 0; }
div[data-testid="stTabs"] [role="tabpanel"] { animation: riseIn var(--motion-200) var(--ease) both; }

/* --- Progress (tinggi tetap; hanya transform yang beranimasi) --------------- */
.scan-banner { margin: var(--s-5) 0 var(--s-2) 0; min-height: 40px; }
.scan-track { height: 2px; background: var(--raised); overflow: hidden; border-radius: 2px; }
.scan-fill { height: 100%; width: 100%; background: var(--text); transform-origin: left center;
  transition: transform var(--motion-320) var(--ease); }
.scan-label { margin-top: var(--s-3); font-size: var(--text-secondary); color: var(--muted); }

/* --- Recording summary (panel L1: border 1px + highlight) -------------------- */
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin: 0 0 var(--s-6) 0; padding: 0;
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-md); box-shadow: var(--highlight);
  overflow: hidden; animation: riseIn var(--motion-320) var(--ease) both; }
.meta div { display: flex; flex-direction: column; gap: var(--s-1); padding: var(--s-4) var(--s-5);
  border-left: 1px solid var(--line); }
.meta div:first-child { border-left: 0; }
.meta dt { font-size: var(--text-meta); line-height: var(--leading-meta); color: var(--faint); }
.meta dd { margin: 0; font-size: var(--text-body); color: var(--text); font-weight: var(--font-medium); }

/* --- Verdict + diagnosis rows ---------------------------------------------- */
.verdict { margin: var(--s-6) 0 var(--s-6) 0; animation: riseIn var(--motion-320) var(--ease) both; }
.verdict-label { font-size: var(--text-meta); line-height: var(--leading-meta); letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--faint); font-weight: var(--font-medium); margin: 0 0 var(--s-2) 0; }
.verdict-value { display: flex; flex-wrap: wrap; gap: var(--s-2) 32px; margin: 0; font-size: clamp(1.75rem, 3vw, 2.25rem);
  font-weight: var(--font-semibold); letter-spacing: -0.015em; line-height: var(--leading-tight); color: var(--text); }
.verdict-value.none { color: var(--muted); }
.verdict-item { display: inline-flex; align-items: center; gap: var(--s-3); }
.verdict-item i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

.dx-list { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-md);
  box-shadow: var(--highlight); overflow: hidden; }
.dx-row { display: grid; grid-template-columns: minmax(160px, 240px) 1fr 64px minmax(112px, 144px);
  align-items: center; gap: var(--s-5); padding: var(--s-5); border-top: 1px solid var(--line);
  transition: background var(--motion-150) var(--ease);
  animation: riseIn var(--motion-320) var(--ease) both; animation-delay: var(--d, 0s); }   /* --d diatur dari Python */
.dx-row:first-child { border-top: 0; }
.dx-name { display: flex; flex-direction: column; gap: var(--s-1); min-width: 0; }
.dx-code { display: inline-flex; align-items: center; gap: var(--s-3); font-size: var(--text-body); font-weight: var(--font-semibold); color: var(--muted); }
.dx-code i { width: 8px; height: 8px; border-radius: 50%; background: var(--c); opacity: 0.5; display: inline-block; }
.dx-long { font-size: var(--text-secondary); color: var(--faint); padding-left: 20px; }
.dx-row.active .dx-code { color: var(--text); }
.dx-row.active .dx-code i { opacity: 1; }
.dx-row.active .dx-long { color: var(--muted); }

.dx-bar { position: relative; height: 4px; background: var(--raised); border-radius: 4px; }
.dx-fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--c); border-radius: 4px; opacity: 0.4;
  transform-origin: left center;
  animation: barGrow var(--motion-320) var(--ease) both; animation-delay: calc(var(--d, 0s) + 160ms); }
.dx-row.active .dx-fill { opacity: 1; }
.dx-tick { position: absolute; top: -4px; width: 1px; height: 12px; background: var(--text); opacity: 0;
  animation: tickIn var(--motion-200) var(--ease) both; animation-delay: calc(var(--d, 0s) + 400ms); }

.dx-pct { text-align: right; font-size: var(--text-h3); font-weight: var(--font-medium); color: var(--muted); }
.dx-row.active .dx-pct { color: var(--text); }
.dx-state { text-align: right; display: flex; flex-direction: column; gap: 2px; font-size: var(--text-secondary); color: var(--faint); }
.dx-state b { font-weight: var(--font-medium); color: var(--faint); }
.dx-row.active .dx-state b { color: var(--c); }
.dx-state span { font-size: var(--text-meta); line-height: var(--leading-meta); color: var(--faint); }
.dx-note { font-size: var(--text-secondary); color: var(--faint); margin: var(--s-4) 0 0 0; max-width: var(--prose); }

/* --- Signal panel --------------------------------------------------------------- */
.wide-plot-hint { color: var(--faint); font-size: var(--text-secondary); margin: var(--s-4) 0 var(--s-3) 0; max-width: var(--prose); }

/* --- Empty state -------------------------------------------------------------------- */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--s-4); margin-top: var(--s-7); }
.step { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-md); box-shadow: var(--highlight);
  padding: var(--s-5); transition: border-color var(--motion-150) var(--ease);
  animation: riseIn var(--motion-320) var(--ease) both; animation-delay: var(--d, 0s); }
.step-num { font-size: var(--text-secondary); color: var(--faint); font-weight: var(--font-medium); }
.step-title { font-size: var(--text-body); font-weight: var(--font-medium); color: var(--text); margin: var(--s-2) 0; }
.step-desc { font-size: var(--text-secondary); color: var(--muted); line-height: var(--leading-body); margin: 0; }

.legend { margin-top: var(--s-8); }
.legend-title { font-size: var(--text-h3); font-weight: var(--font-medium); line-height: var(--leading-tight); color: var(--text); margin: 0 0 var(--s-4) 0; }
.legend-grid { display: flex; flex-wrap: wrap; gap: var(--s-3) 32px; }
.legend-item { display: inline-flex; align-items: center; gap: var(--s-3); font-size: var(--text-secondary); color: var(--muted); }
.legend-item i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.legend-item b { color: var(--text); font-weight: var(--font-semibold); }

.disclaimer { margin: var(--s-7) 0 0 0; padding-top: var(--s-4); border-top: 1px solid var(--line);
  font-size: var(--text-meta); line-height: var(--leading-meta); color: var(--faint); max-width: var(--prose); }

/* --- Motion (hanya opacity, transform, dan dash offset garis dekoratif) --------------- */
@keyframes fadeIn { from { opacity: 0; } }
@keyframes riseIn { from { opacity: 0; transform: translateY(8px); } }
@keyframes barGrow { from { transform: scaleX(0); } }
@keyframes tickIn { to { opacity: 0.6; } }
@keyframes ecgDraw { to { stroke-dashoffset: 0; } }

/* --- Narrow (< 768px) ----------------------------------------------------------------- */
@media (max-width: 767.98px) {
  :root { --gutter: 16px; }
  .block-container { padding-top: var(--s-5) !important; }
  .hero { margin-bottom: var(--s-6); }
  .meta { grid-template-columns: 1fr 1fr; }
  .meta div { padding: var(--s-3) var(--s-4); }
  .meta div:nth-child(odd) { border-left: 0; }
  .meta div:nth-child(n+3) { border-top: 1px solid var(--line); }
  .dx-row { grid-template-columns: 1fr auto; gap: var(--s-3) var(--s-4); padding: var(--s-4); }
  .dx-bar { grid-column: 1 / -1; grid-row: 2; }
  .dx-pct { grid-row: 1; grid-column: 2; }
  .dx-state { grid-column: 1 / -1; grid-row: 3; text-align: left; flex-direction: row; gap: var(--s-3); align-items: baseline; }
  .steps { gap: var(--s-3); }
  .step { padding: var(--s-4); }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-delay: 0ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
  .ecg-trace-path { stroke-dashoffset: 0; }
  .dx-tick { opacity: 0.6; }
}
"""


def inject_css():
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)