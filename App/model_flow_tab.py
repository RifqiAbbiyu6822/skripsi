"""
model_flow_tab.py
Tab "Alur Model" untuk app.py: menjelaskan alur kerja SE-ConvNeXt1D dari input
sampai keputusan label. Semua angka (kernel, stride, channel, shape, jumlah blok)
dibaca LANGSUNG dari model yang sedang dimuat, jadi otomatis ikut berubah kalau
checkpoint-nya varian nano/tiny/small.
"""
import math

import pandas as pd
import streamlit as st
import torch


# ---------------------------------------------------------------------------
# helper: ambil info arsitektur & shape asli dari model
# ---------------------------------------------------------------------------
def _trace_real(model, model_input) -> tuple:
    """
    Jalankan model pada sinyal ASLI yang diunggah (model_input, shape (12, L), sudah
    melewati preprocessing yang sama dengan inference). Forward hook mencatat:
      - shape output tiap tahap
      - rata-rata bobot SE (gate sigmoid) tiap stage untuk rekaman ini
    """
    device = next(model.parameters()).device
    x = torch.as_tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)

    shapes, hooks = {}, []
    se_vals = {i: [] for i in range(4)}

    def shape_hook(name):
        def hook(_m, _i, out):
            shapes[name] = tuple(out.shape)
        return hook

    def se_hook(stage_idx):
        def hook(_m, _i, out):
            se_vals[stage_idx].append(float(out.detach().mean().item()))
        return hook

    for i in range(4):
        hooks.append(model.downsample_layers[i].register_forward_hook(shape_hook(f"down{i}")))
        hooks.append(model.stages[i].register_forward_hook(shape_hook(f"stage{i}")))
        for blk in model.stages[i]:
            hooks.append(blk.se.fc.register_forward_hook(se_hook(i)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    se_means = [sum(v) / len(v) if v else float("nan") for v in se_vals.values()]
    return shapes, se_means


def _read_arch(model) -> dict:
    stem_conv = model.downsample_layers[0][0]
    info = {
        "stem": dict(
            cin=stem_conv.in_channels, cout=stem_conv.out_channels,
            k=stem_conv.kernel_size[0], s=stem_conv.stride[0], p=stem_conv.padding[0],
        ),
        "down": [],
        "stages": [],
        "head_in": model.head.in_features,
        "head_out": model.head.out_features,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    for i in range(1, 4):
        conv = model.downsample_layers[i][1]
        info["down"].append(dict(
            cin=conv.in_channels, cout=conv.out_channels,
            k=conv.kernel_size[0], s=conv.stride[0],
        ))
    for i in range(4):
        blk = model.stages[i][0]
        dim = blk.dwconv.in_channels
        info["stages"].append(dict(
            depth=len(model.stages[i]), dim=dim,
            dw_k=blk.dwconv.kernel_size[0],
            expand=blk.pwconv1.out_features // dim,
            se_hidden=blk.se.fc[0].out_features,
            dp_first=getattr(model.stages[i][0].drop_path, "drop_prob", 0.0),
            dp_last=getattr(model.stages[i][-1].drop_path, "drop_prob", 0.0),
        ))
    return info


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# penjelasan singkat tiap kelas (bahasa awam, bukan klaim kalau model punya cabang khusus)
CLASS_NOTES = {
    "NORM": "Rekaman EKG tanpa kelainan diagnostik yang menonjol. Kelas mayoritas di PTB-XL.",
    "MI": "Infark miokard. Biasanya terlihat dari perubahan segmen ST dan gelombang Q yang muncul di kelompok lead tertentu.",
    "STTC": "Perubahan segmen ST/gelombang T (iskemia, dll). Pola ini sifatnya morfologi gelombang, jadi dibantu kernel depthwise.",
    "CD": "Gangguan konduksi (misalnya blok cabang berkas). Cirinya QRS melebar atau bentuknya berubah.",
    "HYP": "Hipertrofi. Kriteria klinisnya bandingin amplitudo antar-lead (Sokolow-Lyon, Cornell), makanya normalisasi global dipakai supaya rasio antar-lead nggak hilang.",
}


# ---------------------------------------------------------------------------
# fungsi utama yang dipanggil dari app.py
# ---------------------------------------------------------------------------
DEFAULT_LABELS_LONG = {
    "NORM": "EKG Normal",
    "MI": "Infark Miokard",
    "STTC": "Perubahan ST/T",
    "CD": "Gangguan Konduksi",
    "HYP": "Hipertrofi",
}
DEFAULT_COLORS = {
    "NORM": "#22C55E", "MI": "#EF4444", "STTC": "#A855F7",
    "CD": "#F59E0B", "HYP": "#3B82F6",
}


def render_model_flow_tab(model, ptb_config, classes, model_input, probs, thresholds, preds,
                          class_labels_long=None, class_colors=None):
    # label & warna opsional: kalau app.py nggak punya, pakai default di modul ini
    class_labels_long = class_labels_long or DEFAULT_LABELS_LONG
    class_colors = class_colors or DEFAULT_COLORS

    arch = _read_arch(model)
    shapes, se_means = _trace_real(model, model_input)

    st_ = arch["stem"]
    n_leads, length = model_input.shape

    st.markdown("### Alur Kerja Model")
    st.caption(
        "Penjelasan ini dibaca langsung dari model yang lagi dipakai dan dijalankan pada rekaman "
        "yang kamu unggah, jadi kernel, channel, dan shape di bawah sesuai checkpoint dan sinyal aslinya."
    )

    # ---------- 1. ringkasan alur ----------
    st.markdown("#### 1. Gambaran umum alur")
    flow = ["Input", "Stem", "Stage 1", "Down 1", "Stage 2", "Down 2",
            "Stage 3", "Down 3", "Stage 4", "GAP", "Tail (Norm + Linear)", "Sigmoid + Threshold"]
    chips = ' <span style="opacity:.6">→</span> '.join(
        f'<span class="legend-chip">{n}</span>' for n in flow
    )
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown(
        f"Singkatnya, sinyal 12-lead (**{n_leads} × {length}** sampel, {ptb_config.sampling_rate} Hz) "
        f"dipadatkan dulu panjangnya oleh *stem*, lalu diproses empat *stage* yang tiap stage-nya "
        f"berisi beberapa blok ConvNeXt1D + SE. Di antara stage ada *downsampling* yang "
        f"menurunkan panjang sinyal sambil menambah jumlah channel. Setelah itu fitur dirata-rata "
        f"sepanjang waktu (GAP), dinormalisasi, lalu dipetakan ke **{arch['head_out']} logit** "
        f"(satu per kelas). Logit diubah jadi probabilitas pakai sigmoid dan dibandingkan dengan "
        f"threshold tiap kelas."
    )

    # ---------- 2. tabel tahapan ----------
    st.markdown("#### 2. Rincian tiap tahap (backbone)")

    def fmt(shape):  # (1, C, L) -> "C × L"
        return f"{shape[1]} × {shape[2]}"

    rows = [{
        "Tahap": "Input",
        "Operasi": "Sinyal 12-lead sudah difilter 0,5–40 Hz & dinormalisasi global",
        "Kernel / Stride": "-",
        "Channel": n_leads,
        "Output (C × L)": f"{n_leads} × {length}",
    }, {
        "Tahap": "Stem (patchify)",
        "Operasi": "Conv1d besar + LayerNorm1d",
        "Kernel / Stride": f"k={st_['k']}, s={st_['s']}, pad={st_['p']}",
        "Channel": f"{st_['cin']} → {st_['cout']}",
        "Output (C × L)": fmt(shapes["down0"]),
    }]
    for i in range(4):
        sg = arch["stages"][i]
        rows.append({
            "Tahap": f"Stage {i + 1}",
            "Operasi": f"{sg['depth']} × blok ConvNeXt1D + SE",
            "Kernel / Stride": f"depthwise k={sg['dw_k']}, s=1",
            "Channel": sg["dim"],
            "Output (C × L)": fmt(shapes[f"stage{i}"]),
        })
        if i < 3:
            d = arch["down"][i]
            rows.append({
                "Tahap": f"Downsampling {i + 1}",
                "Operasi": "LayerNorm1d + Conv1d",
                "Kernel / Stride": f"k={d['k']}, s={d['s']}",
                "Channel": f"{d['cin']} → {d['cout']}",
                "Output (C × L)": fmt(shapes[f"down{i + 1}"]),
            })
    rows.append({
        "Tahap": "GAP",
        "Operasi": "Global average pooling sepanjang waktu",
        "Kernel / Stride": "-",
        "Channel": arch["head_in"],
        "Output (C × L)": f"{arch['head_in']}",
    })
    st.dataframe(pd.DataFrame(rows).astype(str), width="stretch", hide_index=True)
    st.caption(
        "Kolom terakhir itu shape asli hasil forward pada rekaman yang kamu unggah, bukan hitungan manual atau data contoh. "
        "Panjang sinyal makin pendek tiap downsampling, tapi jumlah channel makin banyak."
    )

    # ---------- 3. isi satu blok ----------
    st.markdown("#### 3. Isi satu blok ConvNeXt1D + SE")
    sg0 = arch["stages"][0]
    with st.expander("Lihat urutan operasi di dalam satu blok", expanded=False):
        st.markdown(
            f"""
Satu blok dikerjakan berurutan seperti ini (angka diambil dari model):

1. **Depthwise Conv1d**, kernel **{sg0['dw_k']}**, `groups = jumlah channel`. Tiap channel dikonvolusi sendiri-sendiri, jadi cuma mencampur informasi di sumbu waktu.
2. **LayerNorm** (setelah tensor dipindah ke format *N, L, C*).
3. **Linear expand** ×**{sg0['expand']}** (inverted bottleneck, misal 64 → 256).
4. **GELU** sebagai aktivasi.
5. **Linear project** balik ke jumlah channel awal.
6. **Layer scale** (parameter γ, inisialisasi 1e-6) supaya awal training blok ini "hampir nol" dan stabil.
7. **SE block**: GAP → Linear (C → C/16) → ReLU → Linear (C/16 → C) → Sigmoid, hasilnya dikali ke tiap channel.
8. **Residual + DropPath**: output ditambah input awal, dan sebagian blok dilewati secara acak saat training (stochastic depth).

Ukuran bottleneck SE (reduction 16, minimal 4) tiap stage, plus rata-rata bobot SE yang muncul pada rekamanmu (0 = channel ditekan, 1 = dilewatkan penuh):
"""
        )
        se_df = pd.DataFrame([{
            "Stage": f"Stage {i + 1}",
            "Jumlah blok": s["depth"],
            "Channel (C)": s["dim"],
            "SE: C → C/r": f"{s['dim']} → {s['se_hidden']} → {s['dim']}",
            "Drop path (blok awal → akhir)": f"{s['dp_first']:.3f} → {s['dp_last']:.3f}",
            "Rata-rata bobot SE (rekaman ini)": f"{se_means[i]:.3f}",
        } for i, s in enumerate(arch["stages"])])
        st.dataframe(se_df, width="stretch", hide_index=True)

    # ---------- 4. tail / head ----------
    st.markdown("#### 4. Tail (classifier head)")
    st.markdown(
        f"Setelah stage 4, output berbentuk **{fmt(shapes['stage3'])}**. Langkah akhirnya:\n\n"
        f"- **Global average pooling** merata-rata sumbu waktu sehingga jadi vektor **{arch['head_in']}** dimensi.\n"
        f"- **LayerNorm({arch['head_in']})** untuk menstabilkan skala fitur.\n"
        f"- **Linear({arch['head_in']} → {arch['head_out']})** menghasilkan 5 logit.\n"
        f"- **Sigmoid** per logit (bukan softmax) karena kasusnya multi-label, jadi satu rekaman boleh "
        f"punya lebih dari satu label aktif."
    )

    # ---------- 5. per kelas ----------
    st.markdown("#### 5. Alur keputusan tiap kelas (rekaman yang diunggah)")
    st.markdown(
        "Backbone-nya **dipakai bersama** oleh semua kelas, jadi nggak ada cabang khusus per kelas. "
        "Yang membedakan hanyalah satu neuron keluaran dan satu threshold untuk tiap kelas."
    )
    cls_rows = []
    for i, c in enumerate(classes):
        p = float(probs[i])
        cls_rows.append({
            "Kelas": c,
            "Keterangan": class_labels_long[c],
            "Neuron head": f"Linear[{i}]",
            "Logit": round(_logit(p), 3),
            "Probabilitas (sigmoid)": round(p, 4),
            "Threshold": round(float(thresholds[c]), 2),
            "Keputusan": "AKTIF" if preds[c] else "tidak aktif",
        })
    st.dataframe(pd.DataFrame(cls_rows), width="stretch", hide_index=True)
    st.caption("Aturannya: label aktif kalau probabilitas ≥ threshold kelas itu.")

    with st.expander("Catatan tiap kelas", expanded=False):
        for c in classes:
            color = class_colors[c]
            st.markdown(
                f'<span class="legend-chip"><span class="legend-dot" style="background:{color}"></span>'
                f'{c}</span> {CLASS_NOTES.get(c, "")}',
                unsafe_allow_html=True,
            )

    # ---------- 6. ukuran model ----------
    st.markdown("#### 6. Ukuran model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total parameter", f"{arch['n_params']:,}")
    c2.metric("Total blok", sum(s["depth"] for s in arch["stages"]))
    c3.metric("Downsampling total", f"×{st_['s'] * 2 ** 3}")
    st.caption(
        f"Panjang sinyal turun dari {length} ke {shapes['stage3'][2]} lewat stem (stride {st_['s']}) "
        f"dan 3 downsampling (stride 2 masing-masing)."
    )