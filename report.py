# report.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

def _plot_risk_bar(proba):
    fig, ax = plt.subplots(figsize=(6,0.8))
    ax.barh([0],[proba], color="#ef553b")
    ax.barh([0],[1-proba], left=[proba], color="#e6e6e6")
    ax.set_xlim(0,1)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def _feat_table_image(features, top_n=20):
    df = pd.DataFrame(list(features.items()), columns=["feature","value"])
    df = df.sort_values("feature").head(top_n)
    fig, ax = plt.subplots(figsize=(6, len(df)*0.25 + 0.5))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1,1.2)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def _waveform_image(audio_bytes):
    try:
        import librosa, io
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        fig, ax = plt.subplots(figsize=(6,1.5))
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y, linewidth=0.4)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

def build_report_bytes(inference, bundle=None, title="NeuroMindX Report"):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-50, title)
    ts = inference.get("ts", datetime.datetime.now())
    if hasattr(ts, "strftime"):
        ts = ts.strftime("%Y-%m-%d %H:%M:%S")
    c.setFont("Helvetica", 9)
    c.drawString(40, h-68, f"Generated: {ts}")
    proba = float(inference.get("proba", 0.0))
    bar = _plot_risk_bar(proba)
    c.drawImage(ImageReader(BytesIO(bar)), 40, h-140, width=400, height=60)
    # summary
    c.setFont("Helvetica", 10)
    if proba>0.7:
        summary = "High model estimate — consider clinical evaluation."
    elif proba>0.4:
        summary = "Moderate model estimate — monitor and consider consultation."
    else:
        summary = "Low model estimate."
    text = c.beginText(40, h-160)
    text.setFont("Helvetica", 10)
    text.textLines(["Summary:", summary, "Note: This is an experimental screening aid."])
    c.drawText(text)
    # SHAP if in bundle
    if bundle and bundle.get("last_shap_png"):
        try:
            c.drawImage(ImageReader(BytesIO(bundle["last_shap_png"])), 40, h-420, width=500, height=200)
        except Exception:
            pass
    # features table
    feat_png = _feat_table_image(inference.get("features", {}), top_n=18)
    c.drawImage(ImageReader(BytesIO(feat_png)), 40, h-720, width=500, height=200)
    # waveform if audio present
    audio_bytes = inference.get("audio_bytes", None) or inference.get("features", {}).get("audio_bytes")
    if audio_bytes:
        wave = _waveform_image(audio_bytes)
        if wave:
            c.drawImage(ImageReader(BytesIO(wave)), 40, h-920, width=500, height=100)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()
