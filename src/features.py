# src/features.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    EMB_AVAILABLE = True
except Exception:
    EMB_AVAILABLE = False

try:
    import librosa
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

TFIDF_MAX = 250
EMB_NAME = "all-MiniLM-L6-v2"
EMB_MODEL = None

def _flatten_rts(df):
    df = df.copy()
    df['rt_mean'] = df['reaction_times'].apply(lambda x: float(np.mean(x)) if hasattr(x, "__iter__") else float(x))
    df['rt_std'] = df['reaction_times'].apply(lambda x: float(np.std(x)) if hasattr(x, "__iter__") else 0.0)
    df['rt_min'] = df['reaction_times'].apply(lambda x: float(np.min(x)) if hasattr(x, "__iter__") else float(x))
    df['rt_max'] = df['reaction_times'].apply(lambda x: float(np.max(x)) if hasattr(x, "__iter__") else float(x))
    return df

def tfidf_fit_transform(corpus, max_features=TFIDF_MAX):
    vect = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vect.fit_transform(corpus)
    cols = [f"tfidf_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X.toarray(), columns=cols), vect

def tfidf_transform(corpus, vect):
    X = vect.transform(corpus)
    cols = [f"tfidf_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X.toarray(), columns=cols)

def embed_texts(corpus):
    if not EMB_AVAILABLE:
        return pd.DataFrame(np.zeros((len(corpus), 1)), columns=["emb_fallback"])
    global EMB_MODEL
    if EMB_MODEL is None:
        EMB_MODEL = SentenceTransformer(EMB_NAME)
    emb = EMB_MODEL.encode(corpus, show_progress_bar=False)
    cols = [f"emb_{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, columns=cols)

def audio_features_from_bytes(byte_content, sr=16000):
    out = {}
    audio_cols = [f"mfcc_mean_{i}" for i in range(13)] + ["zcr_mean","rmse_mean","tempo"]
    if not AUDIO_AVAILABLE or byte_content is None:
        return {c:0.0 for c in audio_cols}
    try:
        import io
        y, sr = librosa.load(io.BytesIO(byte_content), sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i, v in enumerate(mfcc.mean(axis=1)):
            out[f"mfcc_mean_{i}"] = float(v)
        out["zcr_mean"] = float(librosa.feature.zero_crossing_rate(y).mean())
        out["rmse_mean"] = float(librosa.feature.rms(y).mean())
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        out["tempo"] = float(tempo[0]) if len(tempo)>0 else 0.0
        return out
    except Exception:
        return {c:0.0 for c in audio_cols}

def build_feature_dataframe(df, tfidf_vect=None, fit_tfidf=False):
    df = _flatten_rts(df).reset_index(drop=True)
    texts = df['text_response'].fillna("").astype(str).tolist()
    if tfidf_vect is None and fit_tfidf:
        tfidf_df, vect = tfidf_fit_transform(texts)
    elif tfidf_vect is not None:
        tfidf_df = tfidf_transform(texts, tfidf_vect); vect = tfidf_vect
    else:
        tfidf_df, vect = tfidf_fit_transform(texts)
    emb_df = embed_texts(texts)
    behavior = df[["rt_mean","rt_std","rt_min","rt_max","age"]].reset_index(drop=True)
    audio_cols = [f"mfcc_mean_{i}" for i in range(13)] + ["zcr_mean","rmse_mean","tempo"]
    audio_list = []
    for idx, row in df.iterrows():
        audio_bytes = row.get("audio_bytes", None)
        feats = audio_features_from_bytes(audio_bytes)
        audio_list.append(feats)
    audio_df = pd.DataFrame(audio_list)
    feat = pd.concat([behavior.reset_index(drop=True), tfidf_df.reset_index(drop=True),
                      emb_df.reset_index(drop=True), audio_df.reset_index(drop=True)], axis=1)
    feat.fillna(0, inplace=True)
    return feat, vect
def extract_features(text):
    # Example: Convert text into simple numeric features
    features = {
        "length": len(text),
        "num_words": len(text.split()),
        "avg_word_length": sum(len(w) for w in text.split()) / (len(text.split()) or 1)
    }
    return features
