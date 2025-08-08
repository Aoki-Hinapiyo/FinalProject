import os
import json
import librosa
import numpy as np
import pandas as pd
import openl3

# ===== 参数配置 =====
MUSIC_DIR = r"E:\GitProject\FinalProject\MusicReference"
EXCEL_PATH = r"E:\GitProject\FinalProject\MusicList.xlsx"
OUTPUT_JSON = r"E:\GitProject\FinalProject\features.json"
SAMPLE_RATE = 48000
EMBEDDING_SIZE = 6144
WINDOW_SIZE = 3.0  # 每片段时长
HOP_SIZE = 1.5     # 滑动步长

# ===== 读取元数据 =====
meta_df = pd.read_excel(EXCEL_PATH)
meta_dict = {row["Name"]: {"Artist": row["Artist"], "Source": row["Source"]}
             for _, row in meta_df.iterrows()}

# ===== OpenL3 模型加载 =====
model = openl3.models.load_audio_embedding_model(
    input_repr="mel256", content_type="music", embedding_size=EMBEDDING_SIZE
)

# ===== 音频预处理（单声道+音量标准化+预加重滤波） =====
def preprocess_audio(y, sr):
    # 单声道
    if y.ndim > 1:
        y = librosa.to_mono(y)
    # 音量标准化
    y = librosa.util.normalize(y)
    # 预加重滤波（增强高频，改善嘈杂环境特征表现）
    y = librosa.effects.preemphasis(y)
    return y

# ===== 多窗口特征提取 =====
def extract_openl3_multiwindow(file_path, sr=SAMPLE_RATE, window=WINDOW_SIZE, hop=HOP_SIZE):
    y, _ = librosa.load(file_path, sr=sr, mono=False)  # 先读原音频
    y = preprocess_audio(y, sr)
    if len(y) < sr * window:
        y = np.pad(y, (0, sr * window - len(y)))
    embeddings = []
    for start in np.arange(0, len(y)/sr - window, hop):
        y_segment = y[int(start*sr): int((start+window)*sr)]
        if len(y_segment) < sr * window:
            continue
        emb, _ = openl3.get_audio_embedding(y_segment, sr, model=model, center=True, hop_size=1.0)
        embeddings.append(np.mean(emb, axis=0))
    if len(embeddings) == 0:
        return np.zeros(EMBEDDING_SIZE)
    return np.mean(embeddings, axis=0).astype(np.float32).tolist()  # 平均池化

# ===== 主程序 =====
def build_feature_db():
    feature_db = {}
    for file in os.listdir(MUSIC_DIR):
        if file.lower().endswith((".flac", ".wav", ".mp3", ".ogg")):
            file_path = os.path.join(MUSIC_DIR, file)
            name = os.path.splitext(file)[0]
            print(f"Processing: {file} ...")
            feature_vector = extract_openl3_multiwindow(file_path)

            if name in meta_dict:
                meta = meta_dict[name]
            else:
                meta = {"Artist": "Unknown", "Source": "Unknown"}

            feature_db[file] = {
                "Name": name,
                "Artist": meta["Artist"],
                "Source": meta["Source"],
                "Feature": feature_vector
            }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_db, f, ensure_ascii=False)
    print(f"特征库已保存到 {OUTPUT_JSON}")

if __name__ == "__main__":
    build_feature_db()
