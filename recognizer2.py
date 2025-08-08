import os
import json
import librosa
import numpy as np
import openl3
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# ===== 参数配置 =====
FEATURE_JSON = r"E:\GitProject\FinalProject\features.json"
SAMPLE_RATE = 48000
EMBEDDING_SIZE = 6144
WINDOW_SIZE = 3.0  # 每个切片的长度（秒）
HOP_SIZE = 1.0     # 滑动步长（秒）
TOP_K = 3          # 输出 Top-k 匹配结果

# ===== 加载模型 =====
model = openl3.models.load_audio_embedding_model(
    input_repr="mel256", content_type="music", embedding_size=EMBEDDING_SIZE
)

# ===== 音频预处理函数 =====
def preprocess_audio(y, sr):
    if y.ndim > 1:
        y = librosa.to_mono(y)
    y = librosa.util.normalize(y)
    y = librosa.effects.preemphasis(y)
    return y

# ===== 单段提取 OpenL3 特征 =====
def extract_feature(y, sr):
    emb, _ = openl3.get_audio_embedding(y, sr, model=model, center=True, hop_size=1.0)
    return np.mean(emb, axis=0)

# ===== 加载特征库 =====
def load_feature_database():
    with open(FEATURE_JSON, "r", encoding="utf-8") as f:
        db = json.load(f)
    names, artists, sources, features = [], [], [], []
    for file, meta in db.items():
        names.append(meta["Name"])
        artists.append(meta["Artist"])
        sources.append(meta["Source"])
        features.append(meta["Feature"])
    features = np.array(features, dtype=np.float32)
    return names, artists, sources, features

# ===== 滑动窗口识别函数 =====
def recognize_sliding(input_path):
    print(f"\n[INFO] 正在处理音频文件：{input_path}")
    # sf.info() 注释，避免 .m4a 报错
    # info = sf.info(input_path)
    # print(f"采样率: {info.samplerate} | 声道: {info.channels} | 时长: {round(info.duration,2)} 秒")

    y, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=False)
    y = preprocess_audio(y, SAMPLE_RATE)

    names, artists, sources, db_features = load_feature_database()

    matches = []
    times = []

    total_duration = len(y) / SAMPLE_RATE
    for start in np.arange(0, total_duration - WINDOW_SIZE, HOP_SIZE):
        y_seg = y[int(start * SAMPLE_RATE): int((start + WINDOW_SIZE) * SAMPLE_RATE)]
        if len(y_seg) < SAMPLE_RATE * WINDOW_SIZE:
            continue
        vec = extract_feature(y_seg, SAMPLE_RATE).reshape(1, -1)
        sims = cosine_similarity(vec, db_features)[0]
        top_idx = sims.argmax()
        matches.append(names[top_idx])
        times.append((start, start + WINDOW_SIZE))
        print(f"[{round(start,1)}s - {round(start+WINDOW_SIZE,1)}s] → 匹配: {names[top_idx]} | 匹配率: {sims[top_idx]*100:.2f}%")

    print("\n📊 Top 匹配统计（投票机制）:")
    counter = Counter(matches)
    for name, count in counter.most_common(TOP_K):
        print(f"{name} 出现 {count} 次")

if __name__ == "__main__":
    path = input("请输入要识别的音频路径：").strip('"')
    if not os.path.exists(path):
        print("❌ 输入路径无效，请检查。")
    else:
        recognize_sliding(path)
