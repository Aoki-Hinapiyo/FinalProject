import os
import json
import librosa
import numpy as np
import openl3
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity

# ===== 参数配置 =====
FEATURE_JSON = r"E:\GitProject\FinalProject\features.json"
SAMPLE_RATE = 48000
EMBEDDING_SIZE = 6144
WINDOW_SIZE = 3.0
HOP_SIZE = 1.5

# ===== OpenL3 模型加载 =====
model = openl3.models.load_audio_embedding_model(
    input_repr="mel256", content_type="music", embedding_size=EMBEDDING_SIZE
)


# ===== 音频预处理 =====
def preprocess_audio(y, sr):
    if y.ndim > 1:
        y = librosa.to_mono(y)  # 强制单声道
    y = librosa.util.normalize(y)  # 音量标准化
    y = librosa.effects.preemphasis(y)  # 预加重
    return y


# ===== 多窗口特征提取 =====
def extract_openl3_multiwindow(file_path, sr=SAMPLE_RATE, window=WINDOW_SIZE, hop=HOP_SIZE):
    # 读取文件信息
    info = sf.info(file_path)
    print(f"\n输入文件信息：\n采样率: {info.samplerate} Hz | 声道: {info.channels} | 时长: {round(info.duration, 2)} 秒")

    y, _ = librosa.load(file_path, sr=sr, mono=False)
    y = preprocess_audio(y, sr)
    if len(y) < sr * window:
        y = np.pad(y, (0, sr * window - len(y)))
    embeddings = []
    for start in np.arange(0, len(y) / sr - window, hop):
        y_segment = y[int(start * sr): int((start + window) * sr)]
        if len(y_segment) < sr * window:
            continue
        emb, _ = openl3.get_audio_embedding(y_segment, sr, model=model, center=True, hop_size=1.0)
        embeddings.append(np.mean(emb, axis=0))
    if len(embeddings) == 0:
        return np.zeros(EMBEDDING_SIZE)
    return np.mean(embeddings, axis=0).astype(np.float32).reshape(1, -1)


# ===== 匹配函数 =====
def recognize_audio(input_path):
    # 加载特征库
    with open(FEATURE_JSON, "r", encoding="utf-8") as f:
        feature_db = json.load(f)

    # 提取输入音频特征
    print(f"\n提取特征中：{input_path}")
    input_vector = extract_openl3_multiwindow(input_path)

    # 计算相似度
    names, artists, sources, features = [], [], [], []
    for file, meta in feature_db.items():
        names.append(meta["Name"])
        artists.append(meta["Artist"])
        sources.append(meta["Source"])
        features.append(meta["Feature"])
    features = np.array(features, dtype=np.float32)
    similarities = cosine_similarity(input_vector, features)[0]

    # 取Top-3
    top_indices = similarities.argsort()[-3:][::-1]
    print("\nTop-3 匹配结果：")
    for i in top_indices:
        print(f"{names[i]} - {artists[i]}（{sources[i]}） | 匹配率：{similarities[i] * 100:.2f}%")


if __name__ == "__main__":
    # 输入音频路径
    input_path = input("请输入要识别的音频文件路径：").strip('"')
    if not os.path.exists(input_path):
        print("文件不存在，请检查路径！")
    else:
        recognize_audio(input_path)
