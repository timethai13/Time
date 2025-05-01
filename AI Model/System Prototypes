import os
import cv2
import torch
import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from torchvision import models, transforms
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
VIDEO_PATH = "moodeng_stream.mp4"
FRAME_DIR = "data/frames"
AUDIO_DIR = "data/audio_chunks"
IMAGE_FEATURES_DIR = "data/features/image"
AUDIO_FEATURES_DIR = "data/features/audio"
FRAME_FPS = 1
CHUNK_DURATION_SEC = 5

# ----------------------------
# STEP 1: CREATE DIRECTORIES
# ----------------------------
for d in [FRAME_DIR, AUDIO_DIR, IMAGE_FEATURES_DIR, AUDIO_FEATURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ----------------------------
# STEP 2: EXTRACT FRAMES
# ----------------------------
def extract_video_frames(video_path, output_dir, fps=1):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    step = int(frame_rate / fps)
    frame_idx = 0
    saved = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % step == 0:
            out_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1
    cap.release()
    print(f"[INFO] Extracted {saved} frames.")

# ----------------------------
# STEP 3: EXTRACT AUDIO CHUNKS
# ----------------------------
def extract_audio_chunks(video_path, output_dir, chunk_duration_sec=5):
    video = VideoFileClip(video_path)
    temp_audio_path = os.path.join(output_dir, "temp_audio.wav")
    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)

    audio = AudioSegment.from_wav(temp_audio_path)
    total_ms = len(audio)
    chunk_ms = chunk_duration_sec * 1000
    count = 0

    for i in range(0, total_ms, chunk_ms):
        chunk = audio[i:i + chunk_ms]
        chunk.export(os.path.join(output_dir, f"chunk_{count:03d}.wav"), format="wav")
        count += 1

    os.remove(temp_audio_path)
    print(f"[INFO] Extracted {count} audio chunks.")

# ----------------------------
# STEP 4: EXTRACT IMAGE FEATURES (ResNet50)
# ----------------------------
def extract_image_features(frame_dir, output_dir):
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(frame_dir)), desc="Image Features"):
            if fname.endswith(".jpg"):
                img = cv2.imread(os.path.join(frame_dir, fname))
                img_tensor = preprocess(img).unsqueeze(0)  # Add batch dim
                feat = model(img_tensor).squeeze().numpy()  # (2048,)
                out_path = os.path.join(output_dir, fname.replace(".jpg", ".npy"))
                np.save(out_path, feat)

# ----------------------------
# STEP 5: EXTRACT AUDIO FEATURES (Wav2Vec2)
# ----------------------------
def extract_audio_features(audio_dir, output_dir):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    with torch.no_grad():
        for fname in tqdm(sorted(os.listdir(audio_dir)), desc="Audio Features"):
            if fname.endswith(".wav"):
                audio_input, sr = sf.read(os.path.join(audio_dir, fname))
                if len(audio_input.shape) > 1:
                    audio_input = audio_input.mean(axis=1)  # Mono

                inputs = processor(audio_input, sampling_rate=sr, return_tensors="pt")
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state  # (1, time, 768)
                pooled = last_hidden.mean(dim=1).squeeze().numpy()  # (768,)
                out_path = os.path.join(output_dir, fname.replace(".wav", ".npy"))
                np.save(out_path, pooled)

# ----------------------------
# RUN ALL STEPS
# ----------------------------
if __name__ == "__main__":
    extract_video_frames(VIDEO_PATH, FRAME_DIR, fps=FRAME_FPS)
    extract_audio_chunks(VIDEO_PATH, AUDIO_DIR, chunk_duration_sec=CHUNK_DURATION_SEC)
    extract_image_features(FRAME_DIR, IMAGE_FEATURES_DIR)
    extract_audio_features(AUDIO_DIR, AUDIO_FEATURES_DIR)
    print("[ALL DONE]")
