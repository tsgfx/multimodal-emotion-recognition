"""
多模态情绪识别演示窗口
基于 Gradio 的交互式 Web 界面

Usage:
    # 先在本地准备好模型和特征目录
    python src/demo.py \
        --audio_checkpoint outputs/checkpoints/audio_wav2vec2_finetune.5class.pt \
        --face_checkpoint outputs/checkpoints/face_resnet18.5class_aug.pt \
        --wav2vec2_pretrained /path/to/wav2vec2-base \
        --share
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _sanitize_proxy_env_for_gradio() -> None:
    """Avoid Gradio/httpx import failures when SOCKS proxy support is absent."""
    try:
        import socksio  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = os.environ.get(key, "")
        if value.lower().startswith(("socks://", "socks5://", "socks5h://", "socks4://")):
            os.environ.pop(key, None)


_sanitize_proxy_env_for_gradio()

import gradio as gr

from config import (
    CHECKPOINT_DIR,
    FACE_IMAGE_SIZE,
    FACE_FRAMES_PER_SAMPLE,
    LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
    AUDIO_SAMPLE_RATE,
    AUDIO_DURATION,
)
from models import build_audio_model, FaceResNet, resolve_pretrained_model_name_or_path
from train_utils import load_checkpoint

try:
    import librosa
except ModuleNotFoundError:
    raise ModuleNotFoundError("librosa required: pip install librosa")


# ── Per-class fusion weights (from classwise_weighted_average) ─────────────────
ALPHA_AUDIO = {
    "angry": 0.8,
    "disgust": 0.5,
    "fearful": 0.2,
    "happy": 0.5,
    "sad": 0.5,
}
EMOTION_COLORS = {
    "angry": "#e74c3c",
    "disgust": "#9b59b6",
    "fearful": "#3498db",
    "happy": "#f1c40f",
    "sad": "#2ecc71",
}


def load_models(
    audio_ckpt: Path,
    face_ckpt: Path,
    wav2vec2_pretrained: str,
    device: torch.device,
):
    wav2vec2_pretrained = resolve_pretrained_model_name_or_path(wav2vec2_pretrained)
    print(f"Wav2Vec2 pretrained: {wav2vec2_pretrained}")
    audio_model = build_audio_model(
        audio_model="wav2vec2_finetune",
        num_classes=len(LABELS),
        dropout=0.3,
        wav2vec2_pretrained=wav2vec2_pretrained,
    ).to(device)
    load_checkpoint(audio_model, audio_ckpt, device)
    audio_model.eval()

    face_model = FaceResNet(num_classes=len(LABELS), pretrained=False).to(device)
    load_checkpoint(face_model, face_ckpt, device)
    face_model.eval()

    return audio_model, face_model


def process_audio(audio_path: Path, device: torch.device) -> torch.Tensor:
    waveform, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    target_len = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return torch.from_numpy(waveform.astype(np.float32)).to(device)


def process_video_frames(video_path: Path, device: torch.device) -> torch.Tensor:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        frames.append(tensor)
    cap.release()

    if not frames:
        raise ValueError("No frames extracted from video")

    if len(frames) >= FACE_FRAMES_PER_SAMPLE:
        indices = np.linspace(0, len(frames) - 1, FACE_FRAMES_PER_SAMPLE).astype(int)
        selected = [frames[i] for i in indices]
    else:
        selected = frames + [frames[-1]] * (FACE_FRAMES_PER_SAMPLE - len(frames))

    return torch.stack(selected, dim=0).to(device)


@torch.no_grad()
def predict(
    audio_path: Path | None,
    video_path: Path | None,
    audio_model,
    face_model,
    device: torch.device,
):
    results = {}

    if audio_path is not None:
        waveform = process_audio(audio_path, device)
        audio_logits = audio_model(waveform.unsqueeze(0))
        audio_probs = torch.softmax(audio_logits, dim=1).cpu().numpy()[0]
        for i, label in enumerate(LABELS):
            results[f"audio_{label}"] = float(audio_probs[i])

    if video_path is not None:
        faces = process_video_frames(video_path, device)
        face_logits = face_model(faces.unsqueeze(0))
        face_probs = torch.softmax(face_logits, dim=1).cpu().numpy()[0]
        for i, label in enumerate(LABELS):
            results[f"face_{label}"] = float(face_probs[i])

    # Fusion using classwise weights
    if audio_path is not None and video_path is not None:
        for i, label in enumerate(LABELS):
            alpha = ALPHA_AUDIO[label]
            fused = alpha * audio_probs[i] + (1 - alpha) * face_probs[i]
            results[f"fusion_{label}"] = float(fused)
        results["has_fusion"] = True
    else:
        results["has_fusion"] = False

    return results


def build_ui(audio_model, face_model, device: torch.device):
    def inference(audio_file, video_file):
        if audio_file is None and video_file is None:
            return {l: 0.0 for l in LABELS}, "请上传音频或视频"

        audio_path = Path(audio_file) if audio_file is not None else None
        video_path = Path(video_file) if video_file is not None else None

        results = predict(audio_path, video_path, audio_model, face_model, device)

        if audio_path and not video_path:
            label_dict = {l: float(results.get(f"audio_{l}", 0.0)) for l in LABELS}
            return label_dict, "仅音频输入"
        elif video_path and not audio_path:
            label_dict = {l: float(results.get(f"face_{l}", 0.0)) for l in LABELS}
            return label_dict, "仅视频输入"
        else:
            fusion_vals = [float(results.get(f"fusion_{l}", 0.0)) for l in LABELS]
            pred_idx = int(np.argmax(fusion_vals))
            pred_label = LABELS[pred_idx]
            pred_conf = fusion_vals[pred_idx]
            label_dict = {l: float(v) for l, v in zip(LABELS, fusion_vals)}
            result_text = f"融合预测：{pred_label}（置信度 {pred_conf:.1%}）"
            return label_dict, result_text

    with gr.Blocks(title="多模态情绪识别演示") as demo:
        gr.Markdown(
            """
            # 多模态情绪识别演示
            **5 类情绪分类**：angry / disgust / fearful / happy / sad

            上传音频和/或视频，系统将分别输出各模态预测概率及融合结果。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="语音输入（可选）",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                audio_info = gr.Markdown("*支持 WAV/MP3 格式，16kHz 采样*")

            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="视频输入（可选）",
                    sources=["upload"],
                )
                video_info = gr.Markdown("*支持 MP4/AVI 格式，系统自动抽帧*")

        run_btn = gr.Button("开始识别", variant="primary", size="lg")

        gr.Markdown("---")
        gr.Markdown("### 识别结果")

        with gr.Row():
            label_box = gr.Label(label="各情绪概率", num_top_classes=5, show_label=True)

        result_text = gr.Textbox(label="融合预测结果", lines=1, interactive=False)

        run_btn.click(
            fn=inference,
            inputs=[audio_input, video_input],
            outputs=[label_box, result_text],
        )

        gr.Markdown("---")
        gr.Markdown(
            """
            ### 说明
            - **仅语音**：使用 Wav2Vec2 fine-tune 模型预测
            - **仅视频**：使用 ResNet18 人脸模型预测
            - **两者都有**：使用 classwise 加权融合（α_audio: angry=0.8, fearful=0.2, 其余=0.5）
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="多模态情绪识别演示窗口")
    parser.add_argument("--audio_checkpoint", type=Path,
        default=CHECKPOINT_DIR / "audio_wav2vec2_finetune.5class.pt")
    parser.add_argument("--face_checkpoint", type=Path,
        default=CHECKPOINT_DIR / "face_resnet18.5class_aug.pt")
    parser.add_argument("--wav2vec2_pretrained", type=str,
        default="wav2vec2-base")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--dry_run", action="store_true", help="Only load models and exit without launching Gradio")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device}")

    audio_model, face_model = load_models(
        args.audio_checkpoint,
        args.face_checkpoint,
        args.wav2vec2_pretrained,
        device,
    )
    print("Models loaded")
    if args.dry_run:
        return

    demo = build_ui(audio_model, face_model, device)
    demo.launch(server_port=args.port, share=args.share, max_file_size="500MB")


if __name__ == "__main__":
    main()
