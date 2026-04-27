from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw" / "ravdess"
PROCESSED_ROOT = DATA_ROOT / "processed"
AUDIO_FEATURE_DIR = PROCESSED_ROOT / "audio_features"
WAV2VEC2_EMBED_DIR = PROCESSED_ROOT / "wav2vec2_embeddings"
FACE_FRAME_DIR = PROCESSED_ROOT / "face_frames"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
FIGURE_DIR = OUTPUT_ROOT / "figures"
METRIC_DIR = OUTPUT_ROOT / "metrics"

TARGET_EMOTIONS = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
}

LABELS = ["angry", "disgust", "fearful", "happy", "sad"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 3.0
N_MELS = 64
N_MFCC = 20
FACE_IMAGE_SIZE = 224
FACE_FRAMES_PER_SAMPLE = 8

RANDOM_SEED = 42
