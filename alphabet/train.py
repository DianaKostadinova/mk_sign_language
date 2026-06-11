

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
    
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "landmarks"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN           = 20    # must match extract_landmarks.py
SAMPLES_PER_CLASS = 600
EPOCHS            = 80
BATCH_SIZE        = 64
LR                = 1e-3
LSTM_HIDDEN       = 128
LSTM_LAYERS       = 2
DROPOUT           = 0.3


# ── Sequence augmentation ─────────────────────────────────────────────────────
HAND_DIM = 63   # 21 landmarks × 3
ARM_DIM  = 18   #  6 landmarks × 3


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    seq      = seq.copy()
    T, feat_dim = seq.shape

    angle    = np.random.uniform(-25, 25) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0],
                    [sin_a,  cos_a, 0],
                    [0,      0,     1]], dtype=np.float32)

    scale = np.random.uniform(0.85, 1.15)
    flip  = np.random.rand() < 0.5

    shear_x = np.random.uniform(-0.15, 0.15)
    shear_y = np.random.uniform(-0.15, 0.15)
    shear = np.array([[1,       shear_x, 0],
                      [shear_y, 1,       0],
                      [0,       0,       1]], dtype=np.float32)
    transform = rot @ shear

    hand_mask = (np.random.rand(21) > 0.1).astype(np.float32)
    arm_mask  = (np.random.rand(6)  > 0.1).astype(np.float32)

    # ── Hand: all frames at once (T, 21, 3) ──────────────────────────────────
    hand = seq[:, :HAND_DIM].reshape(T, 21, 3)
    hand = hand @ transform.T
    hand *= scale
    if flip:
        hand[:, :, 0] *= -1
    hand += np.random.normal(0, 0.02, hand.shape).astype(np.float32)
    hand *= hand_mask[None, :, None]
    seq[:, :HAND_DIM] = hand.reshape(T, HAND_DIM)

    # ── Arm: all frames at once ───────────────────────────────────────────────
    if feat_dim > HAND_DIM:
        n_arm = (feat_dim - HAND_DIM) // 3
        arm = seq[:, HAND_DIM:].reshape(T, n_arm, 3)
        arm = arm @ transform.T
        arm *= scale
        if flip:
            arm[:, :, 0] *= -1
        arm += np.random.normal(0, 0.01, arm.shape).astype(np.float32)
        arm *= arm_mask[None, :, None]
        seq[:, HAND_DIM:] = arm.reshape(T, feat_dim - HAND_DIM)

    # ── Time warp: vectorized linear interp, no per-dim loop ─────────────────
    warp_strength = np.random.uniform(0.8, 1.2)
    n_warped = max(2, int(T * warp_strength))
    t_src = np.clip(np.linspace(0, T - 1, n_warped), 0, T - 1)
    lo    = np.floor(t_src).astype(int)
    hi    = np.minimum(lo + 1, T - 1)
    alpha = (t_src - lo)[:, None]                        # (n_warped, 1)
    warped = seq[lo] * (1 - alpha) + seq[hi] * alpha    # (n_warped, feat_dim)

    indices = np.linspace(0, n_warped - 1, T, dtype=int)
    return warped[indices].astype(np.float32)


def mixup_sequences(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Interpolate between two sequences of the same class."""
    lam = np.random.beta(0.4, 0.4)
    return (lam * seq_a + (1 - lam) * seq_b).astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────
class LetterSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, samples_per_class: int):
        self.X               = X   # (N, SEQ_LEN, 63)
        self.y               = y   # (N,) int
        self.samples_per_class = samples_per_class
        self.classes         = np.unique(y)
        self.n_classes       = len(self.classes)
        self.class_indices   = {c: np.where(y == c)[0] for c in self.classes}

    def __len__(self):
        return self.n_classes * self.samples_per_class

    def __getitem__(self, idx):
        cls      = self.classes[idx % self.n_classes]
        raw_idx  = np.random.choice(self.class_indices[cls])
        aug      = augment_sequence(self.X[raw_idx])

        # MixUp: 50% chance — blend with another sequence from the same class
        if len(self.class_indices[cls]) > 1 and np.random.rand() < 0.5:
            idx2 = np.random.choice(self.class_indices[cls])
            aug2 = augment_sequence(self.X[idx2])
            aug  = mixup_sequences(aug, aug2)

        return torch.from_numpy(aug.astype(np.float32)), torch.tensor(int(cls), dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────
class SignLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, n_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        last_hidden  = h_n[-1]   # take the last layer's hidden state
        return self.head(last_hidden)


# ── Training / eval ───────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, X_val, y_val, device):
    model.eval()
    X_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_val).to(device)
    logits = model(X_t)
    preds  = logits.argmax(1).cpu().numpy()
    acc    = (preds == y_val).mean()
    return acc, preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    X_raw    = np.load(DATA_DIR / "letters_seq.npy")       # (N, SEQ_LEN, 63)
    y_labels = np.load(DATA_DIR / "letters_labels.npy")    # (N,) str

    le = LabelEncoder()
    y  = le.fit_transform(y_labels)

    n_classes = len(le.classes_)
    print(f"Classes: {n_classes}  |  Sequence shape: {X_raw.shape[1:]}")
    print(f"Letters: {list(le.classes_)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Stratified split — val samples are fully excluded from training
    X_train_raw, X_val_raw, y_train, y_val_raw = train_test_split(
        X_raw, y, test_size=0.15, stratify=y, random_state=42
    )

    # Augment the held-out val set once for a stable evaluation target
    VAL_AUG = 10
    X_val_list, y_val_list = [], []
    for x_seq, label in zip(X_val_raw, y_val_raw):
        for _ in range(VAL_AUG):
            X_val_list.append(augment_sequence(x_seq))
            y_val_list.append(label)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int64)

    train_dataset = LetterSeqDataset(X_train_raw, y_train, SAMPLES_PER_CLASS)
    train_loader  = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    input_dim = X_raw.shape[2]   # 81 with arm landmarks, 63 hand-only
    model     = SignLSTM(input_dim, LSTM_HIDDEN, LSTM_LAYERS, n_classes, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc     = 0.0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _            = evaluate(model, X_val, y_val, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            marker = " <- best" if val_acc >= best_val_acc else ""
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  "
                  f"train={train_acc:.3f}  val={val_acc:.3f}{marker}")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")

    # Save
    model.load_state_dict(best_model_state)
    input_dim = X_raw.shape[2]   # 81 with arm landmarks, 63 hand-only
    torch.save({
        "model_state": best_model_state,
        "lstm_hidden": LSTM_HIDDEN,
        "lstm_layers": LSTM_LAYERS,
        "input_dim":   input_dim,
        "seq_len":     SEQ_LEN,
        "n_classes":   n_classes,
        "dropout":     DROPOUT,
    }, MODELS_DIR / "lstm_letters.pt")
    np.save(MODELS_DIR / "label_encoder.npy", le.classes_)
    print(f"Model saved -> {MODELS_DIR / 'lstm_letters.pt'}")

    # Confusion matrix
    _, val_preds = evaluate(model, X_val, y_val, device)
    cm   = confusion_matrix(y_val, val_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Letter Classifier — Confusion Matrix (LSTM)")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Confusion matrix -> {MODELS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
