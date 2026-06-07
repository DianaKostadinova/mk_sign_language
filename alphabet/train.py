

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed for saving plots
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "landmarks"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 300     # augmented samples generated per letter per epoch
EPOCHS            = 60
BATCH_SIZE        = 64
LR                = 1e-3
HIDDEN            = [256, 128, 64]
DROPOUT           = 0.3


# ── Online augmentation ───────────────────────────────────────────────────────
def augment(landmarks: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a (63,) normalized landmark vector.
    All operations keep the data physically plausible.
    """
    pts = landmarks.reshape(21, 3).copy()

    # 1. Random rotation around the z-axis (in-plane, ±20°)
    angle = np.random.uniform(-20, 20) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0],
                    [sin_a,  cos_a, 0],
                    [0,      0,     1]], dtype=np.float32)
    pts = pts @ rot.T

    # 2. Random scale (±10%)
    scale = np.random.uniform(0.9, 1.1)
    pts *= scale

    # 3. Gaussian noise (tiny jitter on each joint)
    pts += np.random.normal(0, 0.02, pts.shape).astype(np.float32)

    # 4. Random horizontal flip (50% chance — mirrors hand shape)
    if np.random.rand() < 0.5:
        pts[:, 0] *= -1

    return pts.flatten()


# ── Dataset ───────────────────────────────────────────────────────────────────
class LetterDataset(Dataset):
    """
    Each __getitem__ call picks a random raw sample for that class index
    and returns a freshly augmented version of it.
    Ensures the model never sees the exact same floats twice.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, samples_per_class: int, augment_fn):
        self.X                = X                  # (N_classes, 63)
        self.y                = y                  # (N_classes,) int labels
        self.samples_per_class = samples_per_class
        self.augment_fn       = augment_fn
        self.classes          = np.unique(y)
        self.n_classes        = len(self.classes)

        # Group indices by class for quick lookup
        self.class_indices = {c: np.where(y == c)[0] for c in self.classes}

    def __len__(self):
        return self.n_classes * self.samples_per_class

    def __getitem__(self, idx):
        # Which class does this index belong to?
        class_label = self.classes[idx % self.n_classes]

        # Pick a random raw sample for this class (only 1 here, but future-proof)
        raw_idx = np.random.choice(self.class_indices[class_label])
        raw     = self.X[raw_idx]

        aug = self.augment_fn(raw).astype(np.float32)
        return torch.from_numpy(aug), torch.tensor(int(class_label), dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], n_classes: int, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model, loader, criterion, optimizer, device):
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
    acc = (logits.argmax(1) == y_t).float().mean().item()
    preds = logits.argmax(1).cpu().numpy()
    return acc, preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load raw data
    X_raw    = np.load(DATA_DIR / "letters.npy")        # (29, 63)
    y_labels = np.load(DATA_DIR / "letters_labels.npy") # (29,) str

    # Encode string labels → integers
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)                     # (29,) int

    n_classes = len(le.classes_)
    input_dim = X_raw.shape[1]
    print(f"Classes: {n_classes}  |  Input dim: {input_dim}")
    print(f"Letters: {list(le.classes_)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # With only 1 sample per class, we use leave-one-out style validation:
    # generate augmented samples, hold out a fixed chunk as validation
    # Generate a fixed validation set (50 augmented samples per class)
    VAL_SAMPLES = 50
    X_val_list, y_val_list = [], []
    for cls in range(n_classes):
        idx = np.where(y == cls)[0][0]
        for _ in range(VAL_SAMPLES):
            X_val_list.append(augment(X_raw[idx]))
            y_val_list.append(cls)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int64)

    # Training dataset (online augmentation)
    train_dataset = LetterDataset(X_raw, y, SAMPLES_PER_CLASS, augment)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Model, loss, optimizer
    model     = MLP(input_dim, HIDDEN, n_classes, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train
    best_val_acc  = 0.0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_acc, _            = evaluate(model, X_val, y_val, device)
        scheduler.step()

        marker = " ← best" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"val_acc={val_acc:.3f}{marker}")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")

    # Save best model
    model.load_state_dict(best_model_state)
    torch.save({
        "model_state": best_model_state,
        "hidden_dims": HIDDEN,
        "input_dim":   input_dim,
        "n_classes":   n_classes,
        "dropout":     DROPOUT,
    }, MODELS_DIR / "mlp_letters.pt")
    np.save(MODELS_DIR / "label_encoder.npy", le.classes_)
    print(f"Model saved → {MODELS_DIR / 'mlp_letters.pt'}")

    # Confusion matrix
    _, val_preds = evaluate(model, X_val, y_val, device)
    cm   = confusion_matrix(y_val, val_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Letter Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Confusion matrix → {MODELS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
