import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import your class (Assuming your file is named cnn.py)
from cnn import FastConvNet

def train_and_visualize():
    print("--- Loading Digits Dataset ---")
    # 1. Load Data (8x8 Digits - simpler than 28x28 MNIST, perfect for quick CPU test)
    digits = load_digits()
    X, y = digits.images, digits.target

    # 2. Preprocessing
    # Expand dims to (N, Channels, H, W) -> (1797, 1, 8, 8)
    X = X[:, np.newaxis, :, :]

    # Normalize to 0-1 range
    X = X / 16.0

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Initialize Your Hybrid CNN
    # We use a small architecture for speed
    cnn = FastConvNet(
        n_filters=8,        # 8 C++ Kernels
        kernel_size=3,
        pool_size=2,
        dense_layers=[64, 64],  # One dense layer
        max_iter=15,        # 15 Epochs
        eta=0.05,           # Learning Rate
        batch_size=16,
        verbose=1,
        random_state=42
    )

    print("\n--- Starting C++ Hybrid Training ---")
    cnn.fit(X_train, y_train)

    # 4. Evaluation
    print("\n--- Evaluating ---")
    y_pred = cnn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # 5. Visualization
    fig = plt.figure(figsize=(14, 6))

    # Create a layout: 1 row, 2 columns.
    # The right column will be further split for the digits.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])

    # --- LEFT PANEL: Loss Curve ---
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(cnn.losses_, label='Training Loss', linewidth=2)
    if cnn.val_losses_:
        ax_loss.plot(cnn.val_losses_, label='Validation Loss', linestyle='--')
    ax_loss.set_title('Training History (C++ Engine)')
    ax_loss.set_xlabel('Batch Updates')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # --- RIGHT PANEL: Predictions Grid ---
    # Create a 4x4 sub-grid inside the right panel
    gs_digits = gs[0, 1].subgridspec(4, 4, wspace=0.1, hspace=0.4)

    fig.text(0.75, 0.92, f'Sample Predictions (Acc: {acc*100:.1f}%)',
             ha='center', fontsize=12, weight='bold')

    for i in range(16):
        ax_digit = fig.add_subplot(gs_digits[i // 4, i % 4])

        # Reshape (1, 8, 8) -> (8, 8)
        img = X_test[i, 0]
        ax_digit.imshow(img, cmap='gray_r') # 'gray_r' inverts colors (like ink on paper)

        # Color-coded labels
        is_correct = y_pred[i] == y_test[i]
        color = 'green' if is_correct else 'red'
        label_text = f"P:{y_pred[i]}\nT:{y_test[i]}"

        ax_digit.set_title(label_text, color=color, fontsize=9)
        ax_digit.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_visualize()