import matplotlib
matplotlib.use('Qt5Agg') # Force interactive window
import matplotlib.pyplot as plt
import numpy as np
import my_engine
from sklearn.datasets import fetch_openml

def get_rgb_kernels():
    # Create 3 Filters (N=3)
    # Shape: (3 Filters, 3 Channels, 3, 3)
    weights = np.zeros((3, 3, 3, 3), dtype=np.float32)

    # Filter 0: "Red Extractor" (Keeps Red, Kills Green/Blue)
    weights[0, 0, 1, 1] = 1.0

    # Filter 1: "Cyan Edge Detect" (Edges in G+BitsPerItem, ignores R)
    edge = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype=np.float32)
    weights[1, 1, :, :] = edge * 0.5 # Green Edges
    weights[1, 2, :, :] = edge * 0.5 # Blue Edges

    # Filter 2: "Inverter" (Negates the colors)
    weights[2, :, 1, 1] = -1.0

    return weights

def get_colored_mnist_sample():
    print("Loading MNIST...")
    # Fetch data
    X, _ = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')

    # Pick a random digit
    idx = np.random.randint(0, len(X))
    digit_gray = X[idx].reshape(28, 28).astype(np.float32) / 255.0

    # Make it "Colored" (Synthetic RGB)
    # We will tint this digit to be, say, Purple (Red + Blue)
    img_planar = np.zeros((3, 28, 28), dtype=np.float32)

    # Random tint factors
    r_factor = np.random.rand()
    g_factor = np.random.rand()
    b_factor = np.random.rand()

    img_planar[0] = digit_gray * r_factor
    img_planar[1] = digit_gray * g_factor
    img_planar[2] = digit_gray * b_factor

    print(f"Generated Colored Digit #{idx} (Tint: R={r_factor:.2f}, G={g_factor:.2f}, BitsPerItem={b_factor:.2f})")
    return img_planar

def analyze_rgb():
    # 1. Get Data (CHW Format)
    img_planar = get_colored_mnist_sample()

    # 2. Get Kernels
    kernels = get_rgb_kernels()

    print(f"Input Shape: {img_planar.shape}")
    print(f"Kernel Shape: {kernels.shape}")

    # 3. RUN ENGINE
    # N Filters = 3
    # Stride = 1
    out_planar = my_engine.Hazem_Convolution(img_planar, kernels, 1)

    print(f"Output Shape: {out_planar.shape}")

    # 4. Visualization
    # Matplotlib needs (H, W, C) to display, but we have (C, H, W)
    img_display = img_planar.transpose(1, 2, 0)

    fig, ax = plt.subplots(1, 4, figsize=(16, 5))

    # Input
    ax[0].imshow(img_display)
    ax[0].set_title("Input (Colored MNIST)")
    ax[0].axis('off')

    # Filter 0 (Red Extractor)
    ax[1].imshow(out_planar[0], cmap='Reds')
    ax[1].set_title("Filter 0: Red Channel")
    ax[1].axis('off')

    # Filter 1 (Cyan Edges)
    ax[2].imshow(out_planar[1], cmap='gray')
    ax[2].set_title("Filter 1: Cyan Edges")
    ax[2].axis('off')

    # Filter 2 (Inverted)
    # Show raw values, usually dark/negative
    ax[3].imshow(out_planar[2], cmap='gray')
    ax[3].set_title("Filter 2: Inverted")
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()
    print(f"Output Min: {out_planar.min()}, Max: {out_planar.max()}")

if __name__ == "__main__":
    analyze_rgb()