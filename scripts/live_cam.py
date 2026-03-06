import cv2
import numpy as np
import my_engine
import time

def get_preservation_kernels():
    # We need 3 Filters (since 1 view is Original)
    # Shape: (3 Filters, 3 Channels, 5 Height, 5 Width)
    # Note: We are using 5x5 kernels now for more visible effects!
    kernels = np.zeros((3, 3, 5, 5), dtype=np.float32)

    # --- Filter 0: Gaussian Blur (5x5) ---
    # This creates a "Bell Curve" of weights. Center is heavy, edges are light.
    # It smooths the image without destroying it.
    # We use OpenCV to generate the 1D kernel, then matrix multiply to get 2D.
    g_1d = cv2.getGaussianKernel(5, sigma=1.5) # 5x1
    g_2d = g_1d @ g_1d.T # 5x5

    # Apply to all RGB channels equally
    kernels[0, 0, :, :] = g_2d
    kernels[0, 1, :, :] = g_2d
    kernels[0, 2, :, :] = g_2d

    # --- Filter 1: Sharpen (3x3 padded to 5x5) ---
    # A standard sharpen kernel is 3x3. We center it in a 5x5 zeros array.
    sharp_3x3 = np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]], dtype=np.float32)

    # Pad to 5x5 (1 pixel border of zeros)
    sharp_5x5 = np.pad(sharp_3x3, ((1,1), (1,1)), 'constant')

    kernels[1, 0, :, :] = sharp_5x5
    kernels[1, 1, :, :] = sharp_5x5
    kernels[1, 2, :, :] = sharp_5x5

    # --- Filter 2: Box Blur (5x5) ---
    # Simple average. Every pixel is 1/25th of the sum.
    # Makes things look "blocky" or out of focus.
    box_k = np.ones((5, 5), dtype=np.float32) / 25.0

    kernels[2, 0, :, :] = box_k
    kernels[2, 1, :, :] = box_k
    kernels[2, 2, :, :] = box_k

    return kernels

def live_convolution():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    kernels = get_preservation_kernels()

    print("--- PRESERVATION FILTER DEMO ---")
    print("Top-Left: Original Color")
    print("Others: Processed (Grayscale Intensity)")
    print("Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Resize
        h, w = 240, 320
        small_frame = cv2.resize(frame, (w, h))

        # 2. Viewport 1: ORIGINAL (Color)
        view1_original = small_frame

        # 3. Input Transpose
        input_tensor = small_frame.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)

        # 4. RUN C++ ENGINE
        # Output is (3 Filters, 240, 320) -> 3 Grayscale maps
        output_tensor = my_engine.Hazem_Convolution(input_tensor, kernels, 1)

        # 5. Post-Process
        # FIX: Handle 2D output (H, W) correctly
        def to_img(tensor_slice):
            # tensor_slice is (H, W) floats

            # Clip to 0-1 (Sharpen can go > 1 or < 0)
            img = np.clip(tensor_slice, 0, 1)

            # Convert to uint8 (0-255)
            img = (img * 255).astype(np.uint8)

            # Convert Grayscale -> BGR so we can stack it with the Color Original
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img

        # Apply the fix to all outputs
        view2_blur = to_img(output_tensor[0])
        view3_sharp = to_img(output_tensor[1])
        view4_box = to_img(output_tensor[2])

        # 6. Grid Layout
        top_row = np.hstack((view1_original, view2_blur))
        bot_row = np.hstack((view3_sharp, view4_box))
        grid_display = np.vstack((top_row, bot_row))

        # 7. FPS & Labels
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid_display, f"FPS: {fps:.1f}", (w-40, h+20), font, 0.6, (255,255,255), 2)
        cv2.putText(grid_display, "ORIGINAL", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(grid_display, "GAUSSIAN", (w+10, 30), font, 0.7, (0, 255, 255), 2)
        cv2.putText(grid_display, "SHARPEN", (10, h+30), font, 0.7, (0, 0, 255), 2)
        cv2.putText(grid_display, "BOX BLUR", (w+10, h+30), font, 0.7, (255, 0, 255), 2)

        cv2.imshow('Hazem Engine - Standard Filters', grid_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_convolution()