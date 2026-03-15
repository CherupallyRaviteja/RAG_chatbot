import cv2
import numpy as np

def pad_image(img, pad=20):
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

if __name__ == "__main__":
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    padded = pad_image(img)
    print("✅ Image padded:", padded.shape)
