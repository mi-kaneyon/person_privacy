#!/usr/bin/env python3
import cv2
import numpy as np
import pytesseract
import torch
from torchvision import models, transforms
import time

# モデルの準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT").to(device)
model.eval()

# セグメンテーション用の変換
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# キャッシュの準備
previous_person_mask = None
previous_text_blocks = []

def segment_person(frame):
    """セグメンテーションで人物領域を検出し、マスクを生成。"""
    input_tensor = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = torch.argmax(output, dim=0).cpu().numpy().astype(np.uint8)
    person_mask = (output_predictions == 15)
    return person_mask

def apply_mosaic_to_mask(frame, mask, block_size=12):
    """セグメンテーションマスクに基づいてモザイクを適用。"""
    if mask.sum() == 0:
        return frame
    mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), np.uint8)
    mask_resized = cv2.dilate(mask_resized, kernel, iterations=1)
    for y in range(0, frame.shape[0], block_size):
        for x in range(0, frame.shape[1], block_size):
            if mask_resized[y:y+block_size, x:x+block_size].any():
                h = min(block_size, frame.shape[0] - y)
                w = min(block_size, frame.shape[1] - x)
                roi = frame[y:y+h, x:x+w]
                small = cv2.resize(roi, (1, 1), interpolation=cv2.INTER_LINEAR)
                frame[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame

def preprocess_image(frame):
    """画像の前処理（文字領域を強調）。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def detect_text_blocks(frame, conf_threshold=15):
    """Tesseract OCRで文字・数字領域を検出。"""
    processed = preprocess_image(frame)
    config = "--psm 11 --oem 3"
    data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)

    blocks = []
    for i in range(len(data["level"])):
        conf = int(data["conf"][i])
        if conf > conf_threshold:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            blocks.append((x, y, w, h))
    return blocks

def apply_mosaic_to_blocks(frame, blocks, block_size=8):
    """モザイク処理を文字・数字領域に適用。"""
    for (x, y, w, h) in blocks:
        x1 = max(0, x - 3)
        y1 = max(0, y - 3)
        x2 = min(frame.shape[1], x + w + 3)
        y2 = min(frame.shape[0], y + h + 3)
        width = x2 - x1
        height = y2 - y1
        if width > 0 and height > 0:
            roi = frame[y1:y2, x1:x2]
            small = cv2.resize(roi, (max(1, width//8), max(1, height//8)), interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Press 'q' to quit.")
    prev_frame_time = 0
    frame_count = 0
    global previous_person_mask, previous_text_blocks

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # 人物セグメンテーション（キャッシュ処理）
        if frame_count % 5 == 0 or previous_person_mask is None:
            person_mask = segment_person(frame)
            previous_person_mask = person_mask
        else:
            person_mask = previous_person_mask

        person_masked_frame = apply_mosaic_to_mask(frame.copy(), person_mask)

        # 文字領域検出（キャッシュ処理）
        if frame_count % 5 == 0 or not previous_text_blocks:
            text_blocks = detect_text_blocks(person_masked_frame, conf_threshold=15)
            previous_text_blocks = text_blocks
        else:
            text_blocks = previous_text_blocks

        fully_masked_frame = apply_mosaic_to_blocks(person_masked_frame, text_blocks)

        # メモリクリア
        if device.type == "cuda":
            torch.cuda.empty_cache()

        end_time = time.time()
        fps = 1 / (end_time - start_time)

        cv2.putText(fully_masked_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Masked Frame", fully_masked_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
