import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

def main():
    # 1) デバイスの用意 (GPU が使えなければ CPU フォールバック)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2) モデルのロード（weights を明示的に指定）
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).to(device).eval()

    # 3) torchvision 推奨の前処理パイプライン
    preprocess = weights.transforms()

    # 4) ビデオキャプチャ設定
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mosaic_level = 16  # モザイク度合い

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- BGR→RGB に変換し PIL Image 化 ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # --- 前処理 + バッチ次元 + デバイス移動 ---
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # --- 推論 ---
        with torch.no_grad():
            out = model(input_tensor)['out']
        pred = out.argmax(1).squeeze().cpu().numpy()

        # --- 人物クラス(15)のマスク作成 ---
        mask = (pred == 15).astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- モザイク画像の作成 ---
        small = cv2.resize(
            frame,
            (frame.shape[1] // mosaic_level, frame.shape[0] // mosaic_level),
            interpolation=cv2.INTER_NEAREST
        )
        frame_mosaic = cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- マスク部分だけモザイクを合成 ---
        composite = np.where(mask[..., None] == 255, frame_mosaic, frame)

        cv2.imshow('Privacy Filter', composite)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
