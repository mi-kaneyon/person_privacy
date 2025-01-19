import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorReducer(nn.Module):
    def __init__(self, n_colors=16):
        super(ColorReducer, self).__init__()
        self.n_colors = n_colors
        self.palette = nn.Parameter(torch.rand(n_colors, 3))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, 3)
        distances = torch.cdist(x, self.palette)  # (B, H*W, n_colors)
        labels = torch.argmin(distances, dim=2)   # (B, H*W)
        reduced = self.palette[labels].reshape(B, H, W, 3).permute(0, 3, 1, 2)
        return reduced

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorReducer(n_colors=16).to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB + Torch
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            frame_t = frame_t.to(device)  # (1, 3, H, W)

            # Forward pass
            reduced_t = model(frame_t)
            # Convert back to NumPy (B, C, H, W)
            reduced_np = (reduced_t * 255).byte().squeeze(0).permute(1,2,0).cpu().numpy()
            # RGB -> BGR
            reduced_bgr = cv2.cvtColor(reduced_np, cv2.COLOR_RGB2BGR)

            cv2.imshow("Real-time color reduce (PyTorch)", reduced_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
