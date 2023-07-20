import cv2
import torch
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms import functional as F

# Load the model with pre-trained weights
model = deeplabv3_resnet101(pretrained=True)
model = model.cuda()
model.eval()

cap = cv2.VideoCapture(0)

# Set the video resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mosaic_level = 16  # Adjust mosaic level as desired

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to PIL format
    image = F.to_tensor(frame).unsqueeze(0).cuda()  # Move the input tensor to GPU

    # Perform inference
    with torch.no_grad():
        output = model(image)['out']
    output_predictions = output.argmax(1).squeeze().cpu().numpy()

    # Create a mask and composite image
    mask = output_predictions == 15  # 15 is the label for 'person' in COCO
    mask = mask.astype(np.uint8) * 255  # Convert the mask to uint8 for compatibility

    # Resize the mask to match the frame dimensions
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mosaic effect only to the person's body
    frame_mosaic = cv2.resize(frame, (frame.shape[1] // mosaic_level, frame.shape[0] // mosaic_level),
                              interpolation=cv2.INTER_NEAREST)
    frame_mosaic = cv2.resize(frame_mosaic, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a composite image by applying mosaic effect only to the person
    composite = np.where(mask_resized[..., None], frame_mosaic, frame)

    # Start or stop recording based on key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if not recording:
            out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
    elif key == ord('w') or key == ord('q'):
        if recording:
            out.release()
            recording = False

    # Write the frame to the output file if recording is enabled
    if recording:
        out.write(composite)

    cv2.imshow('frame', composite)
    if key == ord('q'):
        break

# Release everything when done
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
