import cv2
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as F

#=======================================================
# 1. Load Mask R-CNN (instance segmentation)
#    - maskrcnn_resnet50_fpn_v2 requires PyTorch 2.0+
#    - If your environment is older, use maskrcnn_resnet50_fpn
#=======================================================
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
model = model.cuda()  # Use GPU (if available)
model.eval()

#=======================================================
# 2. Initialize webcam (or any video source)
#=======================================================
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Mosaic level (the higher the number, the coarser the mosaic)
mosaic_level = 16

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (PyTorch)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image to a PyTorch tensor and move to GPU
    input_tensor = F.to_tensor(rgb_frame).unsqueeze(0).cuda()

    #=======================================================
    # 3. Inference with Mask R-CNN
    #=======================================================
    with torch.no_grad():
        outputs = model(input_tensor)[0]
        # outputs is a dict with:
        #   - 'boxes':  [N,4]
        #   - 'labels': [N]
        #   - 'scores': [N]
        #   - 'masks':  [N,1,H,W]

    #=======================================================
    # 4. Identify all "person" instances
    #    COCO dataset: 'person' class ID = 1
    #=======================================================
    person_indices = []
    for i, label_id in enumerate(outputs['labels']):
        if label_id.item() == 1:  # 'person'
            person_indices.append(i)

    #=======================================================
    # 5. Create mosaic version of the frame
    #    (shrink -> enlarge) to produce a mosaic effect
    #=======================================================
    h, w = frame.shape[:2]
    frame_small = cv2.resize(frame, (w // mosaic_level, h // mosaic_level),
                             interpolation=cv2.INTER_NEAREST)
    frame_mosaic = cv2.resize(frame_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Prepare a composite image
    composite = frame.copy()

    #=======================================================
    # 6. Apply mosaic only to each person's region
    #=======================================================
    for i in person_indices:
        score = outputs['scores'][i].item()
        
        # Skip low-confidence detections (threshold 0.5 as example)
        if score < 0.5:
            continue
        
        # Retrieve the mask, shape = [1,H,W]
        mask = outputs['masks'][i][0]  # (H,W)
        mask = (mask > 0.5).byte().cpu().numpy()  # 0 or 1
        
        # Expand mask to 3 channels
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Overwrite the regions where mask=1 with frame_mosaic
        composite = np.where(mask_3ch == 1, frame_mosaic, composite)

        # (Optional) You can draw a bounding box or label if you want:
        # box = outputs['boxes'][i].cpu().numpy().astype(int)
        # x1, y1, x2, y2 = box
        # cv2.rectangle(composite, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(composite, "Masked", (x1, y1 - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    #=======================================================
    # 7. Show the result
    #=======================================================
    cv2.imshow("Frame", composite)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
