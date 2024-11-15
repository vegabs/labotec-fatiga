from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import requests

image = Image.open('data/img2.jpg')

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small-dwr')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small-dwr")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes

# print results
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
