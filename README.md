# YOLO-V8-Object-Detection-and-Segmentation
YOLO V8 Object detection and segmentation
---

![image](https://github.com/user-attachments/assets/775a2b49-406b-4fbc-85c1-934c5b415ff7)

#### How YOLO works ?
- YOLO (You Only Look Once) is a family of object detection models that can detect and classify objects in an image in a single forward pass of the neural network. Unlike traditional object detection methods that first generate region proposals and then classify them, YOLO treats object detection as a single regression problem, directly predicting bounding boxes and class probabilities from the entire image in one evaluation.

#### Key Concepts of YOLO:

![image](https://github.com/user-attachments/assets/5f39939b-f183-4ec3-9ca7-32acc8cd5077)


- Single Forward Pass: YOLO divides the input image into an S×S grid. (13x13 or 19x19 cells)
Each grid cell is responsible for predicting a certain number of bounding boxes and their corresponding confidence scores, as well as class probabilities.

- Bounding Box Predictions: Each grid cell predicts multiple bounding boxes(anchors) and their associated confidence scores. The confidence score reflects how certain the model is that the bounding box contains an object. The bounding box co-ordinates are normalized to fall between 0 and 1.
  
- Class Predictions: Along with bounding boxes, each grid cell predicts the probability distribution over predefined classes. The class probabilities are conditioned on the grid cell containing an object.
  
- Thresholding and Non-Max Suppression: Thresholding is applied to filter out low-confidence predictions. NMS is used to eliminate redundant boxes, keeping only one with the highest IOU score for each detected object.

### What's New in YOLOV8?

- Instance Segmentation: Pixel-level segmentation and bounding boxes to predict the exact shape of each object with class label. Allows it to be used for scientific image analysis.

- YOLOv8 is designed to be faster, more accurate, and more versatile, making it a powerful tool for a wide range of computer vision tasks, classification, detection, instance segmentation, tracking, and pose estimation.

- If your application prioritizes speed and can afford a trade-off in accuracy, YOLO might be the better choice. However, for scenarios where accuracy is paramount, I would recommend using Faster R-CNN or Mask R-CNN within the Detectron2 framework, which tend to provide more precise results at the cost of slower inference times.

- I'm currently developing a project that will compare the performance of YOLOV8 and Detectron2 on scientific images, evaluating their effectiveness in terms of both speed and accuracy across different tasks. Please feel free to checkout the progress on my GitHub: GITHUB
Anchor Free Detection: Anchor-free detection aims to simplify this process by eliminating the need for predefined anchors. Traditional object detection models, like YOLOv3 or SSD, rely on predefined anchor boxes to propose regions in the image where objects might be located.

- Feature Extraction: The backbone extracts feature maps from the input image, capturing important details like object boundaries, textures, and patterns.

- Key Point Prediction: The model predicts specific points on the object, such as the center or the corners. For example, in FCOS, the model predicts the center point of the object along with its distance to the edges of the bounding box.

- Bounding Box Regression: The model directly regresses the bounding box coordinates based on the predicted key points. This approach allows the model to adapt better to objects of varying shapes and sizes, without being constrained by predefined anchors.

### Let's see How Point-based Predictions make YOLOV8 model Anchor free ?

- FCOS (Fully Convolutional One-Stage Object Detection)

- Algorithm Overview:

- Anchor-Free Detection: FCOS is one of the pioneering anchor-free object detection algorithms. It eliminates the need for predefined anchor boxes by directly predicting the object's center and the distances from this center to the four sides of the bounding box.

- Point-Based Predictions: FCOS predicts four values for each pixel in the feature map:
The distance from the pixel (assumed to be the center of an object) to the top, bottom, left, and right boundaries of the bounding box.
A class probability score that indicates the likelihood of the object belonging to a specific class.

### How It Works:

- Feature Maps: The backbone network (e.g., CSPDarknet) extracts features from the image, creating a set of feature maps at different scales.

- Center Prediction: For each location in the feature map, FCOS predicts whether this point is the center of an object. Model predicts a center-ness score for each location. The center-ness score indicates the likelihood that the pixel is the center of an object.

- Bounding Box Regression: If the point is predicted as a center, FCOS also predicts the distances to the four sides of the bounding box.

- Classification: Finally, the model outputs the class probabilities associated with the detected object.
