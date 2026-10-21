# YOLO11 Object Detection Project Issue 034 - Practical System for Fine-Grained Detection of Goats in Natural Environment Based on YOLO11 (Dataset + Source Code Model + UI Interface)

## 1. Data phase

### 1.1 Data Set Preparation

The dataset of goat fine-grained detection in natural environment is a dataset for target detection task. The main application fields of this dataset are **livestock/agriculture/farming**. The example is shown as follows:

![](01-561e2430.png)

Background: Agriculture, as a key industry for ensuring global food security and economic stability, relies on the strong adaptability of goats to their environment, which results in high milk and meat production and quality. However, traditional goat behavior analysis relies on manual observation (time-consuming and prone to errors) or GPS tracking (contact with animals and insufficient accuracy at ranch scale), making it difficult to meet the needs for precise monitoring. Automated computer vision technology provides feasible solutions for goat detection and tracking, but existing livestock detection datasets have significant deficiencies: mainstream general datasets (such as COCO, PASCAL VOC) lack the "goat" category, specialized datasets (such as Roboflow-related datasets) have low annotation quality (inaccurate bounding boxes, missed or incorrect labels), and lack high-quality data covering diverse natural environments. To address this, the current dataset of fine-grained goat detection in natural environments fills the gap in specialized high-quality goat detection datasets, providing benchmark support for algorithm development in precision agriculture, animal welfare monitoring, and behavior analysis.

Dataset Details: This dataset consists of **real indoor and outdoor scenes**. The images in the dataset are built on the basis of **outdoor pastures and indoor fences**, captured under various lighting and angles, and labeled according to Roboflow tools and YOLO format. There are approximately **6100 high-quality JPG images** in the dataset, accompanied by corresponding txt label files, each corresponding to one image. The labeling format is uniformly YOLO-style, divided into **training set**, **validation set, test set** at an approximate ratio of **8:1:1** for easy training and comparative testing of YOLO series algorithms. The distribution of data categories mainly covers a common **1 type** of livestock, such as goats, etc.

1.2 Data Set Structure Organization

The YOLO format dataset structure is as follows:

**yolo_datasets/******

**- train******

**-- images******

**-- labels******

**- valid******

**-- images******

**-- labels******

**- test******

**-- images******

**-- labels******

The training set contains approximately **4900** high-quality jpg images (including some png images), the validation set includes about **610** high-quality jpg images (including some png images), and the test set consists of approximately **610** high-quality jpg images (including some png images).

### 1.3 Data Set Download

[Note: The **yolo_datasets** folder should be prepared and placed in the same directory level as the following project, yolov11_det. Do not place the **yolo_datasets** folder inside the project.]

☆ You can directly obtain the converted Yolo format dataset (as mentioned later) by **seeing the end of the document**.

### 1.4 Environmental Installation

Execute the following instructions:

pip install ultralytics==8.3.59 -i https://pypi.tuna.tsinghua.edu.cn/simple

## 2. Training phase

### 2.1 Configuration File Modifications

[Prerequisites: CUDA, Torch, and Ultralytics versions must match and be available, using the Yolo11m pre-trained model]

[Need to enter the project "yolov11_det"]

Modify the data configuration file ./custom.yaml, set the training and validation sets paths, ensure there is a consistent number of categories and category names.

![](02-5121fd8d.png)

### 2.2 Execute Training Commands

Modify the training parameters in the train.py file as needed, such as: for Windows systems, set workers to 0; for Linux systems, set it to 8; other parameters should be modified according to hardware configuration requirements.

Execute training instructions:

python train.py

![](03-d15fc965.png)

After training 120 epochs (with early stopping mechanism, 120 epochs), the model is saved in the runs/detect/train/weights folder.

### 2.3 Training the Generated Results File

1) Core model

**weights/******

Store the trained model weights, which are the core output of training.

best.pt: Optimal model weights for validation set performance (e.g., mAP, loss) recommended for actual deployment.

last.pt: The final model weights at the end of training (potentially overfitted, not recommended for direct use).

2) Training configuration

**args.yaml******

Record all hyperparameters and configurations during training (such as model type, epochs, learning rate, dataset path, image size, etc.) for reproducing the training process or viewing parameter settings.

3) Logging

**events.out.tfevents.*******

TensorBoard log file, recording loss, mAP, learning rate and other indicators during training/validation. Visualize the training curve through tensorboard --logdir run/detect/train.

**results.csv******

Record the key metrics of each training round in CSV format (such as epoch, train/box_loss, val/mAP50, etc.), which can be used for Excel or script analysis of training trends.

### 2.4 Evaluating and Visualizing Results

1) F1_curve.png (F1 score)

![](04-a95e9bfe.png)

From the figure, it is evident that when a fixed confidence value is set, the F1 score of all_classes achieves its **maximum** value. This indicates that the model performs well in predicting classes with this confidence level, demonstrating both accuracy and recall.

What is the F1 score? The F1 score is the harmonic mean of precision and recall, taking into account both the accuracy of model predictions and their coverage.

2) P_curve.png (Accuracy Curve)

![](05-8091f6d7.png)

The deep blue line in the figure represents the overall accuracy, which increases with increasing confidence. At a fixed confidence level, the accuracy reaches 1.00. The other lines represent the accuracy at different confidence levels for various categories.

Precision curve, which helps analyze the model's performance in detection tasks, especially under different confidence settings. By this curve, a suitable confidence threshold can be selected to optimize the model's detection results, balancing precision and recall.

3) PR_curve.png (Precision-Recall curve)

![](06-71c0bdef.png)

The curve in the image represents the Precision-Recall Curve (Precision-Recall Curve), which shows the precision and recall of the classifier at different thresholds. Specifically, the horizontal axis represents Recall (recall rate), ranging from 0.0 to 1.0; the vertical axis represents Precision (precision rate), ranging from 0.0 to 1.0. The main curve in the image:

(1) The color lines represent the precision-recall curves of various categories, with varying degrees of precision for each category.

(2) The deep blue line represents the average precision-recall curve for all categories, with precision also maintaining a stable value around the maximum.

All these curves show a slight decrease in precision with an increase in recall rate, but overall the precision remains at a very high level.

By this curve, the overall performance of the model in the target detection task can be evaluated, and an appropriate confidence threshold can be found to strike a balance between precision and recall. The closer the PR curve is to the upper right corner, the better the model performs, and the larger the area under the curve, indicating that the model performs more stably across different confidence levels.

4) R_curve.png (Recall curve)

![](07-303709b0.png)

The R curve in the diagram represents the Recall-Confidence Curve (Recall-Confidence Curve), which shows the recall rate at different confidence levels. Specifically, the horizontal axis represents confidence levels, ranging from 0.0 to 1.0; the vertical axis represents recall rates, ranging from 0.0 to 1.0. The main curve in the diagram:

(1) Each color line represents the change in recall rate for each category as a function of confidence. This curve shows that, as confidence increases, the recall rate for each category initially changes less, then rapidly decreases after a certain point.

(2) Deep blue line: Represents the average recall rate of all categories as a function of confidence.

Display the recall performance of the model under different confidence thresholds. The recall curve aids in evaluating the model's ability to detect true targets in a target detection task. By examining this curve, users can assess the model's detection capabilities and analyze whether the model is comprehensively detecting true targets at varying thresholds. If the recall curve performs well at lower confidence levels, it indicates that the model can effectively detect most targets; however, if the curve declines rapidly, adjustments to the model or dataset may be necessary to avoid missed detection issues.

5) results.png (results indicator curve)

![](08-a0c6711c.png)

This image file shows the trend of key metrics (such as loss, mAP) during training.

box_loss: Position loss curve

Meaning: Bounding box loss, measuring the error between the predicted bounding box and the true one. Includes center position, width, height, etc.

Chart: Typically displays the loss value curve on training rounds or steps, with lower values indicating more accurate predictions of bounding boxes by the model.

cls_loss: Classification Loss Curve

Meaning: Category loss, measuring the gap between predicted class labels and true class labels. Reflects the accuracy of the classifier.

Chart: Shows the loss value curve for training epochs or steps, with lower values indicating better classification performance.

dfl_loss: Target Detection Loss Curve

Meaning: Distribution Focal Loss, which addresses the issue of class imbalance in positioning tasks and helps the model focus more on the difficult-to-predict targets.

Chart: Shows the loss value curve for training epochs or steps, with lower values indicating better performance in distributed positioning tasks.

Precision(B): Precision curve

Meaning: Precision (B-class), indicating how many proportion of true positives are predicted by the model. B-class usually refers to background class.

Chart: Shows the precision curve on training rounds or steps, with higher precision indicating higher accuracy in background class detection.

Recall(B): recall curve

Meaning: Recall rate (B-class), which represents the proportion of actual positive samples detected by the model. B-class usually refers to background classes.

Chart: Shows recall curves on training epochs or steps, with higher recall indicating stronger detection capabilities of the model for background classes.

mAP50(B): The average precision when the IoU threshold is 0.5

Mean accuracy (B class) at IoU threshold of 0.5, reflects the model's accuracy at this IoU threshold.

Chart: Shows the mAP50 value curve on training epochs or steps, with higher mAP50 values indicating higher detection accuracy of background classes by the model.

mAP50-95(B): The average precision between 0.5 and 0.95 for IoU thresholds

Mean accuracy (B-class) within the range of IoU thresholds from 0.5 to 0.95, evaluating the detection performance of models under different IoU thresholds.

Chart: Shows the mAP50-95 value curve on training epochs or steps, with higher mAP50-95 values indicating stronger overall performance of the model on background classes.

## 3. Inference phase

### 3.1 Starting the Inference

Predict using the trained model (input test image folder input_test_images):

Execute the following instructions:

python predict.py

![](09-544a357f.png)

After the test is completed, the results are saved in the output_results folder.

### 3.2 Output Display

Output:

![](10-b59ce0a2.png)

## 4. System UI interface

### 4.1 Implementing PyQt5 Code

Implement the interface code `ui_window.py` and the logic code `ui_main_gui.py`.

1) Environment setup

Execute the following instructions:

pip install ultralytics==8.3.59 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install PyQt5==5.15.9 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install PyQt5-Qt5==5.15.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

2) Set the model path

In the file ui_main_gui.py, specify the path of the optimal model after training.

![](11-2dbe3554.png)

### 4.2 Final System Interface

Place UI test images/videos in the folder ui_test_files.

Execute the following instructions:

python ui_main_gui.py

Display as follows:

![](12-57b0052b.png)

After the detection is completed (automatically), save the results in the ui_result directory (which will be created).

**Interface Features**:

1) Supports input sources such as images, videos, and cameras.

2) Confidence and Intersection over Union (IoU) parameters can be adjusted as needed.

3) Detection status (start/pause/end) etc. Can be set

4) Comparison of original image and resultant image

5) Support real-time statistics display of table data.

6) Support for automatic data saving of results


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
