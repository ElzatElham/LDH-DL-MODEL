# LDH-DL-MODEL
This is the official PyTorch implementation of our paper: 

**Quantification and Classification of Lumbar Disc Herniation on Axial Magnetic Resonance Images Using Deep Learning Models**
![image](https://github.com/ElzatElham/LDH-DL-MODEL/blob/main/image.png)

# Installation

Pip install the [YOLOv8](https://github.com/ultralytics/ultralytics) package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.
```
pip install ultralytics
```
# Pretrained model

The pretrained model urls of Object Detection([YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)), Semantic Segmentation([YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt)), Keypoint Detection([YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose-p6.pt))]

# Dataset
The structure of the yaml files for the three datasets as follows:

[Object Detection](https://github.com/ElzatElham/LDH-DL-MODEL/blob/main/Object%20_Detection_train.yaml)

[Semantic Segmentation](https://github.com/ElzatElham/LDH-DL-MODEL/blob/main/Semantic_Segmentation_train.yaml)

[Keypoint Detection](https://github.com/ElzatElham/LDH-DL-MODEL/blob/main/Keypoint_Detection_train.yaml)

# Run training and testing

Training model:
```
python YOLOV8_training.py --weights yolov8.pt  --data /home/data.yaml  --epochs 200 --imgsz 512 --batch 16 --device 0 --patience 50 --name train_result
```

Testing model:
```
python YOLOV8_testing.py --weights --weights yolov8.pt  --data /home/data.yaml --device 0
```

# Citation
```
@article{jocher2023yolo,
  title={YOLO by Ultralytics},
  author={Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year={2023},
  publisher={Jan}
}
```






