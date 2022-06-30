# Thyroid Nodule Detection Model

## Table of contents
- [Prerequisites](#Prerequisites)
- [PrepareData](#PrepareData)
- [Train](#Train)
- [Evaluation](#Evaluation)
- [Inference](#Inference)


## Prerequisites

The thyroid nodule detection model of our work base on the [yolov5](https://github.com/ultralytics/yolov5/tree/v3.1) of branch v3.1, please refer to the requirements and doc of yolov5 on branch v3.1.

## PrepareData
You must preprocess your detection dataset annotations into yolo's txt format. And random split your dataset into train/val. A example of data are also provided.
```
data
├── images
|   ├── train
|   |   ├── xxx1.jpg
|   |   └── xxx2.jpg
|   └── val
|       ├── xxx3.jpg
|       └── xxx4.jpg
└── labels
    ├── train
    |   ├── xxx1.txt
    |   └── xxx2.txt
    └── val
        ├── xxx3.txt
        └── xxx4.txt
```

## Train
Since we use the official yolov5 training script, here we only give the commands used for training. Run command as below.
```
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 416 416 --data thyroid.yaml --cfg yolov5s.yaml --hyp data/hyp.thyroid.yaml --weights weights/yolov5s.pt --sync-bn --device 4,5,6,7 --name test --multi-scale --cache-images
```

## Evaluation
- Modify xxx.pt with your own model path
- Run command as below
```
python test.py --img 416 --conf 0.001 --iou-thres 0.5 --batch 64 --device 4 --data thyroid.yaml --weights xxx.pt --task test --verbose --classes 0
```

## Inference
- Fill in the source and save-dir param with your own path and modify xxx.pt with your own model path
- Run command as below
```
python detect.py --img 416 --source xxx --device 4 --weights xxx.pt --save-dir xxx --iou-thres 0.5 --conf-thres 0.3 --save-img --classes 0
```
