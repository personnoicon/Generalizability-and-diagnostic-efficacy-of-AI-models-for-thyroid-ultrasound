## Setup
### Prerequisites
- tqdm
- pillow
- scikit-learn
- matplotlib
- torch==1.7.0
- torchvision==0.8.1
- torchtoolbox
- warmup_scheduler

## Prepare data
A example of dataset folder and json file as followed:

```
datasets
├── 000001
|   |--000001_BModeH.jpg
|   |--000001_BModeZ.jpg
├── 000002
|   |--000002_BModeH.jpg
|   |--000002_BModeZ.jpg
├── *****
|   |--********.jpg
|   |--********.jpg
```
```
json file contents

{"samples": [{"image_name": "datasets/000001/000001_BModeH.jpg", "image_labels": ["malignant"]}, 
{"image_name": "datasets/000001/000001_BModeZ.jpg", "image_labels": ["malignant"]}, 
{"image_name": "datasets/000002/000002_BModeH.jpg", "image_labels": ["benign"]},
{"image_name": "datasets/000002/000002_BModeZ.jpg", "image_labels": ["benign"]},
.............], "labels": ["malignant"]}
```

## Training
Training a classification model with default arguments. Model file and tensorboard log file art written out to 
directory ```backup``` respectively after starting traing.
```
python train.py
```

## Inference
You can run inference with image on the best model by:
```
python infer.py
```

## Validation and Testing
You can validation and testing on the best model by:
```
python eval.py
```

