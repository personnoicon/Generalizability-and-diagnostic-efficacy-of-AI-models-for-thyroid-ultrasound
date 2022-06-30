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
├── 0a2c277925509608ca997d9f2880214dce4f3e81
|   |--0a2c277925509608ca997d9f2880214dce4f3e81_A001_BModeH.jpg
|   |--0a2c277925509608ca997d9f2880214dce4f3e81_A001_BModeZ.jpg
├── 0a38ecbd4b5adf2129efff1d8b37b0813dca6fad
|   |--0a38ecbd4b5adf2129efff1d8b37b0813dca6fad_A001_BModeH.jpg
|   |--0a38ecbd4b5adf2129efff1d8b37b0813dca6fad_A001_BModeZ.jpg
├── *****
|   |--********.jpg
|   |--********.jpg
```
```
json file contents

{"samples": [{"image_name": "datasets/acc48d6cfff1c72e782227e9d01b2838e14e2eba/000002.jpg", "image_labels": ["malignant"]}, 
{"image_name": "datasets/acc48d6cfff1c72e782227e9d01b2838e14e2eba/000001.jpg", "image_labels": ["malignant"]}, 
{"image_name": "datasets/bcfb2b2d043c861675ee0ea9265a85a23dfbb9fb/FU_LIANG_MING_20190910094900_0951370.jpg", "image_labels": ["benign"]},
{"image_name": "datasets/bcfb2b2d043c861675ee0ea9265a85a23dfbb9fb/FU_LIANG_MING_20190910094900_0951280.jpg", "image_labels": ["benign"]},
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

