## Setup
### Prerequisites
- keras==2.4.3
- opencv-python==4.3.0
- numpy
- scipy
- matplotlib


## Prepare data
A example of dataset folder and json file as followed:

```
data
├── image
|   |--000001.jpg
|   |--000002.jpg
├── mask
|   |--000001.jpg
|   |--000002.jpg

```
```
txt file contents

./data/image/000001.jpg ./data/mask/000001.jpg
./data/image/000002.jpg ./data/mask/000002.jpg
```

## Training
Training a classification model with default arguments. Model file and tensorboard log file art written out to 
directory ```backup``` respectively after starting traing.
```
python train_seg_fpn_pretrain.py
```


## Validation and Testing
You can validation and testing on the best model by:
```
python predict_seg.py
```

