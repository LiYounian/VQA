# VQA
Younian Li

## Dependencies

- Python 3.6
  - torch > 0.4
  - torchvision 0.2
  - h5py 2.7
  - tqdm 4.19

## Dataset
- This URL provides data that was preprocessed using Faster R-CNN. The dimension of the data is 36.
  - This dataset comes from [bottom-up top-down][0]
```
https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip
```
Run this file to adjust the picture information so it can be used for training.
```
python preprocess-images.py
```
- This creates two `h5py` database (23 GB and 35 GB).
- Other data can be found on the official website of VQA.
```
https://visualqa.org/download.html
```

## Train
Run the 'train.py' file to train the model. You can change the model you want to train in 'config.py'. 
The default is final model, in the 'final_model.py' file.

```
python train.py
```
- The training time is very long, you can view the already trained process in ‘slurm-5719.out’.

## Evaluate Accuracy
- To evaluate accuracy in various categories, you can run
```
python train.py --test --resume=./logs/YOUR_MODEL.pth
``` 
- The results of the training are presented in the paper and PPT.

[0]: https://github.com/peteanderson80/bottom-up-attention
