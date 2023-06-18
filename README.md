# MFA
Official code for Motional Foreground Attention-based (MFA) Video Crowd Counting. 

## Train 
1. Download the dataset. 
2. Change the path to the path which includes 'train.py'.
3. Run 'pre_train.py' to conduct stage-1 training. 
```shell 
python pre_train.py (dataset path) (save path) amb (gpu id) (task name) -m (roi file path)
```
4. Run 'train.py' to conduct stage-2 training. 
```shell 
python train.py (dataset path) (save path) (stage-1 model path) PreTrain (gpu id) (task name) -m (roi file path)
```

## Test 
1. Download the dataset. 
2. Change the path to the path which includes 'test.py'.
3. Run 'test.py' to conduct testing. 
```shell
python test.py (dataset path) (model weight path) (gpu id) -m (roi file path)
```
