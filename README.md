# mtl-text-recognition
multi-task learning for text recognition with joint ctc-attention.
## Update features
+ support **variable** length of squence training and inference (fixed height).
+ **Chinese** character recognition.
+ **Joint CTC-Attention**
## Getting Started
### Dependency
- This work was tested with PyTorch 1.1.0, CUDA 9.0, python 3.6 and Ubuntu 16.04. <br> You may need `pip3 install torch==1.1.0`
- requirements : lmdb, pillow, torchvision, nltk, natsort
```
pip3 install lmdb pillow torchvision nltk natsort
```
### Run demo with pretrained model(Chinese version)
1. Download pretrained model from [baidu](http)
```
CUDA_VISIBLE_DEVICES=0 python3 infer.py ${image_path}
```
### Training and evaluation
1. Train CRNN model
```
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_data data/synch/lmdb_train --valid_data data/synch/lmdb_val \
	--select_data / --batch_ratio 1 \
  --sensitive \
  --num_iter 400000 \
  --output_channel 512 \
  --hidden_size 256 \
	--Transformation None \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction CTC \
  --experiment_name none_resnet_bilstm_ctc
  --continue_model saved_models/pretrained_model.pth
```
2. Train CTC-Attention model
```
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_data data/synch/lmdb_train --valid_data data/synch/lmdb_val \
	--select_data / --batch_ratio 1 \
  --sensitive \
  --num_iter 400000 \
  --output_channel 512 \
  --hidden_size 256 \
	--Transformation None \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction CTC \
  --mtl \
  --without_prediction \
  --experiment_name none_resnet_bilstm_ctc
  --continue_model saved_models/pretrained_model.pth
```
### prediction results

## Acknowledgements
1. This implementation has mainly been based on this great repository: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
2. SynthText Generation has mainly been based on [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator/tree/master/TextRecognitionDataGenerator)
