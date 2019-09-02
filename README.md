# mtl-text-recognition
multi-task learning for text recognition with joint ctc-attention.
## Update features
+ support **variable** length of squence training and inference (fixed height).
+ **Chinese** character recognition.
+ **Joint CTC-Attention**<br><br>
<img src="./figures/mtl_arch.png" width="480" title="CTC-Attention model architecture"> <br><br>
## Getting Started
### Dependency
- This work was tested with PyTorch 1.1.0, CUDA 9.0, python 3.6 and centos7 
- requirements : pytorch, lmdb, pillow, torchvision, nltk, natsort
```
pip3 install torch==1.1.0
pip3 install lmdb pillow torchvision nltk natsort
```
### Run demo with pretrained model(中文+英文字符版本，使用config中的chn.txt文件)
1. Download pretrained model(crnn) from [baidu](https://pan.baidu.com/s/1k6__hYRdq8BnyihP8ImLWw)code:iiw7<br>
pretrained crnn model configuration:
```
--output_channel 512 \
--hidden_size 256 \
--Transformation None \
--FeatureExtraction ResNet \
--SequenceModeling BiLSTM \
--Prediction CTC \
--experiment_name none_resnet_bilstm_ctc
--continue_model saved_models/pretrained_model.pth
```
run demo with pretrained mdoel.
```
CUDA_VISIBLE_DEVICES=0 python3 infer.py ${image_path}
```
2. CTC-Attention pretrained model will release later.

### Training and evaluation
1. Train CRNN model
```
CUDA_VISIBLE_DEVICES=0 python train.py \
	--train_data data/synch/lmdb_train \
	--valid_data data/synch/lmdb_val \
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
	--train_data data/synch/lmdb_train \
	--valid_data data/synch/lmdb_val \
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
