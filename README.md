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
1. Download pretrained model(crnn) from [baidu](https://pan.baidu.com/s/1tsqdunmZQV17ckqllP4YJw) code:un8d<br>
pretrained crnn model configuration:
```
--output_channel 512 \
--hidden_size 256 \
--Transformation None \
--FeatureExtraction ResNet \
--SequenceModeling BiLSTM \
--Prediction CTC \
--continue_model saved_models/best_accuracy.pth
```
run demo with pretrained mdoel.
```
# change config.py file and run:
CUDA_VISIBLE_DEVICES=0 python3 infer.py ${image_path}
```
2. CTC-Attention pretrained model will release later.<br><br>
## prediction results
| demo images | None-ResNet512-BiLSTM256-CTC| None-ResNet768-BiLSTM384-CTC |
|     ---      |     ---      |     ---      |
<img src="./demo_images/demo_0.jpg" width="300">    |   同达电动车配件   |     |
| <img src="./demo_images/demo_1.jpg" width="300">    |   微信14987227   |     |
| <img src="./demo_images/demo_2.png" width="300">      |   快乐大本营20190629期:张艺兴李荣浩惊喜同台合唱彭昱畅破音三连引     |       |
| <img src="./demo_images/demo_3.png" width="300">  |   每周三中午12:00   |     |
| <img src="./demo_images/demo_4.png" width="300">      |   整套征兵甄别程序的一个部分     |       |
| <img src="./demo_images/demo_5.png" width="300">    |   再热烈的鼓掌   |     |
| <img src="./demo_images/demo_6.png" width="300">      |   厂外恒升拆车件     |       |
| <img src="./demo_images/demo_7.png" width="300">    |   我想说你为什么   |     |
| <img src="./demo_images/demo_8.jpg" width="300">      |    我抱吧他在你怀里一直在哭    |       |
| <img src="./demo_images/demo_9.jpg" width="300">    |   如果没有这个阿姨的话   |     |
| <img src="./demo_images/demo_10.jpg" width="300">      |    因为我觉得有你了我才有安全感    |       |

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
  	--experiment_name none_resnet_bilstm_ctc \
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
  	--experiment_name none_resnet_bilstm_ctc \
  	--continue_model saved_models/pretrained_model.pth
```

## Acknowledgements
1. This implementation has mainly been based on this great repository: [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
2. SynthText Generation has mainly been based on [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator/tree/master/TextRecognitionDataGenerator)
