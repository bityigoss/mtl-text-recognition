# coding=utf-8

import os
import torch


class ConfigOpt:
    def __init__(self):
        self.cur_path = os.path.abspath(os.path.dirname(__file__))
        self.workers = 4
        self.batch_size = 1
        self.saved_model = os.path.join(self.cur_path, "models/best_accuracy_768_98335.pth")
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 280
        self.rgb = False
        self.sensitive = True
        self.PAD = False
        self.Transformation = 'None'  # None|TPS
        self.FeatureExtraction = 'ResNet'  # VGG|RCNN|ResNet
        self.SequenceModeling = 'BiLSTM'  # None|BiLSTM
        self.Prediction = 'CTC'  # CTC|Attn
        self.num_fiducial = 20
        self.input_channel = 1
        # self.output_channel = 512
        # self.hidden_size = 256
        self.output_channel = 768
        self.hidden_size = 384
        self.num_gpu = torch.cuda.device_count()
        self.char_dict = "config/chn_dict.txt"
        self.character = self.get_character()
        self.mtl = False
        self.ctc_num_class = 0
        self.num_class = 0

    def get_character(self):
        ch_chars = ""
        ch_path = os.path.join(self.cur_path, self.char_dict)
        with open(ch_path) as charf:
            for line in charf:
                line = line.strip()
                ch_chars += line.encode("utf-8", 'strict').decode("utf-8", 'strict')
        return ch_chars




