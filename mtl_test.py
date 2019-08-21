# coding=utf-8
import os
import time
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from nltk.metrics.distance import edit_distance

from utils.utils import CTCLabelConverter, AttnLabelConverter, Averager
from utils.dataset import hierarchical_dataset, AlignCollate
from mtl_model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validation(model, ctc_criterion, attn_criterion, evaluation_loader, ctc_converter, attn_converter, opt):
    """ validation or evaluation """
    for p in model.parameters():
        p.requires_grad = False
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    ctc_correct = 0
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        #image = image_tensors.cuda()
        image = image_tensors.to(device)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        ctc_text_for_loss, ctc_length_for_loss = ctc_converter.encode(labels)
        attn_text_for_loss, attn_length_for_loss = attn_converter.encode(labels)

        start_time = time.time()
        ctc_preds, attn_preds = model(image, text_for_pred)
        forward_time = time.time() - start_time
        # ctc
        ctc_preds = ctc_preds.log_softmax(2)
        # Calculate evaluation loss for CTC deocder.
        preds_size = torch.IntTensor([ctc_preds.size(1)] * batch_size)
        ctc_preds = ctc_preds.permute(1, 0, 2)  # to use CTCloss format
        ctc_cost = ctc_criterion(ctc_preds, ctc_text_for_loss, preds_size, ctc_length_for_loss)
        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = ctc_preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        ctc_preds_str = ctc_converter.decode(preds_index.data, preds_size.data)

        # attention
        attn_preds = attn_preds[:, :attn_text_for_loss.shape[1] - 1, :]
        target = attn_text_for_loss[:, 1:]  # without [GO] Symbol
        attn_cost = attn_criterion(attn_preds.contiguous().view(-1, attn_preds.shape[-1]), target.contiguous().view(-1))
        # select max probabilty (greedy decoding) then decode index to character
        _, attn_preds_index = attn_preds.max(2)
        attn_preds_str = attn_converter.decode(attn_preds_index, length_for_pred)
        attn_labels = attn_converter.decode(attn_text_for_loss[:, 1:], attn_length_for_loss)

        cost = opt.ctc_weight * ctc_cost + (1.0 - opt.ctc_weight) * attn_cost
        infer_time += forward_time
        valid_loss_avg.add(cost)
        # calculate accuracy.
        #for attn_pred, attn_gt in zip(attn_preds_str, attn_labels):
        for pred, gt, attn_pred, attn_gt in zip(ctc_preds_str, labels, attn_preds_str, attn_labels):
            attn_pred = attn_pred[:attn_pred.find('[s]')]  # prune after "end of sentence" token ([s])
            attn_gt = attn_gt[:attn_gt.find('[s]')]

            if pred == gt:
                ctc_correct += 1
            if attn_pred == attn_gt:
                n_correct += 1
            norm_ED += edit_distance(attn_pred, attn_gt) / len(attn_gt)

    accuracy = n_correct / float(length_of_data) * 100
    ctc_accuracy = ctc_correct / float(length_of_data) * 100

    return valid_loss_avg.val(), accuracy, ctc_accuracy, norm_ED, attn_preds_str, attn_labels, infer_time, length_of_data



