# coding=utf-8

import sys
import logging
logging.basicConfig(
    format='[%(asctime)s] [%(filename)s]:[line:%(lineno)d] [%(levelname)s] %(message)s', level=logging.INFO)


"""
将中文标注文件*.txt中的标签index转化为字符
"""


def index2char(char_file_path, data_file_path, out_file_path):
    """
    :param char_file_path: 字典列表文件
    :param data_file_path: 标注数据文件: filename\tlabel list split with space
    :param out_file_path: 转化后
    :return:
    """
    char_list = ['']
    with open(char_file_path) as charf:
        for line in charf:
            line = line.strip("\n")
            char_list.append(line)
    print(len(char_list))
    outfile = open(out_file_path, 'w')
    with open(data_file_path) as infile:
        for line in infile:
            line = line.strip()
            sgs = line.split()
            image_path = sgs[0]
            labels = map(int, sgs[1:])
            chars = []
            for idx in labels:
                chars.append(char_list[idx])
            char_line = "".join(chars)
            print(f"{image_path}\t{char_line}", file=outfile)
    outfile.close()


if __name__ == "__main__":
    char_fname = "../config/char_std_5990.txt"
    data_fname = "/data/grayenv/dataset/ocr/SynChDataset/data_train.txt"
    out_fname = "../data/train_utf8.txt"
    index2char(char_fname, data_fname, out_fname)

