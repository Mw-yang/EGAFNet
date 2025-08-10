import cv2
import torch
import numpy
import numpy as np
import cv2 as cv
import os
import PIL.Image as Image

def evaluate_indicators(tp: int, fp: int, tn: int, fn: int):
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2*precision*recall/(precision+recall)
    OA = (tp+tn)/(tp+tn+fp+fn)
    IoU = tp / (tp + fp + fn)

    return IoU, recall, precision, f1, OA

def evaluate_indicatorss(ttp: int, ffp: int, ttn: int, ffn: int):
    rrecall = ttp/(ttp+ffn)
    pprecision = ttp/(ttp+ffp)
    ff1 = 2*pprecision*rrecall/(pprecision+rrecall)
    OOA = (ttp+ttn)/(ttp+ttn+ffp+ffn)
    IIoU = ttp / (ttp + ffp + ffn)

    return IIoU, rrecall, pprecision, ff1, OOA

# num = len(os.listdir(r'测试标签路径'))，获取标签的数量。
num = len(os.listdir(r'E:\Dataset\WHU512\test\label1'))
# num = len(os.listdir(r'E:\Dataset\China500\test\label'))
tp = 0
tn = 0
fp = 0
fn = 0
step = 0
for i in range(1, num + 1):
    ttp = 0
    ttn = 0
    ffp = 0
    ffn = 0
    gt = cv2.imread(r'E:\Dataset\WHU512\test\label1\%i.tif' % i, -1)
    # gt = cv2.imread(r'E:\Dataset\China500\test\label\%i.png' % i, -1)
    pred = cv2.imread(r'E:\DLlesson\test_images\EGAFNet5e04\%i.png' % i, -1)  # 后缀根据数据情况修改
    gt = cv2.resize(gt, (256, 256), cv2.INTER_AREA)
    _, pred = cv2.threshold(pred, 120, 255, cv.THRESH_BINARY)  # 第二个参数为可设置阈值，像素值超过阈值则被赋值255
    cv2.imwrite(r'E:\DLlesson\threshold\EGAFNet5e04\%i.png' % i, pred)

    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt)
    output_mtx = torch.squeeze(pred).cpu().detach().numpy()
    label_mtx = torch.squeeze(gt).detach().numpy()

    pixel_count = output_mtx.shape[0]

    for i in range(pixel_count):
        for j in range(pixel_count):
            input_positive = output_mtx[i][j] > 0.5
            label_positive = label_mtx[i][j] > 0.5

            # if input and label are the same class
            if not (input_positive ^ label_positive):

                # if positive
                if input_positive:
                    tp += 1
                    ttp += 1
                else:
                    tn += 1
                    ttn += 1

            else:
                if input_positive:
                    fp += 1
                    ffp += 1
                else:
                    fn += 1
                    ffn += 1

    step += 1
IoU, recall, precision, f1, OA = evaluate_indicators(tp, fp, tn, fn)

print( "tp = {}, tn = {}, fp = {}, fn = {}".format(tp, tn, fp, fn))
print( "OA = {}, precision = {}, recall = {}, f1 = {}, IoU = {}".format(OA, precision, recall, f1, IoU))
print("准确率:")
print(OA)
print("精确率:")
print(precision)
print("召回率:")
print(recall)
print("F1分数:")
print(f1)
print("交并比:")
print(IoU)
print("Finished!!!")