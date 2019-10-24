import os
import torch
import numpy as np

import time

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from sklearn.metrics import auc

from utils.evaluation import evaluate_sample


def get_detection_model(num_classes=2):
    """
    Get Faster R-CNN model pretrained on COCO dataset
    :param num_classes: The number of classes that the model should detect
    :return: Torch model
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    count = 0
    for child in model.backbone.children():
        if count == 0:
            for param in child.parameters():
                param.requires_grad = False
        count += 1

    for param in model.roi_heads.box_head.fc6.parameters():
        param.requires_grad = False

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    count = 0
    global_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device).float() for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        dict_loss = model(images, targets)
        losses = sum(loss for loss in dict_loss.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        count += 1
        global_loss += float(losses.cpu().detach().numpy())

        if count % 10 == 0:
            print("Loss value after {} batches is {}".format(count, round(global_loss / count, 2)))

    return global_loss


def train(model, num_epochs,train_loader, test_loader, optimizer, device, save_path):
    """
    Train the model
    :param model:
    :param num_epochs:
    :param train_loader:
    :param test_loader:
    :param optimizer:
    :param device:
    :param save_path
    :return:
    """
    for epoch in range(num_epochs):
        print("epoch {}/{}..".format(epoch, num_epochs))
        start = time.time()
        train_one_epoch(model, optimizer, train_loader, device)
        mAP = evaluate(model, test_loader, device=device)
        end = time.time()

        print("epoch {} done in {}s".format(epoch, round(end - start, 2)))
        print("mAP after epoch {} is {}:".format(epoch, round(mAP, 3)))

        if (epoch + 1) % 5 == 0:
            print("Model saved to {}".format(os.path.join(save_path,
                                                          str(epoch) + ".pth")))
            torch.save(model.state_dict(),
                       os.path.join(save_path, str(epoch) + ".pth"))

        print('#' * 25)

    torch.save(model.state_dict(),
               os.path.join(save_path, str(num_epochs) + ".pth"))


def evaluate(model, test_loader, device):
    """
    Computes precision and recall for a given trehsold (default = 0.5)
    :param model :
    :param test_loader:
    :param device:
    :return : tuple containing precision and recall
    """
    results = []
    model.eval()
    nbr_boxes = 0
    with torch.no_grad():
        for batch, (images, targets_true) in enumerate(test_loader):
            images = list(image.to(device).float() for image in images)
            targets_pred = model(images)

            targets_true = [{k: v.cpu().float() for k, v in t.items()} for t in targets_true]
            targets_pred = [{k: v.cpu().float() for k, v in t.items()} for t in targets_pred]

            for ii in range(len(targets_true)):
                target_true = targets_true[ii]
                target_pred = targets_pred[ii]
                nbr_boxes += target_true['labels'].shape[0]

                results = results + evaluate_sample(target_pred, target_true)

    results = sorted(results, key=lambda k: k['score'], reverse=True)

    acc_TP = np.zeros(len(results))
    acc_FP = np.zeros(len(results))
    recall = np.zeros(len(results))
    precision = np.zeros(len(results))

    if results[0]['TP'] == 1:
        acc_TP[0] = 1
    else:
        acc_FP[0] = 1

    for ii in range(1, len(results)):
        acc_TP[ii] = results[ii]['TP'] + acc_TP[ii - 1]
        acc_FP[ii] = (1 - results[ii]['TP']) + acc_FP[ii - 1]

        precision[ii] = acc_TP[ii] / (acc_TP[ii] + acc_FP[ii])
        recall[ii] = acc_TP[ii] / nbr_boxes

    return auc(recall, precision)
