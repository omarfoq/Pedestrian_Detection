import os
import time

import torch
from torch.utils.data import DataLoader

from engine import get_detection_model, train_one_epoch, evaluate, train
from utils.dataset import PennFudanDataset
from utils.transforms import Compose, ToTensor
from utils.utils import collate_fn, show_sample

root_path = "../data"
save_path = "../models"
show_example = False
num_epochs = 10
batch_size = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == "__main__":
    # Create Dataloader
    dataset = PennFudanDataset(root_path, transforms=Compose([ToTensor()]))

    # split the dataset in train and test set
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    if show_example:
        img, target = dataset[10]
        show_sample(img, target)

    # Create the model
    model = get_detection_model(num_classes=2)
    model.to(device)

    # Construct optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    train(model, num_epochs, train_loader, test_loader, optimizer, device, save_path)




