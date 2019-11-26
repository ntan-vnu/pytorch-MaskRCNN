import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from PIL import Image
import cv2
import numpy as np

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

from PennFudanDataset import PennFudanDataset

from engine import train_one_epoch, evaluate
import utils


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load('markrcnn_state_dict.pth'))

    # move model to the right device
    model.to(device)

    img, target = next(iter(data_loader_test))
    img = [it.to(device) for it in img]

    model.eval()
    predict = model(img)

    img = img[0].to('cpu').numpy()
    img = img.transpose(1, 2, 0) * 255
    cv2.imwrite('output/img.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    colors = [list(np.random.choice(range(256), size=3)) for i in range(len(predict[0]['masks']))]
    mask = np.zeros(img.shape)
    for i, mm in enumerate(predict[0]['masks']): 
        mm = mm.to('cpu').detach().numpy()[0]
        mm = (mm > 0.5).astype(int)
        mm = np.stack([mm]*3, axis=2) * colors[i]
        mask += mm
    cv2.imwrite('output/mask.jpg', mask)
    

if __name__ == "__main__":
    main()
    pass
