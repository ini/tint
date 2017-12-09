import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
from PIL import Image

# Adapted from https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py
def get_scene_info(path, useGPU=True):
    # the architecture to use
    arch = 'resnet50'

    # load the pre-trained weights
    model_file = 'whole_%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/whole_%s_places365.pth.tar' % arch
        os.system('wget ' + weight_url)

    if useGPU:
        model = torch.load(model_file)
    else:
        model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!

    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
            trn.Scale(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the test image
    img = Image.open(path).convert('RGB')
    input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit).data.squeeze()
    return h_x.numpy()
