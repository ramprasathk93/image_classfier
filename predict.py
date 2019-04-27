import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import torch 
from torch import nn
from torch import optim
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import argparse

def argumentParser():
    #setup parser
    parser = argparse.ArgumentParser(description='Predict a flower')
    parser.add_argument('image_dir', action="store")
    parser.add_argument('checkpoint', action="store")
    parser.add_argument('--top_k', action='store', dest='topk', type=int, default=3)
    parser.add_argument('--category_names', action='store', dest='cat_names', default="cat_to_name.json")
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=False)
    return parser.parse_args()

def process_image(image_path):
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = image_transform(Image.open(image_path))
    return image_tensor

def predictImage(image_path, model, topk, gpu):
    # TODO: Implement the code to predict the class from an image file
    image_torch = process_image(image_path)
    image_torch = image_torch.unsqueeze_(0)
    image_torch = image_torch.float()
    model.eval()
    if (gpu == True and torch.cuda.is_available()):
        model.cuda()
        output = model.forward(image_torch.cuda())
        print('Utilising GPU for predicting!')        
    else:
        model.cpu()
        output = model.forward(image_torch)
        print("Utilising CPU for predicting!")
    
    probability = F.softmax(output.data,dim=1)
    
    #storing the probablities and indices corresponding to classes
    probabilities = np.array(probability.topk(topk)[0][0])
    indices = np.array(probability.topk(topk)[1][0])
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()} #invert the dictionary
    top_classes = [np.int(index_to_class[index]) for index in indices]
    
    return probabilities, top_classes

def loadCheckpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(torchvision.models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def loadCategories(category_names_file):
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def main():
    arguments = argumentParser()
    model = loadCheckpoint(arguments.checkpoint)
    categories = loadCategories(arguments.cat_names)
    probabilities, top_classes = predictImage(arguments.image_dir, model, arguments.topk, arguments.gpu)
    flower_names = [categories[str(index)] for index in top_classes]
    top_category = flower_names[0]
    probability = str(round(probabilities[0] * 100, 2))
    print("The given flower could be a " + top_category + " with a probability of " + probability + "%.")
    
    print("\nTop k classes are: ")
    print("Category : Probability")
    k = 0 
    while (k < len(top_classes)):
        print(flower_names[k] + " : " + str(probabilities[k]))
        k += 1
    
if __name__ == "__main__":
    main()