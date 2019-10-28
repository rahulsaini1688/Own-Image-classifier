from get_input_args import get_input_args
import torch
from torchvision import datasets,transforms,models
import torchvision.models as models
import numpy as np
from torch import nn,optim
import helper
from collections import OrderedDict
from PIL import Image
import json

import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

args = get_input_args()

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 512)),
                          ('relu', nn.ReLU()),
                          ('do1', nn.Dropout(p=0.5)), 
                          ('fc2', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        classifier.load_state_dict(checkpoint['state_dict'])
    model.classifier = classifier
    model.class_to_idx = checkpoint['model.class_to_idx']
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['state_optimizer'])
    return (model, optimizer, criterion)
(model , optimizer, criterion) = load_checkpoint('checkpoint.pth')

print(model)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
   
   
    width, height = image.size 
    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    if width < height:
        
        new_height = int(height * 256 / width)
        image = image.resize((256, new_height))
    else:
        new_width = int(width *256 / height)
        image = image.resize((new_width, 256))
    width, height = image.size 
    # preparation for the 4 value tuple to be used for image.crop function
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width - left)
    bottom = (height - top)

    # Crop the center of the image
    im = image.crop((left, top, right, bottom))
    # Convert into Numpy array
    np_image = np.array(im)
    #Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1.
    np_image = np_image/255
    #list to array
    np_mean = np.array( [0.485, 0.456, 0.406])
    np_std = np.array([0.229, 0.224, 0.225])
    #the network expects the images to be normalized in a specific way.
    image_normalized = (np_image - np_mean) / np_std
    
    image_transpose = image_normalized.transpose(2,0,1)
    
    return torch.from_numpy(image_transpose)
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    #ax.imshow(image)
    
    return ax

image_output = process_image('ImageClassifier/flowers/test/1/image_06743.jpg')
#imshow(image_output)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    with torch.no_grad():
        image = process_image(image_path)
        image.unsqueeze_(0)
        image = image.float()
        
        logps = model.forward(image)       
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(5, dim=1)
        top_p = top_p.tolist()[0]
        top_class = top_class.tolist()[0]
        idx_to_class = {model.class_to_idx[i]: i for i in model.class_to_idx}
        labels = []
    for c in top_class:
        labels.append(cat_to_name[idx_to_class[c]])

    return top_p, labels
        
print('here')    
probs, classes = predict('ImageClassifier/flowers/test/54/image_05402.jpg', model)
print(probs)
print(classes)        



# TODO: Display an image along with the top 5 classes
fig = plt.figure(figsize = (6,10))

#Create axes for flower
ax = fig.add_axes([.2, .4, .445, .445])

#Display + process image
result = process_image('flowers/test/25/image_06583.jpg')
ax = imshow(result, ax);

#Title for graph


#Predict image
probs, classes = predict('flowers/test/25/image_06583.jpg', model)

#Displays bar graph with axes
ax1 = fig.add_axes([0, -.355, .775, .775])

# Get range for probabilities
y_pos = np.arange(len(classes))

# Plot as a horizontal bar graph
plt.barh(y_pos, probs, align='center', alpha=0.5)
plt.yticks(y_pos, classes)
plt.xlabel('probabilities')
plt.savefig('final.png',dpi=fig.dpi)