from get_input_args import get_input_args
import torch
from torchvision import datasets,transforms,models
import numpy as np
from torch import nn,optim
import helper
from collections import OrderedDict
from PIL import Image
import json
#import os
#os.environ['QT_QPA_PLATFORM']='offscreen'
#import matplotlib.pyplot as plt


args = get_input_args()
data_dir = args.data_dir
checkpoint_dir = args.save_dir
train_model = args.arch
train_lr = args.learning_rate
train_epochs = args.epochs

train_hidden = args.hidden_layer
train_device = args.gpu



train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
validation_data_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
testing_data_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_image_datasets = datasets.ImageFolder(train_dir,transform = data_transforms )
valid_image_datasets = datasets.ImageFolder(valid_dir,transform = validation_data_transforms )
test_image_datasets = datasets.ImageFolder(test_dir,transform = testing_data_transforms )


# TODO: Using the image datasets and the trainforms, define the dataloaders
train_data = torch.utils.data.DataLoader(train_image_datasets,batch_size=32,shuffle = True)
valid_data = torch.utils.data.DataLoader(train_image_datasets,batch_size=32,shuffle = True)
test_data = torch.utils.data.DataLoader(train_image_datasets,batch_size=32,shuffle = True)

data_iter = iter(valid_data)
images, labels = next(data_iter)
#helper.imshow(images[0],normalize = True)


with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
device = torch.device("cuda" if (torch.cuda.is_available() and train_device is "gpu")  else "cpu")
model=  models.vgg11(pretrained=True) 




#Grads off
for param in model.parameters():
    param.requires_grad = False
    
#modify classifier

classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088,train_hidden)),
                            ('relu', nn.ReLU()),
                            ('dropout',nn.Dropout(.5)),
                            ('fc2', nn.Linear(train_hidden,102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
model.classifier = classifier
model


criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr = train_lr)

model.to(device);
epochs = train_epochs
steps = 0
running_loss = 0
print_every = 100
for epoch in range(epochs):
    for inputs, labels in train_data:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_data:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(valid_data):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_data):.3f}")
            running_loss = 0
            model.train()

# TODO: Do validation on the test set

test_loss = 0
accuracy = 0
model.eval()
model.to(device)
with torch.no_grad():
    for inputs, labels in test_data:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
                    
        test_loss += batch_loss.item()
                    
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape) 
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
    print(f"Epoch {epoch+1}/{epochs}.. "
    f"Test loss: {test_loss/len(test_data):.3f}.. "
      
    f"Validation accuracy: {accuracy/len(test_data):.3f}")
    
# TODO: Save the checkpoint 
model = model.to('cpu')
checkpoint = {
    'input_size' : 25088, 
              'output_size': 102, 
              'classifier': model.classifier,
              'state_dict' : model.classifier.state_dict(),
              'model.class_to_idx' : train_image_datasets.class_to_idx,
              'epochs' : 5,
              'state_optimizer': optimizer.state_dict()
}
torch.save(checkpoint,'checkpoint.pth')





