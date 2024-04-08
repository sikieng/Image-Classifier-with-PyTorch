import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json
import numpy as np
from PIL import Image
import torchvision
import torch.nn.functional as F

def load_model(arch, hidden_units, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        num_in_features = 25088
    else:
        model = models.densenet121(pretrained=True)
        num_in_features = 1024

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_in_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.to(device)
    return model, device, num_in_features


def train_model(epochs, trainloader, validateloader, model, device, criterion, optimizer):
    steps = 0
    running_loss = 0
    print_every = 40
    start = time.time()
    print('Model is Training')
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
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
                    for inputs, labels in validateloader:
                        inputs, labels = inputs.to(device), labels.to(device)  # transfering tensors to the GPU

                        logps = model.forward(inputs)
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {valid_loss / len(validateloader):.3f}.. "
                      f"Accuracy: {accuracy / len(validateloader):.3f}")
                running_loss = 0
                model.train()

    end = time.time()
    total_time = end - start
    print(" Model Trained in: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))


def save_checkpoint(file_path, model, image_datasets, epochs, optimizer, learning_rate, input_size, output_size, arch, hidden_units):
    model.class_to_idx = image_datasets[0].class_to_idx
    bundle = {
        'pretrained_model': arch,
        'input_size': input_size,
        'output_size': output_size,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'classifier': model.classifier,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict()
    }

    torch.save(bundle, file_path)
    print("Model succesfully saved to storage")


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    _model = getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    _model.input_size = checkpoint['input_size']
    _model.output_size = checkpoint['output_size']
    _model.learning_rate = checkpoint['learning_rate']
    _model.hidden_units = checkpoint['hidden_units']
    _model.learning_rate = checkpoint['learning_rate']
    _model.classifier = checkpoint['classifier']
    _model.epochs = checkpoint['epochs']
    _model.load_state_dict(checkpoint['state_dict'])
    _model.class_to_idx = checkpoint['class_to_idx']
    _model.optimizer = checkpoint['optimizer']
    return _model


def process_image(image):
    resize = 256
    crop_size = 224
    (width, height) = image.size

    if height > width:
        height = int(max(height * resize / width, 1))
        width = int(resize)
    else:
        width = int(max(width * resize / height, 1))
        height = int(resize)
    im = image.resize((width, height))
    # crop image
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size
    im = im.crop((left, top, right, bottom))

    # color channels
    im = np.array(im)
    im = im / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    im = np.transpose(im, (2, 0, 1))
    return im


def predict(image_path, model, top_k, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0)
    image = image.float()

    with torch.no_grad():
        output = model.forward(image.cuda())

    p = F.softmax(output.data, dim=1)
    top_p = np.array(p.topk(top_k)[0][0])

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(p.topk(top_k)[1][0])]
    return top_p, top_classes, device


def load_names(category_names_file):
    with open(category_names_file) as file:
        category_names = json.load(file)
    return category_names
