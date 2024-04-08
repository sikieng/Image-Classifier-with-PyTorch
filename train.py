import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json
import all_functions
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action="store", metavar='data_dir', type=str,default="./flowers/")
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='checkpoint.pth')
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()


def main():
    print("Loading the data...")
    print("Hello")
    args = parse_args()
    #print("Hello1")
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_data = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    testing_data = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    validation_data = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = [
        datasets.ImageFolder(train_dir, transform=training_data),
        datasets.ImageFolder(valid_dir, transform=validation_data),
        datasets.ImageFolder(test_dir, transform=testing_data)
    ]

    dataloaders = [
        torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
        torch.utils.data.DataLoader(image_datasets[1], batch_size=32, shuffle=True),
        torch.utils.data.DataLoader(image_datasets[2], batch_size=32, shuffle=True)
    ]

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    model, device, num_in_features = all_functions.load_model(args.arch, args.hidden_units, args.gpu)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    all_functions.train_model(args.epochs, dataloaders[0], dataloaders[1], model, device, criterion, optimizer)

    file_path = args.save_dir

    output_size = 102
    all_functions.save_checkpoint(file_path, model, image_datasets, args.epochs, optimizer, args.learning_rate,
                    num_in_features, output_size, args.arch, args.hidden_units)


if __name__ == "__main__":
    main()
