import argparse
import torch
import torchvision
import all_functions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar='image_path', type=str, default='flowers/test/1/image_06752.jpg')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    model = all_functions.load_checkpoint(checkpoint)

    top_p, classes, device = all_functions.predict(image_path, model, top_k, gpu)

    category_names = all_functions.load_names(category_names)

    labels = [category_names[str(index)] for index in classes]

    print(f"Results for your File: {image_path}")
    print(labels)
    print(top_p)

    for i in range(len(labels)):
        print("{} - {} with a probability of {}".format((i+1), labels[i], top_p[i]))


if __name__ == "__main__":
    main()
