import argparse
import os
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import crop
import torchvision.transforms as T
import json
import math
import matplotlib.pyplot as plt


class GrapeBunchesDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, img_size, crop_size, apply_mask=False, transform=None, target_transform=None, target_scaling=None):
        with open(annotations_file) as dictionary_file:
            json_dictionary = json.load(dictionary_file)
        
        self.img_info = json_dictionary['images']
        self.img_dir = img_dir
        self.fixed_img_size = img_size      # img_size expressed as (height, width)
        self.crop_size = crop_size          # crop_size expressed as (height, width)
        self.apply_mask = apply_mask        # whether to isolate the single bunch with its mask
        self.transform = transform
        self.target_transform = target_transform
        self.target_scaling = target_scaling

        filtered_ann = []
        img_id = 0
        for ann in json_dictionary['annotations']:
            if ann['image_id'] != img_id:
                img_id = ann['image_id']
                img_width, img_height = 0, 0
                for img in json_dictionary['images']:
                    if img['id'] == img_id:
                        img_width, img_height = img['width'], img['height']
                        break
            if ann['attributes']['tagged']:
                if ann['attributes']['volume'] > 0.0 and ann['attributes']['weight'] > 0.0:
                    half_crop_width = math.ceil(crop_size[1]/2)
                    half_crop_height = math.ceil(crop_size[0]/2)
                    # check whether bboxes are distant from borders at least half of corresponding crop size
                    # rescale half_crop_... for img_size
                    x_scale, y_scale = self.x_y_scale(img_width, img_height)
                    if x_scale != 1.0:
                        half_crop_width /= x_scale
                    if y_scale != 1.0:
                        half_crop_height /= y_scale
                    from_left = ann['bbox'][0] >= half_crop_width
                    from_top = ann['bbox'][1] >= half_crop_height
                    from_right = img_width-(ann['bbox'][0]+ann['bbox'][2]/2) >= half_crop_width
                    from_bottom = img_height-(ann['bbox'][1]+ann['bbox'][3]/2) >= half_crop_height
                    if from_left and from_top and from_right and from_bottom:
                        filtered_ann.append(ann)
        # we only add filtered annotations, that is, grapes which have been
        # tagged, grapes with a volume/weight value > 0.0, and grapes that
        # are distant from all image borders at least half of crop_size.
        self.img_labels = filtered_ann

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        ann = self.img_labels[idx]            
        label = ann['attributes']['volume']
        if self.target_scaling:
            min = self.target_scaling[0]
            max = self.target_scaling[1]
            label = (label - min) / (max - min)

        img_id = ann['image_id']
        img_filename = []
        img_size = []
        for img in self.img_info:
            if img['id'] == img_id:
                img_filename = img['file_name']
                img_size = [img['height'], img['width']]
                break

        img_path = os.path.join(self.img_dir, img_filename)
        image = read_image(img_path)
        bbox = ann['bbox']                  # bbox format is [x,y,width,height]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # apply segmentation mask if required
        if self.apply_mask:
            pass                            # TODO: implement masking

        # resize the image if needed
        if self.fixed_img_size[0] != img_size[0] or self.fixed_img_size[1] != img_size[1]:
            image = T.Resize(size=self.fixed_img_size)(image)
            # Calculate the scaling factor for the bounding box
            x_scale, y_scale = self.x_y_scale(img_size[1], img_size[0])
            # Resize the bounding box
            bbox = [
                bbox[0] * x_scale,
                bbox[1] * y_scale,
                bbox[2] * x_scale,
                bbox[3] * y_scale
                ]
        # crop the image with a fixed custom bbox around current bbox center
        bbox_center = (bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)        # center coordinates format is (x,y)
        custom_x = round(bbox_center[0] - self.crop_size[1]/2)      
        custom_y = round(bbox_center[1] - self.crop_size[0]/2)     
        assert custom_x >= 0.0
        assert custom_y >= 0.0
        assert custom_x+self.crop_size[1] <= self.fixed_img_size[1]
        assert custom_y+self.crop_size[0] <= self.fixed_img_size[0]

        custom_bbox = (custom_x, custom_y, self.crop_size[1], self.crop_size[0])
        img_crop = crop(image, custom_bbox[1], custom_bbox[0], custom_bbox[3], custom_bbox[2])

        return img_crop, label

    def x_y_scale(self, img_width, img_height):
        x_scale = self.fixed_img_size[1] / img_width
        y_scale = self.fixed_img_size[0] / img_height

        return x_scale, y_scale


class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(128*18*34, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128*18*34)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device, torch.float32)
        y = y.unsqueeze(1)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 4 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Check test accuracy 
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    training_mode = model.training
    # call model.eval() method before inferencing to set the dropout and batch
    # normalization layers to evaluation mode
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device, torch.float32)
            y = y.unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    model.train(training_mode)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, help="path of the json file containing annotations")
    parser.add_argument("img_dir", type=str, help="path of the images directory")
    args = vars(parser.parse_args())

    json_file = args["json_file"]
    img_dir = args["img_dir"]

    transform = T.ConvertImageDtype(torch.float32)
    dataset = GrapeBunchesDataset(json_file, img_dir, (1280, 720), (275, 145),
                                  transform=transform, target_scaling=(0.0, 1080.0))

    # Create data loaders
    batch_size = 4
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    test_dataloader = DataLoader(dataset, batch_size=batch_size)

    # Display images and labels
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.savefig('my_plot.png')

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    model = CNNRegressor().to(device)
    print(model)

    # Optimizing the model parameters
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Iterate training over epochs
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")


# #### Save the mdoel ####
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")


# #### Make predictions ####
# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
