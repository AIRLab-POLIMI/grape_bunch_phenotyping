import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from configs.base_cfg import get_base_cfg_defaults
from torcheval.metrics import R2Score
from torcheval.metrics import MeanSquaredError
from datasets.grape_bunches_dataset import GrapeBunchesDataset
from datasets.vine_plants_dataset import VinePlantsDataset

# Logging metadata with Neptune
import neptune.new as neptune

run = neptune.init_run(project='AIRLab/grape-bunch-phenotyping',
                       mode='debug',        # use 'debug' to turn off logging, 'async' otherwise
                       name='CNNRegressor',
                       tags=['scaling_from_train', 'all_images', 'without_depth', 'stratified_split', 'not_occluded'])


class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the convolutional layers
        self.conv1_1 = nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1)  # the number of input channels is inferred from the input.size(1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(64, 197, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(197)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(197, 256, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout2d(p=0.2)

        # Define the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the fully connected layers
        self.fc1 = nn.LazyLinear(4096)           # LazyLinear automatically infers the input size
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 2622)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(2622, 1)

    def forward(self, x):
        # batch normalization is applied before relu as suggested in the
        # original paper but there is a debate whether is better before
        # or after
        x = self.dropout1(F.relu(self.bn1(self.conv1_2(self.conv1_1(x)))))
        x = self.pool(x)
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(x)
        x = self.dropout4(F.relu(self.bn4(self.conv4_2(self.conv4_1(x)))))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return x


# Training loop
def train(dataloader, model, loss_fn, optimizer, device):
    r2score_metric = R2Score(device=device)
    mse_metric = MeanSquaredError(device=device)
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device, torch.float32)
        y = y.unsqueeze(1)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Compute metrics
        r2score_metric.update(pred, y)
        r2 = r2score_metric.compute()
        mse_metric.update(pred, y)
        mse = mse_metric.compute()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 4 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Train error: R2: {r2:>7f}, MSE: {mse:>7f}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return r2, mse, loss


# Check test accuracy
def test(dataloader, model, loss_fn, device):
    r2score_metric = R2Score(device=device)
    mse_metric = MeanSquaredError(device=device)
    num_batches = len(dataloader)
    training_mode = model.training
    # call model.eval() method before inferencing to set the dropout and batch
    # normalization layers to evaluation mode
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device, torch.float32)
            y = y.unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            r2score_metric.update(pred, y)
            r2 = r2score_metric.compute()
            mse_metric.update(pred, y)
            mse = mse_metric.compute()
    test_loss /= num_batches
    print(f"Test Error: \n R2: {r2:>7f}, MSE: {mse:>7f}, Avg loss: {test_loss:>7f} \n")
    model.train(training_mode)

    return r2, mse, test_loss


def main():
    # custom configurations
    cfg = get_base_cfg_defaults()
    # if args.eval_only:
    #     cfg.merge_from_file("./configs/base_test_cfg.yaml")
    # else:
    #     cfg.merge_from_file("./configs/base_train_cfg.yaml")
    cfg.merge_from_file("./configs/base_train_cfg.yaml")
    cfg.freeze()

    # Log parameters in Neptune
    PARAMS = {'image_size': str(cfg.DATASET.IMAGE_SIZE),
              'crop_size': str(cfg.DATASET.CROP_SIZE),
              'masking': cfg.DATASET.MASKING,
              'target_scaling': str(cfg.DATASET.TARGET_SCALING),
              'not_occluded': cfg.DATASET.NOT_OCCLUDED,
              'epochs': cfg.SOLVER.EPOCHS,
              'base_lr': cfg.SOLVER.BASE_LR,
              'batch_size': cfg.DATALOADER.BATCH_SIZE,
              'optimizer': 'SGD',
              'momentum': cfg.SOLVER.MOMENTUM,
              'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
              'nesterov': cfg.SOLVER.NESTEROV
              }

    # Pass parameters to the Neptune run object
    run['cfg_parameters'] = PARAMS

    # convert into float32 and scale into [0,1]
    convertdtype = T.ConvertImageDtype(torch.float32)

    # TODO: add image standardization with mean and std
    color_transforms = T.Compose([
        T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0)]),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))])
        ])

    color_depth_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(0, 20)),
        T.RandomAffine(degrees=0, translate=(0.1, 0.3))
    ])

    train_dataset = None
    test_dataset = None

    if cfg.DATASET.TYPE == 'bunches':
        train_dataset = GrapeBunchesDataset(cfg.DATASET.ANNOTATIONS_PATH_TRAIN,
                                            cfg.DATASET.TARGET,
                                            cfg.DATASET.IMAGES_PATH_TRAIN,
                                            cfg.DATASET.IMAGE_SIZE,
                                            cfg.DATASET.CROP_SIZE,
                                            depth_dir=cfg.DATASET.DEPTH_PATH_TRAIN,
                                            apply_mask=cfg.DATASET.MASKING,
                                            color_transform=color_transforms,
                                            color_depth_transform=color_depth_transforms,
                                            target_scaling=cfg.DATASET.TARGET_SCALING,
                                            horizontal_flip=True,
                                            not_occluded=cfg.DATASET.NOT_OCCLUDED)
        test_dataset = GrapeBunchesDataset(cfg.DATASET.ANNOTATIONS_PATH_TEST,
                                            cfg.DATASET.TARGET,
                                            cfg.DATASET.IMAGES_PATH_TEST,
                                            cfg.DATASET.IMAGE_SIZE,
                                            cfg.DATASET.CROP_SIZE,
                                            depth_dir=cfg.DATASET.DEPTH_PATH_TEST,
                                            apply_mask=cfg.DATASET.MASKING,
                                            color_transform=None,
                                            color_depth_transform=None,
                                            target_scaling=train_dataset.min_max_target,
                                            not_occluded=cfg.DATASET.NOT_OCCLUDED)
    elif cfg.DATASET.TYPE == 'plants':
        train_dataset = VinePlantsDataset(cfg.DATASET.ANNOTATIONS_PATH_TRAIN,
                                          cfg.DATASET.TARGET,
                                          cfg.DATASET.IMAGES_PATH_TRAIN,
                                          cfg.DATASET.IMAGE_SIZE,
                                          depth_dir=cfg.DATASET.DEPTH_PATH_TRAIN,
                                          transform=transforms,
                                          depth_transform=convertdtype,
                                          target_scaling=cfg.DATASET.TARGET_SCALING,
                                          horizontal_flip=True,
                                          not_occluded=cfg.DATASET.NOT_OCCLUDED)
        test_dataset = VinePlantsDataset(cfg.DATASET.ANNOTATIONS_PATH_TEST,
                                          cfg.DATASET.TARGET,
                                          cfg.DATASET.IMAGES_PATH_TEST,
                                          cfg.DATASET.IMAGE_SIZE,
                                          depth_dir=cfg.DATASET.DEPTH_PATH_TEST,
                                          transform=convertdtype,
                                          depth_transform=convertdtype,
                                          target_scaling=train_dataset.min_max_target,
                                          not_occluded=cfg.DATASET.NOT_OCCLUDED)

    assert cfg.DATALOADER.BATCH_SIZE <= len(train_dataset), "Batch size is larger than the training dataset size"

    # Create data loaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.DATALOADER.BATCH_SIZE,
                                  shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=len(test_dataset),
                                 shuffle=False,
                                 drop_last=False)

    # Display images and labels
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[sample_idx]
        img = img[0:3, :, :]
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
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR,
                                momentum=cfg.SOLVER.MOMENTUM,
                                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                nesterov=cfg.SOLVER.NESTEROV)

    # Iterate training over epochs
    for t in range(cfg.SOLVER.EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        r2, mse, loss = train(train_dataloader, model, loss_fn, optimizer, device)
        run['metrics/R2_score_train'].log(r2)
        run['metrics/MSE_train'].log(mse)
        run['metrics/total_loss_train'].log(loss)
        r2, mse, loss = test(test_dataloader, model, loss_fn, device)
        run['metrics/R2_score_test'].log(r2)
        run['metrics/MSE_test'].log(mse)
        run['metrics/total_loss_test'].log(loss)
    print("Done!")


if __name__ == '__main__':
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

    main()
