"""
train_modified.py but with the following changes:
# Add variational inference through the pyro library and the required changes to certain PyTorch API calls
"""

from __future__ import print_function, division
import torch
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io
import warnings
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
import time
import copy
import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from datetime import datetime
import os
import sys

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

warnings.filterwarnings("ignore")

IMAGE_DIR = "./images/"

OUTPUT_DIR = "./output/"

BATCH_SIZE = 4
NUM_WORKERS = 10
NUM_CLASSES = 114

HOLDOUT_SET_LIST = [
        #"expert_select",
        #"random_holdout",
        #"a12",
        #"a34",
        #"a56",
        #"dermaamin",
        #"br",
        "random_holdout50",
    ]

"""
class VariationalLinear(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = PyroSample(
            lambda self: dist.Normal(0., 1.).expand([self.out_features, self.in_features]).to_event(2)
        )
        self.bias = PyroSample(
            lambda self: dist.Normal(0., 1.).expand([self.out_features]).to_event(1)
        )

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
"""

class VariationalLinear(PyroModule):
    def __init__(self, in_features, out_features, layer_id):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_id = layer_id

        # register learnable params in Pyro's parameter store
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_log_var = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        weight_std = torch.exp(0.5 * self.weight_log_var)
        bias_std = torch.exp(0.5 * self.bias_log_var)

        # sample weights and biases from normal distribution
        weight_pyro = pyro.sample(
                f"weight_pyro_{self.layer_id}", dist.Normal(self.weight_mu, weight_std).to_event(2)
            )
        if weight_pyro.dim() > 2:
            weight_pyro = torch.mean(weight_pyro, dim=0)
        bias_pyro = pyro.sample(f"bias_pyro_{self.layer_id}", dist.Normal(self.bias_mu, bias_std).to_event(1))

        if bias_pyro.dim() > 1:
            bias_pyro = torch.mean(bias_pyro, dim=0)

        return torch.nn.functional.linear(x, weight_pyro, bias_pyro)


def pyro_model(model, inputs, labels):
    pyro.module("model", model)
    with pyro.plate("data", len(inputs)):
        outputs = model(inputs)
        pyro.sample("obs", dist.Categorical(logits=outputs), obs=labels)

def adjust_learning_rate(pyro_optimizer, epoch, initial_lr, step_size, gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every specified number of epochs"""
    lr = initial_lr * (gamma ** (epoch // step_size))
    for name, _ in pyro.get_param_store().named_parameters():
        pyro_optimizer.set_learning_rate(lr, name)

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def train_model(
    label,
    dataloaders,
    device,
    dataset_sizes,
    model,
    svi,
    initial_lr,
    num_epochs=2,
):
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                adjust_learning_rate(pyro_optimizer, epoch, initial_lr, 2)
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # running_total = 0
            print(phase)
            # Iterate over data.
            dataloader_loop = tqdm(dataloaders[phase])
            dataloader_loop.set_description(desc=f"Epoch [{epoch}/{num_epochs - 1}]")
            for _, batch in enumerate(dataloader_loop):
                inputs = batch["image"].to(device)
                labels = batch[label]
                labels = torch.from_numpy(np.asarray(labels)).to(device)
                # zero the parameter gradients
                pyro.clear_param_store()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    inputs = inputs.float()  # ADDED AS A FIX
                    # Use SVI for loss computation and optimization
                    loss = svi.step(model, inputs, labels)
                    if phase == "train":
                        # No need to perform backward and step, as svi.step already handles optimization
                        pass
                    else:
                        # In validation phase, use svi.evaluate_loss instead of svi.step
                        loss = svi.evaluate_loss(model, inputs, labels)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # statistics
                running_loss += loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # print("Loss: {}/{}".format(running_loss, dataset_sizes[phase]))
            print("Accuracy: {}/{}".format(running_corrects, dataset_sizes[phase]))
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            training_results.append([phase, epoch, epoch_loss, epoch_acc])
            if epoch > 10:
                if phase == "val" and epoch_loss < best_loss:
                    print("New leading accuracy: {}".format(epoch_acc))
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == "val":
                best_loss = epoch_loss
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    pyro.get_param_store().load_state_dict(best_model_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ["phase", "epoch", "loss", "accuracy"]
    return model, training_results


class SkinDataset:
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(
            self.root_dir, self.df.loc[self.df.index[idx], "hasher"]
        )
        image = io.imread(img_name)
        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], "hasher"]
        high = self.df.loc[self.df.index[idx], "high"]
        mid = self.df.loc[self.df.index[idx], "mid"]
        low = self.df.loc[self.df.index[idx], "low"]
        fitzpatrick_scale = self.df.loc[self.df.index[idx], "fitzpatrick_scale"]
        if self.transform:
            image = self.transform(image)
        sample = {
            "image": image,
            "high": high,
            "mid": mid,
            "low": low,
            "hasher": hasher,
            "fitzpatrick_scale": fitzpatrick_scale,
        }
        return sample


def custom_load(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_dir="",
    val_dir="",
    image_dir=IMAGE_DIR,
    label="low",
):
    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    class_sample_count = np.array(train[label].value_counts().sort_index())
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in train[label]])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"), len(samples_weight), replacement=True
    )
    dataset_sizes = {"train": train.shape[0], "val": val.shape[0]}
    transformed_train = SkinDataset(
        csv_file=train_dir,
        root_dir=image_dir,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )
    transformed_test = SkinDataset(
        csv_file=val_dir,
        root_dir=image_dir,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
            # shuffle=True,
            num_workers=num_workers,
        ),
        "val": torch.utils.data.DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return dataloaders, dataset_sizes


if __name__ == "__main__":
    start_time = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    # In the custom_load() function, make sure to specify the path to the images
    print(
        "\nPlease specify number of epochs and 'dev' mode or not... e.g. python train.py 10 full \n"
    )
    n_epochs = int(sys.argv[1])
    dev_mode = sys.argv[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df_imgname = pd.read_csv("md5hash_image_downloads.csv", index_col=0)
    if dev_mode == "dev":
        print("Loading fitzpatrick dataset in dev mode.")
        df = pd.read_csv("fitzpatrick17k.csv").sample(1000)
    else:
        print("Loading fitzpatrick dataset in regular mode.")
        df = pd.read_csv("fitzpatrick17k.csv")
    # Clean dataframe so that only downloaded images are included
    df = df[df["md5hash"].isin(df_imgname["md5hash_downloads"])]
    # Take sample of dataframe
    df, _ = train_test_split(df, test_size=0.5, random_state=42, stratify=df["fitzpatrick_scale"])
    print(f"Fitzpatrick dataframe shape after filtering: {df.shape}")
    print(df["fitzpatrick_scale"].value_counts())
    print("Rows: {}".format(df.shape[0]))
    df["low"] = df["label"].astype("category").cat.codes
    df["mid"] = df["nine_partition_label"].astype("category").cat.codes
    df["high"] = df["three_partition_label"].astype("category").cat.codes
    df["hasher"] = df["md5hash"]

    for holdout_set in HOLDOUT_SET_LIST:
        if holdout_set == "random_holdout50":
            train, test, y_train, y_test = train_test_split(
                df, df.low, test_size=0.5, random_state=4242, stratify=df.low
            )
        train_path = f"{OUTPUT_DIR}temp_train.csv"
        test_path = f"{OUTPUT_DIR}temp_test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        print("Training Shape: {}, Test Shape: {} \n".format(train.shape, test.shape))
        for indexer, label in enumerate(["low"]):
            print(label)
            weights = np.array(
                max(train[label].value_counts())
                / train[label].value_counts().sort_index()
            )
            label_codes = sorted(list(train[label].unique()))
            dataloaders, dataset_sizes = custom_load(
                256, 20, "{}".format(train_path), "{}".format(test_path), label=label
            )
            print(dataset_sizes)
            """
            model_ft = models.vgg16(pretrained=True)
            for param in model_ft.parameters():
                param.requires_grad = False
            model_ft.classifier[6] = nn.Sequential(
                VariationalLinear(4096, 256, 0),
                nn.ReLU(),
                nn.Dropout(0.4),
                VariationalLinear(256, len(label_codes), 1),
                nn.LogSoftmax(dim=1),
            )
            """
            model_ft = models.resnet18(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            print(f"Number of model features: {num_ftrs}")
            num_classes = len(label_codes)
            model_ft.fc = nn.Sequential(
                VariationalLinear(num_ftrs, 256, 0),
                nn.ReLU(),
                nn.Dropout(0.4),
                VariationalLinear(256, len(label_codes), 1),
                nn.LogSoftmax(dim=1),
            )
            total_params = sum(p.numel() for p in model_ft.parameters())
            print("{} total parameters".format(total_params))
            # Count the trainable parameters
            total_trainable_params = sum(p.numel() for p in model_ft.fc.parameters() if p.requires_grad)
            print("{} total trainable parameters".format(total_trainable_params))
            model_ft = model_ft.to(device)
            #model_ft = nn.DataParallel(model_ft)
            class_weights = torch.FloatTensor(weights).to(device)
            criterion = nn.NLLLoss()
            initial_lr = 0.001
            pyro_optimizer = pyro.optim.Adam({"lr": initial_lr})
            guide = AutoDiagonalNormal(pyro_model, init_scale=0.01)
            svi = SVI(pyro_model, guide, pyro_optimizer, loss=Trace_ELBO())
            pyro.clear_param_store()
            print("\nTraining classifier for {}........ \n".format(label))
            print("....... processing ........ \n")
            model_ft, training_results = train_model(
                label,
                dataloaders,
                device,
                dataset_sizes,
                model_ft,
                svi,
                initial_lr,
                n_epochs,
            )
            print("Training Complete")
            pyro.get_param_store().save(
                "{}model_path_{}_{}_{}.pth".format(
                    OUTPUT_DIR, n_epochs, label, holdout_set
                ),
            )
            print("gold")
            training_results.to_csv(
                "{}training_{}_{}_{}.csv".format(
                    OUTPUT_DIR, n_epochs, label, holdout_set
                )
            )
            model = model_ft.eval()
            pyro.get_param_store().load(
                "{}model_path_{}_{}_{}.pth".format(
                    OUTPUT_DIR, n_epochs, label, holdout_set
                ),
            )
            loader = dataloaders["val"]
            prediction_list = []
            fitzpatrick_scale_list = []
            hasher_list = []
            labels_list = []
            p_list = []
            topk_p = []
            topk_n = []
            d1 = []
            d2 = []
            d3 = []
            p1 = []
            p2 = []
            p3 = []
            # List to hold the logits for all classes
            logits_list = []
            with torch.no_grad():
                running_corrects = 0
                for i, batch in enumerate(dataloaders["val"]):
                    inputs = batch["image"].to(device)
                    classes = batch[label].to(device)
                    fitzpatrick_scale = batch["fitzpatrick_scale"]
                    hasher = batch["hasher"]
                    outputs = model(inputs.float())
                    probability = outputs
                    ppp, preds = torch.topk(probability, 1)
                    if label == "low":
                        _, preds5 = torch.topk(probability, 3)
                        topk_p.append(np.exp(_.cpu()).tolist())
                        topk_n.append(preds5.cpu().tolist())
                    running_corrects += torch.sum(preds == classes.data)
                    p_list.append(ppp.cpu().tolist())
                    prediction_list.append(preds.cpu().tolist())
                    labels_list.append(classes.tolist())
                    fitzpatrick_scale_list.append(fitzpatrick_scale.tolist())
                    hasher_list.append(hasher)
                    logits_list.append(outputs.tolist())
                acc = float(running_corrects) / float(dataset_sizes["val"])
            if label == "low":
                for j in topk_n:
                    for i in j:
                        d1.append(i[0])
                        d2.append(i[1])
                        d3.append(i[2])
                for j in topk_p:
                    for i in j:
                        p1.append(i[0])
                        p2.append(i[1])
                        p3.append(i[2])
                df_x = pd.DataFrame(
                    {
                        "hasher": flatten(hasher_list),
                        "label": flatten(labels_list),
                        "fitzpatrick_scale": flatten(fitzpatrick_scale_list),
                        "prediction_probability": flatten(p_list),
                        "prediction": flatten(prediction_list),
                        "d1": d1,
                        "d2": d2,
                        "d3": d3,
                        "p1": p1,
                        "p2": p2,
                        "p3": p3,
                    }
                )
                # Append the logits for all classes
                
                # For each batch, outputs.tolist() results in a list of lists where the outer list is of length "batch size" (e.g. 64) and each inner list
                # is of length "number of classes" (e.g. 114 in this case). Then logits_list, is another outer list with length of the "number of batches".
                # So we want to "stack" the middle layer so that we have a list of lists with outer list of length "number of images" and each inner length
                # the "number of classes".
                logits_list_flat = [item for sublist in logits_list for item in sublist]

                df_logits = pd.DataFrame(logits_list_flat)
                # Rename columns
                df_logits_colnames = {i:f"logit_class_{i}" for i in range(NUM_CLASSES)}
                df_logits.rename(columns=df_logits_colnames, inplace=True)

                # Concat to the existing dataframe
                df_output = pd.concat([df_x,df_logits], axis=1)

            else:
                print(len(flatten(hasher_list)))
                print(len(flatten(labels_list)))
                print(len(flatten(fitzpatrick_scale_list)))
                print(len(flatten(p_list)))
                print(len(flatten(prediction_list)))
                df_output = pd.DataFrame(
                    {
                        "hasher": flatten(hasher_list),
                        "label": flatten(labels_list),
                        "fitzpatrick_scale": flatten(fitzpatrick_scale_list),
                        "prediction_probability": flatten(p_list),
                        "prediction": flatten(prediction_list),
                    }
                )
            df_output.to_csv(
                "{}results_{}_{}_{}.csv".format(
                    OUTPUT_DIR, n_epochs, label, holdout_set
                ),
                index=False,
            )
            print("\n Accuracy: {} \n".format(acc))
        print("done")
    end_time = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")