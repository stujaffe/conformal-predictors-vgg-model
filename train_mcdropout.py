"""
train_modified.py but with the following changes:
# Add monte carlo dropout to the training and inference/prediction stage of the model
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
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import time
import copy
import sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from datetime import datetime

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

warnings.filterwarnings("ignore")

IMAGE_DIR = "./images/"

OUTPUT_DIR = "./output/"

BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_CLASSES = 114
NUM_MC_SAMPLES = 5

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


######################################################################################################################################################################################
# Monte Carlo Dropout Definitions
######################################################################################################################################################################################

class MC_Dropout(nn.Module):
    def __init__(self, p):
        super(MC_Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        # We want to set training=True even for the inference/prediction stage
        return nn.functional.dropout(x, p=self.p, training=True)

def predict_mc_dropout(model, inputs, num_samples=50):
    # Set model evaluation mode
    model.eval()

    # Activate training mode only for the dropout layers so we maintain the evaluation mode for the other non-dropout layers
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

    # Aggregate the outputs over the number of samples
    outputs = [model(inputs) for _ in range(num_samples)]
    # Take the mean of our samples, return as log probabilities
    output_mc = torch.stack(outputs).mean(dim=0)

    return output_mc

######################################################################################################################################################################################
# Other model definitions
######################################################################################################################################################################################


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
    criterion,
    optimizer,
    scheduler,
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
                scheduler.step()
            else:
                model.eval()  # Set model to evaluate mode

            # Only eval the model every 10 epochs to save time
            if phase == "val" and (epoch+1) % 10 != 0:
                print(f"Currently in epoch {epoch}, not evaluating the model in this epoch.")
                continue

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
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    inputs = inputs.float()  # ADDED AS A FIX
                    # Use mc dropout function for prediction
                    outputs = predict_mc_dropout(model=model, inputs=inputs, num_samples=NUM_MC_SAMPLES)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # print("Loss: {}/{}".format(running_loss, dataset_sizes[phase]))
            print("Accuracy: {}/{}".format(running_corrects, dataset_sizes[phase]))
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            training_results.append([phase, epoch, epoch_loss, epoch_acc])
            if epoch >= 10:
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
    model.load_state_dict(best_model_wts)
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
    print(f"Fitzpatrick dataframe shape after filtering: {df.shape}")
    print(df["fitzpatrick_scale"].value_counts())
    print("Rows: {}".format(df.shape[0]))
    df["low"] = df["label"].astype("category").cat.codes
    df["mid"] = df["nine_partition_label"].astype("category").cat.codes
    df["high"] = df["three_partition_label"].astype("category").cat.codes
    df["hasher"] = df["md5hash"]

    for holdout_set in HOLDOUT_SET_LIST:
        if holdout_set == "expert_select":
            df2 = df
            train = df2[df2.qc.isnull()]
            test = df2[df2.qc == "1 Diagnostic"]
        elif holdout_set == "random_holdout":
            train, test, y_train, y_test = train_test_split(
                df, df.low, test_size=0.2, random_state=4242, stratify=df.low
            )
        elif holdout_set == "random_holdout50":
            train, test, y_train, y_test = train_test_split(
                df, df.low, test_size=0.5, random_state=4242, stratify=df.low
            )
        elif holdout_set == "dermaamin":
            combo = set(
                df[df.url.str.contains("dermaamin") == True].label.unique()
            ) & set(df[df.url.str.contains("dermaamin") == False].label.unique())
            df = df[df.label.isin(combo)]
            df["low"] = df["label"].astype("category").cat.codes
            train = df[df.url.str.contains("dermaamin") == False]
            test = df[df.url.str.contains("dermaamin")]
        elif holdout_set == "br":
            combo = set(
                df[df.url.str.contains("dermaamin") == True].label.unique()
            ) & set(df[df.url.str.contains("dermaamin") == False].label.unique())
            df = df[df.label.isin(combo)]
            df["low"] = df["label"].astype("category").cat.codes
            train = df[df.url.str.contains("dermaamin")]
            test = df[df.url.str.contains("dermaamin") == False]
            print(train.label.nunique())
            print(test.label.nunique())
        elif holdout_set == "a12":
            train = df[(df.fitzpatrick_scale == 1) | (df.fitzpatrick_scale == 2)]
            test = df[(df.fitzpatrick_scale != 1) & (df.fitzpatrick_scale != 2)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print(combo)
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train["label"].astype("category").cat.codes
            test["low"] = test["label"].astype("category").cat.codes
        elif holdout_set == "a34":
            train = df[(df.fitzpatrick_scale == 3) | (df.fitzpatrick_scale == 4)]
            test = df[(df.fitzpatrick_scale != 3) & (df.fitzpatrick_scale != 4)]
            combo = set(train.label.unique()) & set(test.label.unique())
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train["label"].astype("category").cat.codes
            test["low"] = test["label"].astype("category").cat.codes
        elif holdout_set == "a56":
            train = df[(df.fitzpatrick_scale == 5) | (df.fitzpatrick_scale == 6)]
            test = df[(df.fitzpatrick_scale != 5) & (df.fitzpatrick_scale != 6)]
            combo = set(train.label.unique()) & set(test.label.unique())
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train["label"].astype("category").cat.codes
            test["low"] = test["label"].astype("category").cat.codes
        print(test.shape)
        print(test.shape)
        train_path = f"{OUTPUT_DIR}temp_train_{holdout_set}.csv"
        test_path = f"{OUTPUT_DIR}temp_test_{holdout_set}.csv"
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
            model_ft = models.vgg16(pretrained=True)
            # Freeze all of the model so nothing is adjusted during training, including the covolutional and pooling layers and the classifier (i.e. fully connected) layers
            for param in model_ft.parameters():
                param.requires_grad = False
            
            # Add more fully connected layers to see if we can improve VGG model
            # The number of trainable parameters was 4,487,026 versus 1,078,130 in the smaller Sequential classifier layer below.
            model_ft.classifier[6] = nn.Sequential(
                nn.Linear(4096, 1024),
                nn.ReLU(),
                MC_Dropout(0.4), # Use the MC_Dropout class so dropout is applied in training and inference/prediction
                nn.Linear(1024, 256), # Add in another intermediate linear layer, ReLU() activation function, and dropout
                nn.ReLU(),
                MC_Dropout(0.4),
                nn.Linear(256, len(label_codes)),
                nn.LogSoftmax(dim=1),
            )
            """
            ### Old additional classification layer from the paper authors
            # Add a classifier layer at the end that outputs the desired number of log probabilities based on the number of skin conditions (usually 114 classes)
            model_ft.classifier[6] = nn.Sequential(
                nn.Linear(4096, 256),
                nn.ReLU(),
                MC_Dropout(0.4), # Use the MC_Dropout class so dropout is applied in training and inference/prediction
                nn.Linear(256, len(label_codes)),
                nn.LogSoftmax(dim=1),
            )
            """
            total_params = sum(p.numel() for p in model_ft.parameters())
            print("{} total parameters".format(total_params))
            total_trainable_params = sum(
                p.numel() for p in model_ft.parameters() if p.requires_grad
            )
            print("{} total trainable parameters".format(total_trainable_params))
            model_ft = model_ft.to(device)
            model_ft = nn.DataParallel(model_ft)
            class_weights = torch.FloatTensor(weights).to(device)
            criterion = nn.NLLLoss()
            optimizer_ft = optim.Adam(model_ft.parameters())
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
            print("\nTraining classifier for {}........ \n".format(label))
            print("....... processing ........ \n")
            model_ft, training_results = train_model(
                label,
                dataloaders,
                device,
                dataset_sizes,
                model_ft,
                criterion,
                optimizer_ft,
                exp_lr_scheduler,
                n_epochs,
            )
            print("Training Complete")
            torch.save(
                model_ft.state_dict(),
                "{}model_mcd_path_{}_{}_{}.pth".format(
                    OUTPUT_DIR, n_epochs, label, holdout_set
                ),
            )
            print("gold")
            training_results.to_csv(
                "{}training_mcd_{}_{}_{}.csv".format(
                    OUTPUT_DIR, n_epochs, label, holdout_set
                )
            )
            model = model_ft.eval()
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
                print("Doing the final validation set predictions.")
                running_corrects = 0
                for i, batch in enumerate(dataloaders["val"]):
                    inputs = batch["image"].to(device)
                    classes = batch[label].to(device)
                    fitzpatrick_scale = batch["fitzpatrick_scale"]
                    hasher = batch["hasher"]
                    inputs = inputs.float()
                    outputs = predict_mc_dropout(model=model, inputs=inputs, num_samples=NUM_MC_SAMPLES)
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
                "{}results_mcd_{}_{}_{}.csv".format(
                    OUTPUT_DIR, n_epochs, label, holdout_set
                ),
                index=False,
            )
            print("\n Accuracy: {} \n".format(acc))
        print("done")
    end_time = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")