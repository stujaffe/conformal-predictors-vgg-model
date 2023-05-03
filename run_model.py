import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import numpy as np
import time
from datetime import datetime

from train_modified import custom_load, flatten

IMAGE_DIR = "./images/"
MODEL_PATH = "./results_20epoch/model_path_20_low_random_holdout.pth"
TEST_RESULTS_PATH = "./results_20epoch/results_20_low_random_holdout.csv"
MODEL_RUN_OUTPUT_DIR = "./results_alllogits_20epoch/"

NUM_CLASSES = 114
BATCH_SIZE = 64
NUM_WORKERS = 12
LABEL = "low"
HOLDOUT_SET = "random_holdout"
DATALOADER_NAME = "val"

if __name__ == "__main__":
    start_time = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")

    print(f"Loading the following model: {MODEL_PATH}")
    print(f"Using the following data as the implied test set: {TEST_RESULTS_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for PyTorch operations.")
    # Load the model checkpoint data
    if device == "cuda":
        checkpoint = torch.load(f=MODEL_PATH)
    else:
        checkpoint = torch.load(f=MODEL_PATH, map_location=torch.device("cpu"))

    # Instantiate the same model used in train.py and load the checkpoint data into it
    model_ft = models.vgg16(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    
    model_ft.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES),
        nn.LogSoftmax(dim=1),
    )
    
    model_ft = nn.DataParallel(model_ft)

    model_ft.load_state_dict(checkpoint)

    total_params = sum(p.numel() for p in model_ft.parameters())
    print("{} total parameters".format(total_params))

    # Load data
    df_all = pd.read_csv("fitzpatrick17k.csv")
    # Filter the results based on the md5hash values of the images we were able to download
    df_imgname = pd.read_csv("md5hash_image_downloads.csv", index_col=0)
    df_filtered = df_all[df_all["md5hash"].isin(df_imgname["md5hash_downloads"])]

    # Replicate additions to dataframe found in train.py
    df_filtered["low"] = df_filtered["label"].astype("category").cat.codes
    df_filtered["mid"] = (
        df_filtered["nine_partition_label"].astype("category").cat.codes
    )
    df_filtered["high"] = (
        df_filtered["three_partition_label"].astype("category").cat.codes
    )
    df_filtered["hasher"] = df_filtered["md5hash"]

    # Split the filtered dataframe into test and train based on the md5hash values in the results CSV file
    df_results = pd.read_csv(TEST_RESULTS_PATH)
    df_train = df_filtered[~df_filtered["md5hash"].isin(df_results["hasher"])]
    df_test = df_filtered[df_filtered["md5hash"].isin(df_results["hasher"])]

    print(f"Number of data points in training set: {df_train.shape[0]}")
    print(f"Number of data points in test set: {df_test.shape[0]}")
    print(
        f"Number of data points in the results csv file (any discrepancy to test set is due to image availability): {df_results.shape[0]}"
    )

    # Save test and train
    df_train.to_csv(path_or_buf=f"{MODEL_RUN_OUTPUT_DIR}temp_train_{HOLDOUT_SET}_alllogits.csv")
    df_test.to_csv(path_or_buf=f"{MODEL_RUN_OUTPUT_DIR}temp_test_{HOLDOUT_SET}_alllogits.csv")

    # Get the dataloaders and dataset sizes using the custom_load() function from train.py so we can mimic the transformations done on the images when they are run through the model.
    dataloaders, dataset_sizes = custom_load(
        batch_size=BATCH_SIZE,
        num_workers=NUM_CLASSES,
        train_dir=f"{MODEL_RUN_OUTPUT_DIR}temp_train_{HOLDOUT_SET}_alllogits.csv",
        val_dir=f"{MODEL_RUN_OUTPUT_DIR}temp_test_{HOLDOUT_SET}_alllogits.csv",
        image_dir=IMAGE_DIR,
        label=LABEL,
    )

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
    logits_list = []
    model = model_ft.eval()
    model.to(device)

    with torch.no_grad():
        running_corrects = 0
        for i, batch in enumerate(dataloaders[DATALOADER_NAME]):
            inputs = batch["image"].to(device)
            classes = batch[LABEL].to(device)
            fitzpatrick_scale = batch["fitzpatrick_scale"]
            hasher = batch["hasher"]
            outputs = model(inputs.float())
            probability = outputs
            ppp, preds = torch.topk(probability, 1)
            if LABEL == "low":
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
    
    # For each batch, outputs.tolist() results in a list of lists where the outer list is of length "batch size" (e.g. 64) and each inner list
    # is of length "number of classes" (e.g. 114 in this case). Then logits_list, is another outer list with length of the "number of batches".
    # So we want to "stack" the middle layer so that we have a list of lists with outer list of length "number of images" and each inner length
    # the "number of classes".
    logits_list_flat = [item for sublist in logits_list for item in sublist]

    df_logits = pd.DataFrame(logits_list_flat)
    # Rename columns
    df_logits_colnames = {i:f"logit_class_{i}" for i in range(NUM_CLASSES)}
    df_logits.rename(columns=df_logits_colnames, inplace=True)

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
    df_output = pd.concat([df_x,df_logits], axis=1)

    print(f"Output dataframe shape: {df_output.shape}")

    df_output.to_csv(
        f"{MODEL_RUN_OUTPUT_DIR}results_20_{HOLDOUT_SET}_{DATALOADER_NAME}_alllogits.csv",
        index=False,
    )

    end_time = datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")