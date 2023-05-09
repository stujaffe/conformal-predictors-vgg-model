### Code Adapted From The Forked Repo Fitzpatrick17k


#### How to run the code:

1. Install the `requirements.txt` in the parent directory.
2. Create a directory called `images`.
3. Run the `download_images.py` file. Make sure the `fitzpatrick17k.csv` file is in the same directory. The images will be downloaded to the directory `images`.
4. Create a directory called `output`.
5. To run `train_modified.py`:
    a. Command is: `python3 train_modified.py [num_epochs] [full|dev]`. E.g. `python3 train_modified.py 20 full` <br />
    b. `full` means the full `fitzpatrick17k.csv` dataset and `dev` means run only 1000 samples from the dataset. Note that with `dev` mode you may run into issues when splitting the dataset using group stratification due to class imbalances. <br />
    c. Run `train_modified.py` on a CUDA enabled machine or else it will take forever, until the end of time basically. An AWS EC2 g4dn.4xlarge was used and it took around 60 to 90 seconds per epoch. <br />
    d. The model will run for 20 epochs over 8 different variations of train/test splits, 7 from the original authors and 1 added by us. You can modify which variations are run with the global variable `HOLDOUT_SET_LIST` at the beginning of the script. <br />
    e. All model outputs will be, as you might suspect, in the directory `output`. <br />
6. To run `train_mcdropout.py` is the same process as `train_modified.py`. We specifically only ran `train_mcdropout.py` for the `random_holdout50` train/test split due to the extra computational overhead of the Monte Carlo dropout. You can adjust the train/test split groups in `HOLDOUT_SET_LIST`. To increase the number of MC dropout iterations, adjust the global variable `NUM_MC_SAMPLES`.
7. To run `train_resnet.py` is the same process as `train_modified.py`.
8. We were not able to run `train_varinf.py` due to lack of GPU memory on the EC2 instance (16GB). Reducing batch size, split size, etc did not help. The VGG model is quite large to begin with, plus the image data, so memory was consumed quickly.
9. To run `run_model.py`:
    a. This script will output all of the log probabilities given a PyTorch model and the test results from a model having only the top 3 log probability outputs. <br />
    b. Specify the directories and file paths: <br />
        1. `IMAGE_DIR`: The directory of images specified earlier. <br />
        2. `MODEL_PATH`: The path of the PyTorch model file produced in training. <br />
        3. `TEST_RESULTS_PATH`: The path of the output of the test set results from training. <br />
        4. `HOLDOUT_SET`: The particular train/test split on which you trained the model (for labeling files only). <br />
        5. `DATALOADER_NAME`: "val" or "train" depending if you want all the log probabilities for the training or validation (test) set. <br />
