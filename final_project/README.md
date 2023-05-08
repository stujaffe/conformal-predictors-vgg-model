### Code Adapted From The Forked Repo Fitzpatrick17k


#### How to run the code:

1. Install the `requirements.txt` in the parent directory.
2. Create a directory called `images`.
3. Run the `download_images.py` file. Make sure the `fitzpatrick17k.csv` file is in the same directory. The images will be downloaded to the directory `images`.
4. Create a directory called `output`.
5. To run `train_modified.py`:
    a. Command is: `python3 train_modified.py [num_epochs] [full|dev]`. E.g. `python3 train_modified.py 20 full`
    b. `full` means the full `fitzpatrick17k.csv` dataset and `dev` means run only 1000 samples from the dataset. Note that with `dev` mode you may run into issues when splitting the dataset using group     stratification due to class imbalances.
    c. Run `train_modified.py` on a CUDA enabled machine or else it will take forever, until the end of time basically. An AWS EC2 g4dn.4xlarge was used and it took around 20-25 minutes per epoch.
    d. The model will run for 20 epochs over 8 different variations of train/test splits, 7 from the original authors and 1 added by us. You can modify which variations are run with the global variable `HOLDOUT_SET_LIST` at the beginning of the script.
    e. All model outputs will be, as you might suspect, in the directory `output`.
6. To run `train_mcdropout.py` is the same process as `train_modified.py`. We specifically only ran `train_mcdropout.py` for the `random_holdout50` train/test split due to the extra computational overhead of the Monte Carlo dropout. You can adjust the train/test split groups in `HOLDOUT_SET_LIST`. To increase the number of MC dropout iterations, adjust the global variable `NUM_MC_SAMPLES`.
7. We were not able to run `train_varinf.py` due to lack of GPU memory on the EC2 instance (16GB). Reducing batch size, split size, etc did not help. The VGG model is quite large to begin with, plus the image data, so memory was consumed quickly.
