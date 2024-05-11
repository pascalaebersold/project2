# SML Project 2
This folder contains the student template of project 2


## Environment Setup
### Google Colab
For those who are running the project with Google Colab, we prepared the interface to run the code in the `handout.ipynb`.

Please check out the installation guide on Moodle for this.
Make sure you upload the whole project2 folder (including uncompressed dataset) to your goolge drive and follow the instructions in the `handout.ipynb` to run the code.

### Local Installation
For those who are running the project locally, please set up an Anaconda environment running `python3.10`. Please check out the installation guide on [Moodle](https://moodle-app2.let.ethz.ch/course/view.php?id=21784) for this.

If you are using Windows, we recommend to use either the VS code terminal or the Anaconda terminal, which is installed with Anaconda.

Please activate your project 2 environment by using:
```
conda activate <environment_name>
```
Then navigate to the folder containing the project files and run:
```
pip install --upgrade pip
pip install -r requirements.txt
```
If you require any additional packages, run:
```
pip install <package_name>
```

Make sure to extract all the data in the `./datasets` folder.

## Running Code
Please note that the scripts `train.py` and `test.py` take arguments when you run them. These arguments are used when the script is carried out. The arguments to a python script can be specified in the following manner:
```bash
python train.py --<argument_1_name> <argument_1_value> --<argument_2_name> <argument_2_value>
```
`train.py` takes two arguments, namely the path to the datasets folder and where the training loop should save your model checkpoints.
`test.py` also takes two arguments, namely the path to the datasets folder, and which test split should be loaded.

For more information on the available arguments to these scripts, please run the following command:
```bash
python train.py -h
python test.py -h
```
### Google Colab
Check out the `Instructions_GoogleColab.ipynb` for instructions on how to run the code in Google Colab.

### Local Installation
To run your solution locally, first make sure you have activated your conda environment. Then open a terminal and run the following command with your arguments to train the model:
```bash
python train.py <your_arguments_here>
```
After training, you can generate the public test predictions and evaluate them by running:
```bash
python test.py --ckpt <path_to_checkpoint> --split public_test
```
The `<path_to_checkpoint>` is the path to the checkpoint file you want to use for generating the predictions. Recall that the `train.py` script saves your model parameters in the form of a checkpoint after each epoch.
The output predictions will be saved in the `./public_test/prediction` folder, which is created if it does not exist yet.

To generate the private test predictions, run:
```bash
python test.py --ckpt <path_to_checkpoint> --split private_test
```
This will generate the predictions for the private test set and save them in the `./private_test/prediction` folder, which is ccreated if it does not exist yet.

## Submission
Please submit the single zip file following the project description 6.2.
A possible method of doing so is to put every file under the folder <submit_file_name>, then running the following command:
```bash
zip -r <submit_file_name>.zip <submit_file_name>
```
Then submit the zip file on Moodle.
