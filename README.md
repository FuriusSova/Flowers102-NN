1. **Building the virtual environment**

The network was trained on the university's servers `csgpu1.cs.york.ac.uk`, 'csgpu1.cs.york.ac.uk', and 'csgpu6.cs.york.ac.uk', equipped with suitable hardware for training neural networks.
The creation of the virtual environment was needed to prevent exceeding the disc quota.

The commands used for the servers `csgpu1.cs.york.ac.uk`, and `csgpu1.cs.york.ac.uk`:

```
python -m venv /scratch/student/{university username}/torch-env
source /scratch/student/{university username}/torch-env/bin/activate
```

The commands used for server `csgpu6.cs.york.ac.uk`:

```
python -m venv /tmp/{university username}/torch-env
source /tmp/{university username}/torch-env/bin/activate
```

2. **Installing the required packages**

Using the created virtual environment, install the required packages using the following commands:

```
pip3 install torch torchvision torchaudio
pip install scikit-learn
pip install matplotlib
```

3. **Training the model**

To train the model, you will need to open the `training.py` file and specify the directory where you would like the datasets to be downloaded. To do that, find the variable `directory` in the `training.py`, and define it with the string containing the desired directory, for instance:
```python
# ... some code

directory = '/tmp/{university username}/train'

# ... some code
```

After specifying the directory for the datasets, you can run the code using the created virtual environment, for example, with the directory of the virtual environment being `/tmp/{university username}/torch-env/`, and the directory where the file `training.py` is located being `/home/userfs/d/{university username}/Desktop/Flowers102-NN/training.py`:
```
/tmp/{university username}/torch-env/bin/python /home/userfs/d/{university username}/Desktop/Flowers102-NN/training.py
```

During the training process, you will be able to see the training progress in the terminal, and at the end of training, the final output in the terminal will specify the accuracy and loss of the trained model on the test dataset. Furthermore, the 'plot.png' file will be created in the same directory where the `training.py` file is located.

4. **Testing a pretrained model**

To train the model, you will need to open the `runTrainedModel.py` file and specify the directory where you would like the datasets to be downloaded. To do that, find the variable `directory` in the `runTrainedModel.py`, and define it with the string containing the desired directory, for instance:
```python
# ... some code

directory = '/tmp/{university username}/train'

# ... some code
```

After specifying the directory for the datasets, you can run the code using the created virtual environment, for example, with the directory of the virtual environment being `/tmp/{university username}/torch-env/`, and the directory where the file `runTrainedModel.py` is located being `/home/userfs/d/{university username}/Desktop/Flowers102-NN/runTrainedModel.py`:
```
/tmp/{university username}/torch-env/bin/python /home/userfs/d/{university username}/Desktop/Flowers102-NN/runTrainedModel.py
```

At the end of training, the final output in the terminal will specify the accuracy and loss of the pretrained model on the test dataset.