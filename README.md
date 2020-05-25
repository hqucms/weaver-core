# Weaver

`Weaver` aims at providing a streamlined yet flexible machine learning R&D framework for high energy physics (HEP) applications. It puts particular emphases on:

- handling common HEP dataset formats (ROOT, HDF5, [awkward array](https://github.com/scikit-hep/awkward-array)) efficiently, up to terabyte-level
- providing a simple way to perform input processing *on-the-fly* (e.g., sample selections, new variable definition, inputs transformation/standardization, sample reweighting, etc.)
- bridging the gap between development and production: neural networks are trained with [PyTorch](https://pytorch.org/) and exported to the [ONNX](http://onnx.ai/) format for fast inference (e.g., using [ONNXRuntime](https://github.com/microsoft/onnxruntime))

> Compared to its predecessor [NNTools](https://github.com/hqucms/NNTools/), `Weaver` simplifies the data processing pipeline by running all the pre-processing on-the-fly, without the necessity of creating intermediate transformed dataset (though it still supports that). The neural network training now uses the more widely adopted `PyTorch` instead of `Apache MXNet`.

<!-- TOC -->

- [Weaver](#weaver)
    - [Set up your environment](#set-up-your-environment)
        - [Install Miniconda (if you don't already have it)](#install-miniconda-if-you-dont-already-have-it)
        - [Set up a conda environment and install the required packages](#set-up-a-conda-environment-and-install-the-required-packages)
    - [Prepare your configuration files](#prepare-your-configuration-files)
        - [Data configuration file](#data-configuration-file)
        - [Model configuration file](#model-configuration-file)
    - [Start running!](#start-running)
        - [Training](#training)
        - [Prediction/Inference](#predictioninference)
        - [Model exportation](#model-exportation)
    - [More about data loading and processing](#more-about-data-loading-and-processing)
    - [Performance consideration](#performance-consideration)

<!-- /TOC -->

## Set up your environment

The `Weaver` package requires Python 3.7+ and a number of packages like `numpy`, `scikit-learn`, `PyTorch`, etc.
To run the neural network training, a Nvidia GPU with [CUDA](https://developer.nvidia.com/cuda-downloads) support is needed.

To manage the Python environment, we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Below are the instructions to set up Miniconda and install the required packages. This needs to be done only once.

### Install Miniconda (if you don't already have it)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the instructions to finish the installation

# Make sure to choose `yes` for the following one to let the installer initialize Miniconda3
# > Do you wish the installer to initialize Miniconda3
# > by running conda init? [yes|no]

# disable auto activation of the base environment
conda config --set auto_activate_base false
```

Verify the installation is successful by running `conda info` and check if the paths are pointing to your Miniconda installation.

If you cannot run `conda` command, check if you need to add the conda path to your `PATH` variable in your bashrc/zshrc file, e.g.,

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
```

### Set up a conda environment and install the required packages

```bash
# create a new conda environment
conda create -n weaver python=3.7

# activate the environment
conda activate weaver

# install the necessary python packages
pip install numpy pandas scikit-learn scipy matplotlib tqdm PyYAML

# install uproot for reading/writing ROOT files
pip install uproot lz4 xxhash

# install PyTables if using HDF5 files
pip install tables

# install onnxruntime if needs to run inference w/ ONNX models
pip install onnxruntime-gpu

# install pytorch, follow instructions for your OS/CUDA version at:
# https://pytorch.org/get-started
# pip install torch
```

## Prepare your configuration files

To train a neural network using `Weaver`, you need to prepare:
  - A YAML *data configuration file* describing how to process the input data.
  - A python *model configuration file* providing the neural network module and the loss function.
  
### Data configuration file

The data configuration file is a [YAML](https://yaml.org/) format file describing how to process the input data. It needs the following sections:
  - `selection` (optional): event selection
  - `new_variables` (optional): new variable definition
  - `inputs` (required): input groups, variables for each group, variable transformation (mean/scale/min/max for standardization, length/pad values for padding/clipping, etc.)
  - `labels` (required): label definition
  - `observers` (optional): additional variables that are not used in the training, but will be added to the output files when running prediction
  - `weights` (optional): instance weight information for sampling the instances during the training

An example of the data configuration file is [data/ak15_points_pf_sv.yaml](data/ak15_points_pf_sv.yaml).
For more details, check [utils/data/config.py](utils/data/config.py) and [utils/dataset.py](utils/dataset.py).

### Model configuration file

The model configuration file specifies the neural network model and the loss function. It needs to implement the `get_model` function (required) and the `get_loss` fucntion (optional, default is `torch.nn.CrossEntropyLoss()`) in the following signatures:

```python
def get_model(data_config, **kwargs):
    model = ... # instance of PyTorch Module
    model_info = {
        'input_names':...,
        'input_shapes':...,
        'output_names':...,
        'dynamic_axes':...,
        }
    return model, model_info

# `get_loss` is optional
# if not provided, fallback to use `torch.nn.CrossEntropyLoss()`
def get_loss(data_config, **kwargs):
    loss_func = ...
    return loss_func
```

An example of the model configuration file is [networks/particle_net_pf_sv.py](networks/particle_net_pf_sv.py).


## Start running!

The [train.py](train.py) script is the top-level script to run for training a neural net, getting prediction from trained models, and exporting trained models to ONNX for production.
To check all the command-line options for `train.py`, run `python train.py -h`. Examples for training, inference and model exportation are shown below:

### Training

```bash
python train.py --data-train '/path/to/train_files/*/*/*/*/output_*.root' \
 --data-config data/ak15_points_pf_sv.yaml \
 --network-config networks/particle_net_pf_sv.py \
 --model-prefix /path/to/models/prefix \
 --num-workers 3 --gpus 0,1,2,3 --batch-size 512 --start-lr 5e-3 --num-epochs 20 --optimizer ranger \
 | tee logs/train.log
```

Note: 
 - `--data-train` supports providing multiple entries, e.g., `--data-train /path/to/A /path/to/B /path/to/C`, and each entry also supports wildcards 
 (`*`, `?`, etc. -- The python `glob` package is used to parse the paths).
 - for training, `--model-prefix` sets the *prefix* part of the paths to save model snapshots. 
 At the end of each epoch, the model parameters will be saved to `/path/to/models/prefix_epoch-%d_state.pt`,
 and the optimizer states will be saved to `/path/to/models/prefix_epoch-%d_optimizer.pt` in case the training is interrupted and needed to be resumed from a certain epoch.

### Prediction/Inference

Once you have a trained model, you can load it to run prediction and test its performance, e.g.,

```bash
python train.py --predict --data-test '/path/to/test_files/*/*/*/*/output_*.root' \
 --data-config data/ak15_points_pf_sv_forTesting.yaml \ 
 --network-config networks/particle_net_pf_sv.py \
 --model-prefix /path/to/models/prefix_epoch-19_state.pt \
 --num-workers 3 --gpus 0,1,2,3 --batch-size 512 \
 --predict-output /path/to/output.root
```

Note: 
 - `--data-test` supports providing multiple entries, e.g., `--data-test /path/to/A /path/to/B /path/to/C`, and each entry also supports wildcards 
 (`*`, `?`, etc. -- The python `glob` package is used to parse the paths).
 - for inference, one can use a data configuration file with different `selection` / `observers` sections, but the `inputs` and `labels` sections must remain unchanged.
 - for inference, one can specify the full path of the model parameters in `--model-prefix`.
 - `--predict-output` sets the path for the output file. Currently support saving to ROOT files (use `.root` extension) or awkward arrays (use `.awkd` extension).

### Model exportation

When you are satisfied with the trained model, you could export it from PyTorch to ONNX format for inference (e.g., using [ONNXRuntime](https://github.com/microsoft/onnxruntime)):

```bash
python train.py -c data/ak15_points_pf_sv.yaml -n networks/particle_net_pf_sv.py -m /path/to/models/prefix_epoch-19_state.pt --export-onnx model.onnx  
```

## More about data loading and processing

To come...

Check also [utils/dataset.py](utils/dataset.py).


## Performance consideration

Loading data from disk can often become a bottleneck. Here are a few tips to get better data loading performance:

 - When using ROOT files as inputs, prepare the files w/ `LZ4` compression:

```C++
f.SetCompressionAlgorithm(ROOT::kLZ4); 
f.SetCompressionLevel(4);
```

and use larger buffer sizes for the TTree branches:
```C++
tree->Branch("x", &var, "x/F", /*bufsize=32000*/1024000);
```
 (e.g., here the buffer size is set to `1024000` instead of the default `32000`).
 - Copy files to faster disk (e.g., SSD) if possible.
 - Enable multiprocessing for data loading. Setting `--num-workers` to 2 or 3 generally gives a good performance. Setting this value too high could overload the disk and degrade the performance. 
   - Note that the memory usage also increases with the number of workers. So if you are getting any memory-related errors, try reducing `--num-workers`.
   - Note that the workload splitting is file-based, so make sure the number of input files is not too small (i.e., make sure each worker is able to load several files to get samples **from all classes**).

