# dream-creator Installation

This guide will walk you through multiple ways to setup `dream-creator` on Ubuntu and Windows. If you wish to install PyTorch and dream-creator on a different operating system like MacOS, installation guides can be found [here](https://pytorch.org).

Note that in order to reduce their size, the pre-packaged binary releases (pip, Conda, etc...) have removed support for some older GPUs, and thus you will have to install from source in order to use these GPUs.


# Ubuntu:

## With A Package Manager:

The pip and Conda packages ship with CUDA and cuDNN already built in, so after you have installed PyTorch with pip or Conda, you can skip to [installing dream-creator](https://github.com/ProGamerGov/dream-creator/blob/master/INSTALL.md#install-dream-creator).


#### Github and pip:

Following the pip installation instructions
[here](http://pytorch.org), you can install PyTorch with the following commands:

```
# in a terminal, run the commands
cd ~/
pip install torch torchvision
```

Or:

```
cd ~/
pip3 install torch torchvision
```

Now continue on to [installing dream-creator](https://github.com/ProGamerGov/dream-creator/blob/master/INSTALL.md#install-dream-creator) to install dream-creator.

### Conda:

Following the Conda installation instructions
[here](http://pytorch.org), you can install PyTorch with the following command:

```
conda install pytorch torchvision -c pytorch
```

Now continue on to [installing dream-creator](https://github.com/ProGamerGov/dream-creator/blob/master/INSTALL.md#install-dream-creator) to install dream-creator.

## From Source:

### (Optional) Step 1: Install CUDA

If you have a [CUDA-capable GPU from NVIDIA](https://developer.nvidia.com/cuda-gpus) then you can
speed up `dream-creator` with CUDA.

Instructions for downloading and installing the latest CUDA version on all supported operating systems, can be found [here](https://developer.nvidia.com/cuda-downloads).


### (Optional) Step 2: Install cuDNN

cuDNN is a library from NVIDIA that efficiently implements many of the operations (like convolutions and pooling)
that are commonly used in deep learning.

After registering as a developer with NVIDIA, you can [download cuDNN here](https://developer.nvidia.com/cudnn). Make sure that you use the appropriate version of cuDNN for your version of CUDA.

Follow the download instructions on Nvidia's site to install cuDNN correctly.

Note that the cuDNN backend can only be used for GPU mode.

### (Optional) Steps 1-3: Install PyTorch with support for AMD GPUs using Radeon Open Compute Stack (ROCm)


It is recommended that if you wish to use PyTorch with an AMD GPU, you install it via the official ROCm dockerfile:
https://rocm.github.io/pytorch.html

- Supported AMD GPUs for the dockerfile are: Vega10 / gfx900 generation discrete graphics cards (Vega56, Vega64, or MI25).

PyTorch does not officially provide support for compilation on the host with AMD GPUs, but [a user guide posted here](https://github.com/ROCmSoftwarePlatform/pytorch/issues/337#issuecomment-467220107) apparently works well.

ROCm utilizes a CUDA porting tool called HIP, which automatically converts CUDA code into HIP code. HIP code can run on both AMD and Nvidia GPUs.


### Step 3: Install PyTorch

To install PyTorch [from source](https://github.com/pytorch/pytorch#from-source) on Ubuntu (Instructions may be different if you are using a different OS):

```
cd ~/
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python setup.py install

cd ~/
git clone --recursive https://github.com/pytorch/vision
cd vision
python setup.py install
```

To check that your torch installation is working, run the command `python` or `python3` to enter the Python interpreter. Then type `import torch` and hit enter.

You can then type `print(torch.version.cuda)` and `print(torch.backends.cudnn.version())` to confirm that you are using the desired versions of CUDA and cuDNN.

To quit just type `exit()` or use  Ctrl-D.

Now continue on to [installing dream-creator](https://github.com/ProGamerGov/dream-creator/blob/master/INSTALL.md#install-dream-creator) to install dream-creator.


# Windows Installation

If you wish to install PyTorch on Windows From Source or via Conda, you can find instructions on the PyTorch website: https://pytorch.org/


### Github and pip

First, you will need to download Python 3 and install it: https://www.python.org/downloads/windows/. I recommend using the executable installer for the latest version of Python 3.

Then using https://pytorch.org/, get the correct pip command, paste it into the Command Prompt (CMD) and hit enter:


```
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```


After installing PyTorch, download the dream-creator Github repository and extract/unzip it to the desired location.

Then copy the file path to your dream-creator folder, and paste it into the Command Prompt, with `cd` in front of it and then hit enter.

In the example below, the dream-creator folder was placed on the desktop:

```
cd C:\Users\<User_Name>\Desktop\dream-creator-master
```

You can now continue on to [installing dream-creator](https://github.com/ProGamerGov/dream-creator/blob/master/INSTALL.md#install-dream-creator), skipping the `git clone` step.


# Install dream-creator

First we clone `dream-creator` from GitHub:

```
cd ~/
git clone https://github.com/ProGamerGov/dream-creator.git
cd dream-creator
```

To test that dream-creator works, we'll first need to create a test dataset:

```
python data_tools/create_test_data.py -output_dir test_data -num_images 1200 -shape_size 200,200
```

Now we calculate the mean and standard deviation of our newly created test dataset:

```
python data_tools/calc_ms.py -data_path test_data
```

The `calc_ms.py` script will then output something that looks like this:

```
-data_mean 126.7094,126.0554,126.9935 -data_sd 73.655,73.3885,73.7058
```

Then using the mean and standard deviation parameter inputs we generated above, we can now train a GoogleNet model using the test dataset:

```
python train_googlenet.py -data_path test_data -balance_classes -batch_size 96 -num_epochs 10 -data_mean 126.7094,126.0554,126.9935 -data_sd 73.655,73.3885,73.7058
```

If everything is working properly with the training script, then you should see output like this:

```
Total 2400 images, split into 2 classes
Classes:
 {'ellipse': 0, 'rectangle': 1}
Model has 10,021,318 learnable parameters

Epoch 1/10
------------
train Loss: 0.9526 Acc: 0.8250
  Time Elapsed 0m 23s
val Loss: 0.0319 Acc: 0.9896
  Time Elapsed 0m 27s

Epoch 2/10
------------
train Loss: 0.3052 Acc: 0.9260
  Time Elapsed 0m 49s
val Loss: 0.0699 Acc: 0.9646
  Time Elapsed 0m 52s

Epoch 3/10
------------
train Loss: 0.2157 Acc: 0.9417
  Time Elapsed 1m 15s
val Loss: 0.0142 Acc: 0.9938
  Time Elapsed 1m 19s

Epoch 4/10
------------
train Loss: 1.6389 Acc: 0.5719
  Time Elapsed 1m 42s
val Loss: 0.6930 Acc: 0.5125
  Time Elapsed 1m 45s

Epoch 5/10
------------
train Loss: 1.3215 Acc: 0.5089
  Time Elapsed 2m 8s
val Loss: 0.6986 Acc: 0.5125
  Time Elapsed 2m 12s
```

After training has finished, we can now visualize the newly created DeepDream model's FC layers using the following command:

```
python vis_fc.py -model_file bvlc_out010.pth -model_epoch 10 -num_iterations 200
```

The `vis_fc.py` script should end up creating two output images, where one image has more circlelike features and the other has more squarelike features. Using more complex datasets that have more classes and images will yield far better looking results. You can find a list of image collection tools, possible sources of images, and duplicate image detection tools on the [dream-creator wiki](https://github.com/ProGamerGov/dream-creator/wiki).

Finally, to visualize a single layer and channel or to DeepDream your own image, we can use the following command:

```
# Random Noise content image
python vis.py -model_file bvlc_out010.pth -layer mixed5a -channel 7 -num_iterations 200

# With Content image
python vis.py -model_file bvlc_out010.pth -layer mixed5a -channel 7 -content_image examples/small/fc_flowers.jpg -image_size 512,512 -num_iterations 200
```