# Dream-Creator

This project aims to simplify the process of creating a custom DeepDream model by using pretrained GoogleNet models and custom image datasets.

Here are some example visualizations created with custom DeepDream models trained on summer themed images:

<div align="center">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/big/fc_beachchair.jpg" height="400px">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/big/fc_icecream.jpg" height="400px">

<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/big/fc_waterslides.jpg" height="400px">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/big/fc_windsurfing.jpg" height="400px">

<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/big/fc_shortshorts.jpg" height="400px">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/big/fc_volleyball.jpg" height="400px">
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/small/fc_flowers.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/small/fc_hotairballoon.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/small/fc_jetski.jpg" height="250px">

<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/small/fc_sunglasses.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/small/fc_surboard.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/small/fc_tent.jpg" height="250px">

<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/small/fc_bike.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/small/fc_bikini.jpg" height="250px">
<img src="https://raw.githubusercontent.com/ProGamerGov/dream-creator/master/examples/small/fc_volleyball.jpg" height="250px">
</div>

---

## Setup:

Dependencies:
* [PyTorch](http://pytorch.org/)

You can find detailed installation instructions for Ubuntu and Windows in the [installation guide](INSTALL.md).

After making sure that PyTorch is installed, you can optionally download the Places365 GoogleNet and Inception5h (InceptionV1) pretrained models with the following command:

```
python models/download_models.py
```

If you just want to create DeepDreams with the pretrained models or you downloaded a pretrained model made by someone else with Dream-Creator, then you can skip ahead to [visualizing models](https://github.com/ProGamerGov/dream-creator#visualizing-results).


# Getting Started

1. [Create & Prepare Your Dataset](https://github.com/ProGamerGov/dream-creator#dataset-creation)

   1. Collect Images

   2. Sort images into the required format.

   3. Remove any corrupt images.

   4. Ensure that any duplicates are removed if you have not done so already

   5. Resize the dataset to speed up training.

   6. Calculate the mean and standard deviation of your dataset.

2. [Train a GoogleNet model](https://github.com/ProGamerGov/dream-creator#googlenet-training)

3. [Visualize the results](https://github.com/ProGamerGov/dream-creator#visualizing-results)

4. If the results are not great, then you may have to go back to step 1-2 and make some changes with what images, categories, and training parameters are used.

It can take as little as 5 epochs to create visualizations that resemble your training data using the main FC/Logits layer. In order to speed up training and create better looking results, the pretrained BVLC model used is partially frozen in order to protect the lower layers from changing.


---


## Dataset Creation

In order to train a custom DeepDream model, you will need to create a dataset composed of images that you wish to use for training. There are a variety of ways that you can aquire images for your dataset, and you will need at least a couple hundred images for each category/class.

DeepDream is most often performed with image classification models trained on image datasets that are composed of different categories/classes. Image classification models attempt to learn the difference between different image classes and in doing so the neurons gain the ability to create dream-like hallucinations. The images you choose, the differences between them, the differences between your chosen classes, and the number of images used will greatly affect the visualizations that can be produced.

PyTorch image datasets are to be structured where the main directory/folder contains subfolders/directories for each category/class. Below an example of the required dataset structure is shown:

```
dataset_dir
│
└───category1
│   │   image1.jpg
│   │   image2.jpg
│   │   image3.jpg
│
└───category2
    │   image1.jpg
    │   image2.jpg
    │   image3.jpg
```

Once you have created your dataset in the proper format, make sure that you remove any duplicate images if you have not done so already. There are a variety of tools that you can use for this task, including free and open source software.

If you have not done so already, you may wish to create a backup copy of your dataset.

Next you will need to verify that none of the images are corrupt in such a way that prevents PyTorch from loading them. To automatically remove any corrupt images from your dataset, use the following command:

```
python data_tools/remove_bad.py -delete_bad -data_path <training_data>
```

Next you will likely want to resize your dataset to be closer to the training image size in order to speed up training. Resizing your dataset will not prevent you from creating larger DeepDream images with the resulting model. The included resizing script will only modify images that go above the specified image size with their height or width.

To resize the images in your dataset, use the following command:

```
python data_tools/resize_data.py -data_path <training_data> -max_size 500
```

Now with your newly resized dataset, you can calculate the mean and standard deviation of your dataset for use in training, and DeepDreaming. Make sure to recalculate the mean and standard deviation again if you modify the dataset by adding or removing images.

To calculate the mean and standard deviation of your dataset, use the following command and save the output for the next step:

```
python data_tools/calc_ms.py -data_path <training_data>
```

Now you can start training your DeepDream model by running the GoogleNet training script. It's recommended that you save the model every 5-10 epochs in order to monitor the quality of the visualizations.


---


## GoogleNet Training

Basic training command:

```
python train_googlenet.py -data_path <training_data> -balance_classes -batch_size 96 -data_mean <mean> -data_sd <sd>
```

**Input options:**
* `-data_path`: Path to the dataset directory/folder that you wish to use.
* `-data_mean`: Your precalculated list of mean values for your chosen dataset.
* `-data_sd`: Your precalculated list of standard deviation values for your chosen dataset.

**Training options:**
* `-num_epochs`: The number of training epochs to use. Default is `120`.
* `-batch_size`: The number of training and validation images to put through the network at the same time. Default is `32`.
* `-learning_rate`: Learning rate to use with the ADAM or SGD optimizer. Default is `1e-2`.
* `-optimizer`: The optimization algorithm to use; either `sgd` or `adam`; default is `sgd`.
* `-train_workers`: How many workers to use for training. Default is `0`.
* `-val_workers`: How many workers to use for validation. Default is `0`.
* `-balance_classes`: Enabling this flag will balance training for each class based on class size.

**Model options:**
* `-model_file`: Path to the `.pth` model file to use for the starting model. Default is the BVLC GoogleNet model.
* `-freeze_to`: Which layer to freeze the model up to; one of `none`, `conv1`, `conv2`, `conv3`, `mixed3a`, `mixed3b`, `mixed4a`, `mixed4b`, `mixed4c`, `mixed4d`, `mixed4e`, `mixed5a`, or `mixed5b`. Default is `mixed3b`.
* `-freeze_aux1_to`: Which layer to freeze the first auxiliary branch up to; one of `none`, `loss_conv`, `loss_fc`, or `loss_classifier`. Default is `none`.
* `-freeze_aux2_to`: Which layer to freeze the second auxiliary branch up to; one of `none`, `loss_conv`, `loss_fc`, or `loss_classifier`. Default is `none`.
* `-delete_branches`: If this flag is enabled, no auxiliary branches will be used in the model.

**Output options:**
* `-save_epoch`: Save the model every `save_epoch` epochs.  Default is `10`. Set to `0` to disable saving intermediate models.
* `-output_name`: Name of the output model. Default is `bvlc_out.pth`.
* `-individual_acc`: Enabling this flag will print the individual accuracy of each class.
* `-save_csv`: Enabling this flag will save loss and accuracy data to txt files.
* `-csv_dir`: Where to save csv files. Default is set to current working directory.

**Other options:**
* `-use_device`: Zero-indexed ID of the GPU to use plus `cuda:`. Default is `cuda:0`.
* `-seed`: An integer value that you can specify for repeatable results. By default this value is random for each run.

**Dataset options:**
* `-val_percent`: The percentage of images from each class to use for validation. Default is `0.2`.

---

# Visualizing Results

## Visualizing GoogleNet FC Layer Results

After training a new DeepDream model, you'll need to test it's visualizations. The best visualizations are found in the main FC layer also known as the 'logits' layer. This script helps you quickly and easily visualize all of a specified layer's channels in a particular model for a particular model epoch, by generating a separate image for each channel.

**Input options:**
* `-model_file`: Path to the pretrained GoogleNet model that you wish to use.
* `-learning_rate`: Learning rate to use with the ADAM or L-BFGS optimizer. Default is `1.5`.
* `-optimizer`: The optimization algorithm to use; either `lbfgs` or `adam`; default is `adam`.
* `-num_iterations`: Default is `500`.
* `-layer`: The specific layer you wish to use. Default is set to `fc`.
* `-extract_neuron`: If this flag is enabled, the center neuron will be extracted from each channel.
* `-image_size`: A comma separated list of `<height>,<width>` to use for the output image. Default is set to `224,224`.
* `-jitter`: The amount of image jitter to use for preprocessing. Default is `32`.
* `-fft_decorrelation`: Whether or not to use FFT spatial decorrelation. If enabled, a lower learning rate should be used.
* `-color_decorrelation`: Whether or not to use color decorrelation. Optionally provide a comma separated list of values for the color correlation matrix.
* `-random_scale`: Whether or not to use random scaling. Optionally provide a comma separated list of values for scales to be randomly selected from.
* `-random_rotation`: Whether or not to use random rotations. Optionally provide a comma separated list of values for rotations to be randomly selected from.

**Processing options:**
* `-batch_size`: How many channel visualization images to create in each batch. Default is `10`.
* `-start_channel`: What channel to start creating visualization images at. Default is `0`.
* `-end_channel`: What channel to stop creating visualization images at. Default is set to `-1` for all channels.

**Only Required If Model Doesn't Contain Them, Options**:
* `-model_epoch`: The training epoch that the model was saved from, to use for the output image names. Default is `120`.
* `-data_mean`: Your precalculated list of mean values that was used to train the model, if they weren't saved inside the model.
* `-num_classes`: The number of classes that the model was trained on. Default is `120`.

**Output options**:
* `-output_dir`: Where to save output images. Default is set to current working directory.
* `-print_iter`: Print progress every `print_iter` iterations. Set to `0` to disable printing.
* `-save_iter`: Save the images every `save_iter` iterations. Default is to `0` to disable saving intermediate results.

**Other options:**
* `-use_device`: Zero-indexed ID of the GPU to use plus `cuda:`. Default is `cuda:0`.
* `-seed`: An integer value that you can specify for repeatable results. By default this value is random for each run.

Basic FC (logits) layer visualization:

```
python vis_multi.py -model_file <bvlc_out120>.pth
```

---

## Performing DeepDream With Your Newly Trained Model

This script lets you create DeepDream hallucinations with trained GoogleNet models.

**Input options:**
* `-model_file`: Path to the pretrained GoogleNet model that you wish to use.
* `-learning_rate`: Learning rate to use with the ADAM or L-BFGS optimizer. Default is `1.5`.
* `-optimizer`: The optimization algorithm to use; either `lbfgs` or `adam`; default is `adam`.
* `-num_iterations`: Default is `500`.
* `-content_image`: Path to your input image. If no input image is specified, random noise is used instead.
* `-layer`: The specific layer you wish to use. Default is set to `mixed5a`.
* `-channel`: The specific layer channel you wish to use. Default is set to `-1` to disable specific channel selection.
* `-extract_neuron`: If this flag is enabled, the center neuron will be extracted from the channel selected by the `-channel` parameter.
* `-image_size`: A comma separated list of `<height>,<width>` to use for the output image. Default is set to `224,224`.
* `-jitter`: The amount of image jitter to use for preprocessing. Default is `32`.
* `-fft_decorrelation`: Whether or not to use FFT spatial decorrelation. If enabled, a lower learning rate should be used.
* `-color_decorrelation`: Whether or not to use color decorrelation. Optionally provide a comma separated list of values for the color correlation matrix.
* `-random_scale`: Whether or not to use random scaling. Optionally provide a comma separated list of values for scales to be randomly selected from.
* `-random_rotation`: Whether or not to use random rotations. Optionally provide a comma separated list of values for rotations to be randomly selected from.

**Only Required If Model Doesn't Contain Them, Options**:
* `-data_mean`: Your precalculated list of mean values that was used to train the model, if they weren't saved inside the model.
* `-num_classes`: The number of classes that the model was trained on, if it wasn't saved inside the model.

**Output options**:
* `-output_image`: Name of the output image. Default is `out.png`.
* `-print_iter`: Print progress every `print_iter` iterations. Set to `0` to disable printing.
* `-save_iter`: Save the images every `save_iter` iterations. Default is to `0` to disable saving intermediate results.

**Other options:**
* `-use_device`: Zero-indexed ID of the GPU to use plus `cuda:`. Default is `cuda:0`.
* `-seed`: An integer value that you can specify for repeatable results. By default this value is random for each run.


Basic DeepDream:

```
python vis.py -model_file <bvlc_out120>.pth -layer mixed5a
```

---

### Dataset Cleaning + Building & Visualization Tools

See [here](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools) for more information on all the included scripts/tools relating to dataset creation, cleaning, and preparation.
