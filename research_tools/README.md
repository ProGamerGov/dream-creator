# Research Tools

The following scripts have been written to aid in the process understanding models.


## Visualizing Polysemantic Channels

Some channels are polysemantic, meaning they contain multiple different features. This script attempts to separate these features into separate images for easier examintion.

**Input options:**
* `-model_file`: Path to the pretrained GoogleNet model that you wish to use.
* `-learning_rate`: Learning rate to use with the ADAM or L-BFGS optimizer. Default is `1.5`.
* `-optimizer`: The optimization algorithm to use; either `lbfgs` or `adam`; default is `adam`.
* `-num_iterations`: Default is `500`.
* `-layer`: The specific layer you wish to use. Default is set to `fc`.
* `-channel`: The specific layer channel you wish to use. Default is `0`.
* `-extract_neuron`: If this flag is enabled, the center neuron will be extracted from the channel selected by the `-channel` parameter.
* `-image_size`: A comma separated list of `<height>,<width>` to use for the output images. Default is set to `224,224`.
* `-jitter`: The amount of image jitter to use for preprocessing. Default is `32`.
* `-fft_decorrelation`: Whether or not to use FFT spatial decorrelation. If enabled, a lower learning rate should be used.

**Processing options:**
* `-batch_size`: How many features to attempt to extract from the selected channel. Default is `4`.

**Only Required If Model Doesn't Contain Them, Options**:
* `-model_epoch`: The training epoch that the model was saved from, to use for the output image names. Default is `120`.
* `-data_mean`: Your precalculated list of mean values that was used to train the model, if they weren't saved inside the model.
* `-num_classes`: The number of classes that the model was trained on. Default is `120`.

**Output options**:
* `-output_dir`: Where to save output images. Default is set to current working directory.
* `-print_iter`: Print progress every `print_iter` iterations. Set to `0` to disable printing.

**Other options:**
* `-use_device`: Zero-indexed ID of the GPU to use plus `cuda:`. Default is `cuda:0`.
* `-seed`: An integer value that you can specify for repeatable results. By default this value is random for each run.

Basic usage:

```
python vis_diverse.py -model_file <bvlc_out120>.pth -layer mixed5a/conv_5x5_relu -channel 5
```



## Detecting Activation Strength

This script lets you see what the strongest, weakest, or most average channels are for specified layers.

Basic usage:

```
python vis_activ.py -model_file <bvlc_out120>.pth -layer mixed5a/conv_5x5_relu -channels 5 -content_image <test_image>.jpg
```
