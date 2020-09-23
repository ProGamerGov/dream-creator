# Dataset, Model & Additional Visualization Tools

The following scripts have been written to aid in the process of creating datasets, extracting data required for training, editing models, and comparison/analysis of training results.

All of these scripts with the exception of `sort_images.py` can be copied to and used anywhere as they do not require any of the other scripts used by Dream-Creator. If you leave the scripts where they are, then you'll have to provide a path to them like for example `data_tools/` before the name of the script if you wish to use them.


1. [Dataset Tools](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#dataset-tools)

   1. [Dataset Mean and Standard Deviation Calculation](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#dataset-mean-and-standard-deviation-calculation)

   2. [FC Channel Contents](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#fc-channel-contents)

   3. [Dataset Image Resizing](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#dataset-image-resizing)

   4. [Corrupt Image Detector & Remover](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#corrupt-image-detector--remover)

   5. [Image Extractor](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#image-extractor)

   6. [Automatic Image Sorter](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#automatic-image-sorter)

2. [Model Tools]()

   1. [Reduce Model Size](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#reduce-model-size)

   2. [Add/Change Model Values](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#addchange-model-values)

3. [Visualization & Training Tools]()

   1. [Graph Training Data](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#graph-training-data)

   2. [Image Grid Creator](https://github.com/ProGamerGov/dream-creator/tree/master/data_tools#image-grid-creator)

---

# Dataset Tools

## Dataset Mean and Standard Deviation Calculation

Basic calculation for GoogleNet training (input range 0-255)

```
python calc_ms.py -data_path <training_data>
```

* `-not_caffe`: Enabling this flag will result in the mean and standard deviation output having a range of 0-1 instead of 0-255.

* `-use_rgb`: Enabling this flag will result in output values being in RGB format instead of BGR.


## Dataset RGB Covariance Matrix Calculation

This script calculates the RGB covariance matrix required for color decorrelation.

```
python calc_cm.py -data_path <training_data>
```

* `-use_rgb`: Enabling this flag will result in output values being in RGB format instead of BGR.
* `-model_file`: Optionally provide a path to the model that you wish to add the color matrix to.
* `-output_file`: Optionally provide a name for the output model. If left blank, then `-model_file` will be used instead.

## FC Channel Contents

This script will print which categories correspond to which FC layer layer channels. This is useful for when you are confused as to which channels correspond to which categories/classes.

```
python print_fc.py -data_path <training_data>
```

* `-print_string`: Enabling this flag will print class names as a comma separated string for use as an input to other scripts.


## Dataset Image Resizing

This script will make sure that every image in your dataset does not go above a specified max height/width. Resizing the images in your dataset to be closer to the training image size will help speed up training significantly.

```
python resize_data.py -data_path <training_data>
```

* `-max_size`: The maximum height/width of images in your dataset. Default is `500`.


## Corrupt Image Detector & Remover

This script will try to automatically detect corrupt images that interfere with the loading of images into PyTorch.

```
python find_bad.py -data_path <training_data>
```

* `-delete_bad`: Enabling this flag will result in corrupt images being deleted automatically from the specified dataset.


## Image Extractor

This tool takes a directory/folder full of sub folders with images, renames them, and then copies every valid image file to a new directory/folder. This tool can be useful for dealing with the outputs of data collection tools.

```
python extract.py -data_path <training_data>
```

* `-remove_names`: An optional comma separated list of image names that should be ignored.
* `-ignore_dirs`: An optional comma separated list of directories/folders to ignore.
* `-ignore_presets`: Optional preset list of images to ignore; one of `none` or `google`. Default is `none`.
* `-output_dir`: The name of the new directory/folder to use for all renamed images. Default is `renamed`.
* `-output_name`: The new basename to use for all images copied to the new directory/folder. Default is `image`.


## Automatic Image Sorter

This script utilizes your pretrained GoogleNet model to sort a folder of images into separate categories/classes.

```
python sort_images.py -unsorted_data <raw_data> -sorted_data <new_training_data> -model_file <bvlc_out120>.pth
```

* `-unsorted_data`: The name of a folder/directory with images that you wish to sort.
* `-sorted_data`: Where to put the sorted images.
* `-model_file`: Path to your pretrained GoogleNet model file.
* `-confidence_min`: The minimum confidence percentage for sorting images. Images with a lower confidence value will not be sorted.
* `-confidence_max`: The maximum confidence percentage for sorting images. Images with a higher confidence value will not be sorted.
* `-class_strings`: This parameter takes a comma separated list of class names to be used for directory/folder class names. The `print_fc.py` script's `-print_string` flag will produce a comma separated list that can be used for this parameter.
* `-data_mean`: Your precalculated list of mean values that was used to train the model, if they weren't saved inside the model.
* `-data_sd`: Your precalculated list of standard deviation values that was used to train the model, if they weren't saved inside the model.
* `-num_classes`: The number of classes that the model was trained on, if it wasn't saved inside the model.

---

# Model Editing

## Reduce Model Size

By default the training scripts save models with extra information like optimizer and learning rate scheduler states. This script will remove this extra information from the model as it is not required for DeepDream, and in doing so the resulting model file size can end up being less than half of the original model's file size.

```
python strip_model.py -model_file <bvlc_out120>.pth -output_name stripped_models.pth
```

* `-delete_branches`: If this flag is enabled, any auxiliary branches in the model will be removed.


## Add/Change Model Values

If need to add or change any of the stored model values then use this script. Any options left as `ignore` or `-1` will not be added/changed. This script can be useful for fixing bugs, adding new models, and adding missing values.

```
python edit_model.py -model_file <bvlc_out120>.pth -base_model bvlc -num_classes 10 -output_name edited_model.pth
```

* `-model_file`: Path to your pretrained GoogleNet model file.
* `-output_name`: Name of the output model. If left blank, no output model will be saved.
* `-data_mean`: Your precalculated list of mean values that was used to train the model. Default is `ignore`.
* `-data_sd`: Your precalculated list of standard deviation values that was used to train the model. Default is `ignore`.
* `-normval_format`: The format of your mean and standard deviation values; one of `bgr`, `rgb`, `ignore`. Default is `ignore`.
* `-has_branches`: Whether or not the model has branches; one of `true`, `false`, `ignore`. Default is `ignore`.
* `-base_model`: The base model used to create your model; one of `bvlc`, `p365`, `5h`, `ignore`. Default is `ignore`.
* `-num_classes`: Set the number of model classes. Default is set to `-1` to ignore.
* `-model_epoch`: Set the model epoch. Default is set to `-1` to ignore.
* `-reverse_normvals`: If this flag is enabled, mean and standard deviation values added to the model and stored in the model will be reversed. In essence BGR values are converted to RGB and vice versa.
* `-print_vals`: If this flag is enabled, all stored model values from the loaded model will be printed.

---

# Comparison of Results

## Graph Training Data

This script creates a line graph from the CSV log file produced by the `train_googlenet.py` script when the `-save_csv` flag is enabled. It's recommended that you also use the  `train_googlenet.py` script's `-individual_acc` flag when saving CSV data.

```
python resize_data.py -csv_file train_acc.txt
```

* `-csv_file`: The name of your csv that was produced when running `train_googlenet.py`.
* `-graph_title`: The name of your graph. Default is `Accuracy Values`.
* `-x_name`: The name of your graph's X axis. Default is `Epoch`.
* `-y_name`: The name of your graph's Y axis. Default is `Accuracy`.
* `-class_strings`: This parameter takes a comma separated list of class names to be used for category names on the graph. The `print_fc.py` script's `-print_string` flag will produce a comma separated list that can be used for this parameter.


## Image Grid Creator

This script will put images created by `vis_multi.py` into a grid for easy comparisons and analysis

```
python make_grid.py -input_path <image_dir>
```

* `-input_path`: Either a comma separated list of images, or a directory with the desired images.
* `-base_name`: This tells the script what basename the target images are using. Default is set to None to disable filtering based on name.
* `-output_image`: The name of the output image. Default is `out.jpg`.
* `-pattern`: Two numbers separated by a comma that delineate the intended grid pattern.
* `-border`: The size of the border used to separate images in the grid. Default is `2`.
* `-epoch`: Optionally only select images from a specific epoch. Default is set to `-1` to disable it.
* `-channel`: Optionally only select images from a specific channel. Default is set to `-1` to disable it.
* `-disable_natsort`: Enabling this flag will disable natural sorting of image names.
