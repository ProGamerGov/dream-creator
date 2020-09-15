import os
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from csv import writer as csv_writer, QUOTE_ALL
from collections import Counter
from utils.inceptionv1_caffe import InceptionV1_Caffe



# Calculate the size of training and validation datasets
def split_percent(n, pctg=0.2):
    return [round(n * (1 - pctg)), round(n * pctg)]


# Create training and validation images from single set of images
def load_dataset(data_path='test_data', val_percent=0.2, batch_size=1, input_size=(224,224), \
                       input_mean=[0.485, 0.456, 0.406], input_sd=[0.229, 0.224, 0.225], use_caffe=False, \
                       train_workers=25, val_workers=5, balance_weights=False, rnd_generator=None):

    num_classes = sum(os.path.isdir(os.path.join(data_path, i)) for i in os.listdir(data_path))

    train_transform_list = [
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=input_mean, std=input_sd),
    ]
    val_transform_list = [
        transforms.Resize(input_size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=input_mean, std=input_sd),
    ]

    if use_caffe:
        range_change = transforms.Compose([transforms.Lambda(lambda x: x*255)])
        rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        train_transform_list = train_transform_list[:-1] + [range_change] + [rgb2bgr] + [train_transform_list[-1]]
        val_transform_list = val_transform_list[:-1] + [range_change] + [rgb2bgr] + [val_transform_list[-1]]

    # Load all images
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
    )

    print('\nTotal ' + str(len(full_dataset)) + ' images, split into ' + str(num_classes) + ' classes')
    get_fc_channel_classes(data_path)
    if val_percent > 0:
        lengths = split_percent(len(full_dataset), val_percent)
        if rnd_generator == None:
            t_data, v_data = torch.utils.data.random_split(full_dataset, lengths)
        else:
            t_data, v_data = torch.utils.data.random_split(full_dataset, lengths, generator=rnd_generator)
    else:
        t_data, v_data = torch.utils.data.Subset(copy.deepcopy(full_dataset), range(0, len(full_dataset))), \
                         torch.utils.data.Subset(copy.deepcopy(full_dataset), range(0, len(full_dataset)))

    # Use separate transforms for training and validation data
    t_data = copy.deepcopy(t_data)
    t_data.dataset.transform = transforms.Compose(train_transform_list)
    v_data.dataset.transform = transforms.Compose(val_transform_list)

    train_loader = torch.utils.data.DataLoader(
        t_data,
        batch_size=batch_size,
        num_workers=train_workers,
        shuffle=True,
        generator=rnd_generator,
    )
    val_loader = torch.utils.data.DataLoader(
        v_data,
        batch_size=batch_size,
        num_workers=val_workers,
        shuffle=True,
        generator=rnd_generator,
    )

    if balance_weights:
        train_class_counts = count_classes(train_loader.dataset)
        train_weights = [1 / train_class_counts[class_id] for class_id in range(num_classes)]
        train_weights = torch.FloatTensor(train_weights)
    else:
        train_weights = None
    return {'train': train_loader, 'val': val_loader}, num_classes, train_weights


# Get the number of images in each class in a dataset
def count_classes(dataset):
    class_counts = dict(Counter(sample_tup[1] for sample_tup in dataset))
    return dict(sorted(class_counts.items()))


# Save model without extra training info
def save_model_simple(model, name):
    model = copy.deepcopy(model)
    torch.save(model.state_dict(), name)


# Save model with info needed to continue later
def save_model(model, optimizer=None, lrscheduler=None, loss=0, epoch=0, output_name='test.pth', fc_only=False, save_info=None):
    output_filename, file_extension = os.path.splitext(output_name)
    save_name = output_filename + str(epoch).zfill(3) + file_extension

    model = copy.deepcopy(model)
    if fc_only:
        model = model.fc
    save_model = {'model_state_dict': model.state_dict()}
    if epoch != 0:
        save_model['epoch'] = epoch
    if optimizer != None:
        save_model['optimizer_state_dict'] = optimizer.state_dict()
    if lrscheduler != None:
        save_model['lrscheduler_state_dict'] = lrscheduler.state_dict()
    if save_info != None:
        save_model['normalize_params'] = save_info[0]
        save_model['num_classes'] = save_info[1]
        save_model['has_branches'] = save_info[2]
        save_model['base_model'] = save_info[3]
    torch.save(save_model, save_name)


# Reset model weights
def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)


# Set global seed
def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    random.seed(seed)


# Get classnames and order used by PyTorch
def get_fc_channel_classes(data_path):
    classes = [d.name for d in os.scandir(data_path) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print('Classes:')
    print('', class_to_idx)


# Save list as CSV
def save_csv_data(csv_path, filename, new_line):
    with open(os.path.join(csv_path, filename), "a+", newline='') as f:
        wr = csv_writer(f, quoting=QUOTE_ALL)
        wr.writerow(new_line)


# Load model class and optionally reset weights
def setup_model(model_file='pt_bvlc.pth', num_classes=120, base_model='bvlc', pretrained=False):
    base_list = {'pt_bvlc.pth': (1000, 'bvlc'), 'pt_places365.pth': (365, 'p365'), 'pt_inception5h.pth': (1008, '5h')}
    base_name, has_branches = os.path.basename(model_file), True
    if base_name.lower() in base_list:
        load_classes, mode = base_list[base_name.lower()]
        is_start_model = True
        if mode == '5h':
            has_branches = False
    else:
        load_classes = num_classes
        is_start_model = False
        try:
            mode = torch.load(model_file, map_location='cpu')['base_model']
        except:
            mode = base_model
        try:
            has_branches = torch.load(model_file, map_location='cpu')['has_branches']
        except:
            pass

    cnn = InceptionV1_Caffe(load_classes, mode=mode, load_branches=has_branches)

    if not pretrained:
        cnn.apply(reset_weights)
    return cnn, is_start_model, mode


# Load checkpoint
def load_checkpoint(cnn, model_file, optimizer, lrscheduler, num_classes, device='cuda:0', is_start_model=True):
    start_epoch, change_fc = 1, False

    checkpoint = torch.load(model_file, map_location='cpu')
    if type(checkpoint) == dict:
        model_keys = list(checkpoint.keys())

        try:
            load_classes = checkpoint['num_classes']
            if num_classes != load_classes:
                is_start_model, change_fc = True, True # Pretend it's a starting model so FC gets replaced
            else:
                load_classes = num_classes
        except:
            load_classes = num_classes

        if not is_start_model or change_fc:
            cnn.replace_fc(load_classes, True)

        cnn.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in model_keys:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        if 'lrscheduler_state_dict' in model_keys:
            lrscheduler.load_state_dict(checkpoint['lrscheduler_state_dict'])
        if 'epoch' in model_keys:
            start_epoch = checkpoint['epoch']
    else:
        cnn.load_state_dict(checkpoint)

    # Setup model for new data set
    if is_start_model:
        cnn.replace_fc(num_classes, True)
    return cnn, optimizer, lrscheduler, start_epoch
