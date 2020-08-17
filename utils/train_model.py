import time
import torch
from utils.training_utils import save_model, count_classes, save_csv_data


# Training function adapted from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def train_model(model, dataloaders, criterion, optimizer=None, lrscheduler=None, num_epochs=120, start_epoch=1, save_epoch=1, \
                output_name='out.pth', device='cuda:0', has_branches=False, fc_only=False, num_classes=0, individual_acc = False, \
                should_save_csv=False, save_info=None):

    if individual_acc:
        data_counts = {}
        data_counts['train'] = count_classes(dataloaders['train'].dataset)
        data_counts['val'] = count_classes(dataloaders['val'].dataset)

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs + start_epoch):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 12)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            epoch_loss, epoch_acc = 0, 0
            if individual_acc:
                class_acc = [0] * num_classes

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if phase == 'train':
                        if has_branches:
                            loss1 = criterion(outputs[0], labels)
                            loss2 = criterion(outputs[1], labels)
                            loss3 = criterion(outputs[2], labels)
                            loss = loss1 + (0.3 * loss2) + (0.3 * loss3)
                            _, preds = torch.max(outputs[0], 1)
                        else:
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                    else:
                        if has_branches:
                            loss = criterion(outputs[0], labels)
                            _, preds = torch.max(outputs[0], 1)
                        else:
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                with torch.set_grad_enabled(False):
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_acc += torch.sum(preds == labels.data).detach()
                    if individual_acc:
                        class_acc_test = torch.where(preds == labels.data, preds.detach(), torch.zeros_like(preds) - 1)
                        for c_val in range(num_classes):
                            class_acc[c_val] += torch.sum(class_acc_test == torch.ones_like(preds) * c_val)

            epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
            epoch_acc = epoch_acc.double() / len(dataloaders[phase].dataset)

            if individual_acc:
                for c_val in range(num_classes):
                    class_acc[c_val] = (class_acc[c_val].item() / data_counts[phase][c_val]) * 100
                    class_acc[c_val] = round(class_acc[c_val], 2)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if individual_acc:
                print('  Class Acc:', {i:j for i,j in enumerate(class_acc)})
                if should_save_csv:
                    save_csv_data(phase + '_acc.txt', [str(epoch)] + class_acc)

            time_elapsed = time.time() - start_time
            print('  Time Elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            if epoch % save_epoch == 0 or epoch == num_epochs:
                save_model(model, optimizer=optimizer, lrscheduler=lrscheduler, epoch=epoch, output_name=output_name, \
                           fc_only=fc_only, save_info=save_info)

        if lrscheduler != None:
            lrscheduler.step()
        print()