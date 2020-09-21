import argparse
import torch
import torchvision
import torchvision.transforms as transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", help="Path to your dataset", type=str, default='')
    parser.add_argument("-batch_size", type=int, default=10)
    parser.add_argument("-not_caffe", action='store_true')
    parser.add_argument("-use_rgb", action='store_true')
    params = parser.parse_args()
    main_calc(params)


# Mean and standard deviation calculations based on code from here: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/4
def main_calc(params):
    transform_list = [transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor()]

    if not params.not_caffe:
        range_change = transforms.Compose([transforms.Lambda(lambda x: x*255)])
        transform_list += [range_change]
    if not params.use_rgb:
        rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        transform_list += [rgb2bgr]

    dataset = torchvision.datasets.ImageFolder(params.data_path, transform=transforms.Compose(transform_list))
    loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, num_workers=0, shuffle=False)

    print('Computing dataset mean (this may take a while)')
    mean = 0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    print('Computing dataset standard deviation (this may take a while)')
    sd = 0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        sd += ((images - mean.unsqueeze(1))**2).sum([0,2])
    sd = torch.sqrt(sd / (len(loader.dataset)*224*224))


    print('\n-data_mean ' + str(round(mean[0].item(), 4)) + ',' + str(round(mean[1].item(), 4)) + ',' + str(round(mean[2].item(), 4)) + \
          ' -data_sd ' + str(round(sd[0].item(), 4)) + ',' + str(round(sd[1].item(), 4)) +',' +  str(round(sd[2].item(), 4)))



if __name__ == "__main__":
    main()