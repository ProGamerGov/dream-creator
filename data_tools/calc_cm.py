import argparse
import torch
import torchvision
import torchvision.transforms as transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", help="Path to your dataset", type=str, default='')
    parser.add_argument("-batch_size", type=int, default=10)
    parser.add_argument("-use_rgb", action='store_true')
     
    parser.add_argument("-model_file", type=str, default='')
    parser.add_argument("-output_file", type=str, default='')
    params = parser.parse_args()
    main_calc(params)


def main_calc(params):
    transform_list = [transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor()]

    if not params.use_rgb:
        rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        transform_list += [rgb2bgr]

    dataset = torchvision.datasets.ImageFolder(params.data_path, transform=transforms.Compose(transform_list))
    loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, num_workers=0, shuffle=False)

    print('Computing dataset covariance matrix (this may take a while)') 
    cov_mtx = 0
    for images, _ in loader:
        for b in range(images.size(0)):   
            cov_mtx += rgb_cov(images[b].permute(1,2,0))

    cov_mtx = cov_mtx / len(loader.dataset)

    U,S,V = torch.svd(cov_mtx)
    epsilon = 1e-10
    svd_sqrt = U @ torch.diag(torch.sqrt(S + epsilon))

    print('Color correlation matrix\n')
    print(svd_sqrt)
    
    print_string = "color_decorrelation " + '"' + str(round(svd_sqrt[0][0].item(), 4)) + ',' + str(round(svd_sqrt[0][1].item(), 4)) + ',' + str(round(svd_sqrt[0][2].item(), 4)) \
                   + ',' + str(round(svd_sqrt[1][0].item(), 4)) + ',' + str(round(svd_sqrt[1][1].item(), 4))  + ',' + str(round(svd_sqrt[1][2].item(), 4)) \
                   + ',' + str(round(svd_sqrt[2][0].item(), 4)) + ',' + str(round(svd_sqrt[2][1].item(), 4)) + ',' + str(round(svd_sqrt[2][2].item(), 4)) + '"'
                   
    print("\n-" + print_string.replace('-', 'n'))

    if params.model_file != '':
        checkpoint = torch.load(params.model_file, map_location='cpu')
        checkpoint['color_correlation_svd_sqrt'] = svd_sqrt
        if params.output_file == '':
            params.output_file = params.model_file
        print('Saving color correlation matrix to ' + params.output_file)
        torch.save(checkpoint, params.output_file)


def rgb_cov(im):
    '''
    Assuming im a torch.Tensor of shape (H,W,3):
    '''
    im_re = im.reshape(-1, 3)
    im_re -= im_re.mean(0, keepdim=True)
    return 1/(im_re.shape[0]-1) * im_re.T @ im_re



if __name__ == "__main__":
    main()
