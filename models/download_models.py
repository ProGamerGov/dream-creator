from os import path
from sys import version_info
if version_info[0] < 3:
    import urllib
else:
    import urllib.request
import argparse



def main():
    params = params_list()
    if params.models == 'all' or params.models == 'places365':
        print("Downloading the Places365 model")
        fileurl = "https://github.com/ProGamerGov/pytorch-places/raw/restructured-models/pt_places365.pth"
        name = "pt_places365.pth"
        download_file(fileurl, name, params.download_path)
    if params.models == 'all' or params.models == 'inception5h':
        print("Downloading the Inception5h model")
        fileurl = "https://github.com/ProGamerGov/pytorch-old-tensorflow-models/raw/restructured-models/pt_inception5h.pth"
        name = "pt_inception5h.pth"
        download_file(fileurl, name, params.download_path)

    print("All selected models have been successfully downloaded")


def download_file(fileurl, name, download_path):
    if version_info[0] < 3:
        urllib.URLopener().retrieve(fileurl, path.join(download_path, name))
    else:
        urllib.request.urlretrieve(fileurl, path.join(download_path, name))


def params_list():
    parser = argparse.ArgumentParser()
    parser.add_argument("-models", choices=['all', 'places365', 'inception5h'], default='all')
    parser.add_argument("-download_path", help="Download location for models", default='models')
    params = parser.parse_args()
    return params



if __name__ == "__main__":
    main()