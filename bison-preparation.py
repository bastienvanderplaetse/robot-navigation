import argparse
import json
import numpy as np
import sys
import urllib.request as req

import display
import explorer_helper as exh

from ImageFactory import ImageFactory
from PIL import Image as PIL_Image
from pycocotools.coco import COCO
from pprint import pprint

try:
    from urllib.request import urlretrieve, urlopen
except ImportError:
    from urllib import urlretrieve
    from urllib2 import urlopen

ANNOTATIONS_DIRECTORY = "annotations"
DATA_TYPE = "val2014"

BISON_DIRECTORY = "bison"
BISON_ANNOTATIONS_DIRECTORY = "{}/{}".format(BISON_DIRECTORY, ANNOTATIONS_DIRECTORY)
BISON_ANNOTATIONS="{}/instances_{}.json".format(BISON_ANNOTATIONS_DIRECTORY, DATA_TYPE)
BISON_URL = "https://raw.githubusercontent.com/facebookresearch/binary-image-selection/master/annotations/bison_annotations.coco{}.json".format(DATA_TYPE)

COCO_DIRECTORY = "coco"
COCO_ANNOTATIONS_DIRECTORY = "{}/{}".format(COCO_DIRECTORY, ANNOTATIONS_DIRECTORY)
COCO_ZIP="{}/annotations_train{}.zip".format(COCO_ANNOTATIONS_DIRECTORY, DATA_TYPE)
COCO_ANNOTATIONS="{}/instances_{}.json".format(COCO_ANNOTATIONS_DIRECTORY, DATA_TYPE)
COCO_ANNOTATIONS_INTERIOR = "annotations/instances_{}.json".format(DATA_TYPE)
COCO_CAPTIONS="{}/captions_{}.json".format(COCO_ANNOTATIONS_DIRECTORY, DATA_TYPE)
COCO_CAPTIONS_INTERIOR = "annotations/captions_{}.json".format(DATA_TYPE)
COCO_URL = "http://images.cocodataset.org/annotations/annotations_train{}.zip".format(DATA_TYPE)

IMAGES_DIRECTORY = "images"
IMAGE_FILE = IMAGES_DIRECTORY + "/{}.jpg"



def check_args(argv):
    """Checks and parses the arguments of the command typed by the user
    Parameters
    ----------
    argv :
        The arguments of the command typed by the user
    Returns
    -------
    ArgumentParser
        the values of the arguments of the commande typed by the user
    """
    parser = argparse.ArgumentParser(description="Convert Bison dataset into \
        a file which can be loaded with NumPy. Each element of the NumPy array \
        is a dict object in which VI1 is the encoding of the first image, VI2 is\
        the encoding of the second image, S is the caption and T is the correct\
        image associated with S (0 for the first one and 1 for the second one)")

    parser.add_argument('OUTPUT', type=str, help="the name of the output file")
    # parser.add_argument('-a', '--annotations', help="file containing the Bison\
    #     annotations")

    args = parser.parse_args()

    return args

# def download_annotations():
#     # Download annotations
#     response = req.urlopen(URL).read().decode('utf-8')
#     response = json.loads(response)
#
#     # Save annotations
#     exh.create_directory(ROOT_DIRECTORY)
#     exh.create_directory(ANNOTATIONS_DIRECTORY)
#     exh.write_json(response, ANNOTATIONS)
#
#     return response
#
# def load_bison_annotations():
#     if filename == None:
#         print("No annotation file selected. Downloading default one")
#         annotations = download_annotations()
#         display.display_info("Default annotation file saved in {}".format(ANNOTATIONS))
#     else:
#         print("Loadingannotations {}".format(filename))
#         annotations = exh.load_json(filename)
#         display.display_ok("Loading done")
#
#     return annotations["data"]

def prepare_directories():
    exh.create_directory(BISON_DIRECTORY)
    exh.create_directory(BISON_ANNOTATIONS_DIRECTORY)
    exh.create_directory(COCO_DIRECTORY)
    exh.create_directory(COCO_ANNOTATIONS_DIRECTORY)
    exh.create_directory(IMAGES_DIRECTORY)

def download_bison():
    if exh.file_exists(BISON_ANNOTATIONS):
        print("Bison annotations already existing")
    else:
        print("Bison annotations do not exist. Downloading...")

        # Download annotations
        response = req.urlopen(BISON_URL).read().decode('utf-8')
        response = json.loads(response)
        display.display_ok("Bison annotations successfully downloaded")

        # Save annotations
        exh.write_json(response, BISON_ANNOTATIONS)
        display.display_info("Bison annotations saved in {}".format(BISON_ANNOTATIONS))

# def download_coco_ann():
#     if exh.file_exists(COCO_ANNOTATIONS):
#         print("COCO annotations already existing")
#     else:
#         # Download ZIP file
#         if exh.file_exists(COCO_ANNOTATIONS_ZIP):
#             print("COCO annotations already downloaded.")
#         else:
#             print("Downloading COCO annotations...")
#             urlretrieve(COCO_ANNOTATIONS_URL, COCO_ANNOTATIONS_ZIP)
#             display.display_ok("Download completed")
#         print("Extracting COCO annotations")
#         exh.unzip_single_file(COCO_ANNOTATIONS_ZIP, COCO_ANNOTATIONS, COCO_ANNOTATIONS_INTERIOR)
#         display.display_info("COCO annotations saved in {}".format(COCO_ANNOTATIONS))

def download_coco(filename, zip_file, zip_url, internal_filename, keyword):
    if exh.file_exists(filename):
        print("COCO {} already existing".format(keyword))
    else:
        # Download ZIP file
        if exh.file_exists(zip_file):
            print("COCO {} already downloaded".format(keyword))
        else:
            print("Downloading COCO {}...".format(keyword))
            urlretrieve(zip_url, zip_file)
            display.display_ok("Download completed")
        print("Extracting COCO {}".format(keyword))
        exh.unzip_single_file(zip_file, filename, internal_filename)
        display.display_info("COCO {} saved in {}".format(keyword, filename))

def download_annotations():
    download_bison()
    download_coco(COCO_ANNOTATIONS, COCO_ZIP, COCO_URL, COCO_ANNOTATIONS_INTERIOR, "annotations")
    download_coco(COCO_CAPTIONS, COCO_ZIP, COCO_URL, COCO_CAPTIONS_INTERIOR, "captions")

def download_image(id, coco):
    img_url = coco.loadImgs(id)[0]['coco_url']
    urlretrieve(img_url, IMAGE_FILE.format(id))

def encode_image(id, image_factory):
    img = PIL_Image.open(IMAGE_FILE.format(id))
    features = np.array(image_factory.get_features(img)).squeeze()
    return features

def convert_data():
    coco_ann = COCO(COCO_ANNOTATIONS)
    coco_captions = COCO(COCO_CAPTIONS)
    image_factory = ImageFactory(resize=256,crop=224)
    annotations = exh.load_json(BISON_ANNOTATIONS)['data']

    print("Converting Bison data set")

    data = []
    counter = 1
    M = len(annotations)
    for annotation in annotations:
        s = "Converting annotation {0} / {1}".format(counter, M)
        print (s, end="\r")

        data_row = []

        for candidate in annotation['image_candidates']:
            image_id = candidate['image_id']
            download_image(image_id, coco_ann)
            features_img = encode_image(image_id, image_factory)
            data_row.append(features_img)

        caption = coco_captions.loadAnns(annotation['caption_id'])[0]['caption']
        data_row.append(caption)

        if annotation['true_image_id'] == image_id:
            data_row.append(1)
        else:
            data_row.append(0)

        data.append(np.array(data_row))
        counter = counter + 1
        
    print(s)

    display.display_ok("Bison dataset converted")
    return np.array(data)

def save_data(data, output):
    filename = "{}/{}".format(BISON_DIRECTORY, output)
    np.save(filename, data)
    display.display_info("Data setsaved at {}".format(filename))

def run(args):
    prepare_directories()
    download_annotations()
    data = convert_data()
    save_data(data, args.OUTPUT)

if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)
