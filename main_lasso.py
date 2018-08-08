import os
import re
import time
import argparse
import warnings
import pandas as pd
import numpy as np
from skimage import img_as_float, img_as_uint
from skimage import exposure
import tifffile
import matplotlib.pyplot as plt
from sklearn import linear_model

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='E:/50_plex/tif', help='path to the directory of images')
parser.add_argument('--default_box', type=str, default='16200_6100_21300_12200', help='xmin_ymin_xmax_ymax')
args = parser.parse_args()

img_dir = args.img_dir


def rescale_histogram(image):
    return exposure.rescale_intensity(image, in_range='image', out_range='dtype')


def get_unmixing_params(images):
    # make a list to keep result parameters
    results = np.zeros((10, 10))

    # for all channels except C0 (which is DAPI)
    for i in range(1, len(images)):
        endmembers = [np.ndarray.flatten(img) for img in images]
        source = endmembers.pop(i)

        clf = linear_model.Lasso(alpha=.0001, copy_X=True, positive=True)
        clf.fit(np.array(endmembers).T, source)
        alphas = np.insert(clf.coef_, i, 0)

        results[i, :] = alphas

    return results


def write_params_to_csv(filename, alphas, round_files):

    def append_nones(length, list_):
        """
        Appends Nones to list to get length of list equal to `length`.
        If list is too long raise AttributeError
        """
        diff_len = length - len(list_)
        if diff_len < 0:
            raise AttributeError('Length error list is too long.')
        return list_ + [None] * diff_len

    # get idx of channels for each file in scrip
    script_rows_idx = [np.nonzero(row)[0] for row in alphas]
    # convert idx of channel to filename
    script_rows_str = []
    for row_idx, row in enumerate(script_rows_idx):
        script_rows_str.append([])
        if len(row) != 0:
            for idx in row:
                new_row = [file for file in round_files if '_C{}_'.format(idx) in file][0]
                script_rows_str[row_idx].append(new_row)
            if len(script_rows_str[row_idx]) < 3:
                script_rows_str[row_idx] = append_nones(3, script_rows_str[row_idx])
            if len(script_rows_str[row_idx]) > 3:
                script_rows_str[row_idx] = script_rows_str[row_idx][:3]
        else:
            script_rows_str[row_idx] = [None] * 3

    df = pd.DataFrame(script_rows_str, columns=['channel_1', 'channel_2', 'channel_3'], index=round_files)
    df.index.name = 'filename'

    # save df to csv file or update existing one
    if os.path.isfile(filename):
        old_df = pd.read_csv(filename, index_col='filename')
        new_df = pd.concat([old_df, df], axis=0)
        new_df.to_csv(filename)
    else:
        df.to_csv(filename)


def unmix_original_images(rois, images, alphas, names):
    for roi, image, alpha, name in zip(rois, images, alphas, names):

        # clean artifacts in image
        max_s = np.max(roi)
        image[image > max_s] = 0

        # unmix whole brain image
        corrected_img = image - np.sum([a * img for a, img in zip(alpha, images) if a != 0.0], axis=0)
        corrected_img[corrected_img < 0] = 0

        # extend shrank histogram
        corrected_img = rescale_histogram(corrected_img)
        # save image
        new_name = os.path.join(img_dir, 'unmixed', os.path.splitext(name)[0] + '_unmixed' + os.path.splitext(name)[1])
        tifffile.imsave(new_name, corrected_img, bigtiff=True)


def main():
    files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    # create dir for results
    if not os.path.exists(os.path.join(img_dir, 'unmixed')):
        os.mkdir(os.path.join(img_dir, 'unmixed'))

    # find how many rounds and channels we have
    num_rounds = max(set([int(re.compile('_R(\d+)_').findall(file)[0])for file in files]))
    num_channels = max(set([int(re.compile('_C(\d+)_').findall(file)[0]) for file in files]))

    default_box = list(map(int, args.default_box.split('_')))
    xmin, ymin, xmax, ymax = default_box

    images = []
    rois = []
    for round_idx in range(1, num_rounds + 1):
        round_files = list(filter(lambda x: '_R{}_'.format(round_idx) in x, files))
        # last channel is brightfield so delete it
        round_files = [file for file in round_files if '_C{}'.format(num_channels) not in file]
        for channel_idx in range(num_channels):
            # get file name
            filename = list(filter(lambda x: '_C{}_'.format(channel_idx) in x, round_files))[0]
            # read image and append to list
            image = img_as_float(tifffile.imread(os.path.join(img_dir, filename)))
            images.append(image)
            rois.append(np.copy(image[ymin:ymax, xmin:xmax]))

        # find unmixing params from ROIs
        alphas = get_unmixing_params(rois)
        write_params_to_csv('step_1.csv', alphas, round_files)
        unmix_original_images(rois, images, alphas, round_files)

        images = []
        rois = []


if __name__ == '__main__':

    main()
    print()
    # script is fixed with size of 3 endmembers
    #TODO: np.count_nonzero(alphas)


