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
parser.add_argument('--img_dir', type=str, default=r'E:\50_plex\tif\pipeline2\IL_corrected', help='path to the directory of input images')
parser.add_argument('--save_dir', type=str, default=r'E:\50_plex\tif\pipeline2\unmixed', help='path to the directory to save unmixed images')
parser.add_argument('--has_brightfield', type=bool, default=True, help='If last channel is brightfield')
parser.add_argument('--default_box', type=str, default='10000_7000_20000_12000', help='xmin_ymin_xmax_ymax')
parser.add_argument('--round_pattern', type=str, default='_R', help='pattern for round idx')
parser.add_argument('--channel_pattern', type=str, default='C', help='pattern for channel idx')

args = parser.parse_args()

img_dir = args.img_dir


def rescale_histogram(image):
    return exposure.rescale_intensity(image, in_range='image', out_range='dtype')


def get_unmixing_params(images):
    # make a list to keep result parameters
    results = np.zeros((len(images), len(images)))

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

    # get idx of channels for each file in script
    script_rows_idx = [np.nonzero(row)[0] for row in alphas]
    # convert idx of channel to filename
    script_rows_str = []
    for row_idx, row in enumerate(script_rows_idx):
        script_rows_str.append([])
        if len(row) != 0:
            new_row = []
            for idx in row:
                new_row.append(round_files[idx])
            script_rows_str[row_idx].extend(new_row)
            if len(script_rows_str[row_idx]) < 3:
                script_rows_str[row_idx] = append_nones(3, script_rows_str[row_idx])
            if len(script_rows_str[row_idx]) > 3:
                script_rows_str[row_idx] = [fname for _, fname in sorted(zip(alphas[row_idx], round_files),
                                                                         reverse=True)[:3]]
        else:
            script_rows_str[row_idx] = [None] * 3

    df = pd.DataFrame(script_rows_str, columns=['channel_1', 'channel_2', 'channel_3'], index=round_files)
    df = df.reindex(columns=df.columns.tolist() + ['xmin', 'ymin', 'xmax', 'ymax'])
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
        if np.sum(alpha) > 0.0:
            corrected_img = image - np.sum([a * img for a, img in zip(alpha, images) if a != 0.0], axis=0)
            corrected_img[corrected_img < 0] = 0
        else:
            corrected_img = image

        # extend shrank histogram
        corrected_img = rescale_histogram(corrected_img)

        # convert to uint16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corrected_img = img_as_uint(corrected_img)

        # save image
        save_name = os.path.join(args.save_dir, name)
        tifffile.imsave(save_name, corrected_img, bigtiff=True)


def main():
    files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    # find how many rounds and channels we have
    rounds = sorted(list(set([int(re.compile(args.round_pattern + '(\d+)').findall(file)[0]) for file in files])))
    channels = sorted(list(set([int(re.compile(args.channel_pattern + '(\d+)').findall(file)[0]) for file in files])))

    default_box = list(map(int, args.default_box.split('_')))
    xmin, ymin, xmax, ymax = default_box

    images = []
    rois = []
    for round_idx in rounds:
        print('*' * 50)
        print('Unxminging round {} ...'.format(round_idx))
        round_files = list(filter(lambda x: args.round_pattern + str(round_idx) in x, files))

        if args.has_brightfield:
            # last channel is brightfield so delete it
            round_files = [file for file in round_files if args.channel_pattern + str(max(channels)) not in file]

        # read images
        print('Reading images.')
        for filename in round_files:
            # read image and append to list
            image = img_as_float(tifffile.imread(os.path.join(img_dir, filename)))
            images.append(image)
            rois.append(np.copy(image[ymin:ymax, xmin:xmax]))

        # find unmixing params from ROIs
        print('Calculating unmixing parameters from ROIs.')
        alphas = get_unmixing_params(rois)
        # create folder and save unmixing parameters into csv file
        print('writing unmixing parameters in {}'.format(os.path.join(args.save_dir, 'unsupervised.csv')))
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        write_params_to_csv(os.path.join(args.save_dir, 'unsupervised.csv'), alphas, round_files)
        # save unmixed images
        print('Unmixing images and writing to disk')
        unmix_original_images(rois, images, alphas, round_files)

        images = []
        rois = []


if __name__ == '__main__':

    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Unmixing pipeline finished successfully in {} seconds.'.format(time.time() - start))


    # TODO: script is fixed with size of 3 endmembers -> np.count_nonzero(alphas)
    # TODO: add brightfiled channel number to args