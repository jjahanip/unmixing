import os
import time
import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import optimize
import tifffile
import matplotlib.pyplot as plt

visualize = False

img_dir = r'E:\50_plex\tif'
script_file = 'script.csv'


df = pd.read_csv('script.csv')

# add/update new alphas for each channel
df["alpha1"] = np.nan if 'alpha1' not in df.columns else None
df["alpha2"] = np.nan if 'alpha2' not in df.columns else None
df["alpha3"] = np.nan if 'alpha3' not in df.columns else None


for index, row in df.iterrows():

    # if no noise channel is given
    if row[["channel_1", "channel_2", "channel_3"]].isnull().all():
        continue

    start = time.time()

    # read parameters
    src_name = row["filename"]
    print('==============================================================')
    print('unmixing image {}'.format(src_name))

    n1_name = row["channel_1"]
    n2_name = row["channel_2"]
    n3_name = row["channel_3"]

    # load images
    source = tifffile.imread(os.path.join(img_dir, src_name)).astype(float) / 65535

    if str(n1_name) == 'nan':
        noise_1 = np.empty_like(source)
    else:
        noise_1 = tifffile.imread(os.path.join(img_dir, n1_name)).astype(float) / 65535

    if str(n2_name) == 'nan':
        noise_2 = np.empty_like(source)
    else:
        noise_2 = tifffile.imread(os.path.join(img_dir, n2_name)).astype(float) / 65535

    if str(n3_name) == 'nan':
        noise_3 = np.empty_like(source)
    else:
        noise_3 = tifffile.imread(os.path.join(img_dir, n3_name)).astype(float) / 65535

    xmin = int(row["xmin"]) if pd.notna(row['xmin']) else source.shape[1] // 2 - 500
    ymin = int(row["ymin"]) if pd.notna(row['ymin']) else source.shape[0] // 2 - 500
    xmax = int(row["xmax"]) if pd.notna(row['xmax']) else source.shape[1] // 2 + 500
    ymax = int(row["ymax"]) if pd.notna(row['ymax']) else source.shape[0] // 2 + 500

    # create small crop for optimization
    s = source[ymin:ymax, xmin:xmax]
    n1 = noise_1[ymin:ymax, xmin:xmax]
    n2 = noise_2[ymin:ymax, xmin:xmax]
    n3 = noise_3[ymin:ymax, xmin:xmax]

    # define optimization function
    def f(params):
        alpha1, alpha2, alpha3 = params
        return LA.norm(s - alpha1 * n1 - alpha2 * n2 - alpha3 * n3)

    # optimize paramteres
    initial_guess = [0, 0, 0]
    result = optimize.minimize(f, initial_guess, bounds=((0, 1), (0, 1), (0, 1)))
    alpha1, alpha2, alpha3 = result.x
    print('\n'.join('alpha_{} = {:.2f}'.format(i, alpha) for i, alpha in enumerate(result.x)))


    def float_to_uint16(image):
        image_u16 = np.copy(image) * 65535
        image_u16[image_u16 < 0] = 0
        return np.uint16(image_u16)


    if visualize:
        c = s - alpha1 * n1 - alpha2 * n2 - alpha3 * n3
        c = float_to_uint16(c)


        ax1 = plt.subplot(311)
        plt.imshow(s, cmap='gray')
        plt.axis('off')
        plt.title('source')

        plt.subplot(334, sharex=ax1, sharey=ax1)
        plt.imshow(n1, cmap='gray')
        plt.axis('off')
        plt.title('noise')

        plt.subplot(335, sharex=ax1, sharey=ax1)
        plt.imshow(n2, cmap='gray')
        plt.axis('off')
        plt.title('noise')

        plt.subplot(336, sharex=ax1, sharey=ax1)
        plt.imshow(n3, cmap='gray')
        plt.axis('off')
        plt.title('noise')

        plt.subplot(313, sharex=ax1, sharey=ax1)
        plt.imshow(c, cmap='gray')
        plt.axis('off')
        plt.title('corrected')

        plt.show()

    # create and save new image
    new_img = source - alpha1 * noise_1 - alpha2 * noise_2 - alpha3 * noise_3
    new_img = float_to_uint16(new_img)

    src_fname = os.path.splitext(src_name)[0]
    new_name = os.path.join(img_dir, src_fname + '_unmixed.tif')
    tifffile.imsave(new_name, new_img, bigtiff=True)

    # update alphas with derived values
    df.loc[index, 'alpha1'] = alpha1
    df.loc[index, 'alpha2'] = alpha2
    df.loc[index, 'alpha3'] = alpha3

    end = time.time()
    df.to_csv(os.path.splitext(script_file)[0] + '_unmixed.csv', index=False)
    print('time = {}'.format(end - start))

print('==============================================================')
print('done!')




