import os
import numpy as np
from numpy import linalg as LA
from scipy import optimize
import tifffile
import matplotlib.pyplot as plt

visualize = True

img_dir = r'E:\\50_plex\\tif'
src_name = 'S1_R2_C1-C11_A1_C9_IlluminationCorrected_stitched_registered.tif'
noise_names = ['S1_R4_C1-C11_A1_C6_IlluminationCorrected_stitched_registered.tif',
               [],
               []
               ]

xmin = 22000
ymin = 13500
xmax = 26000
ymax = 14500


# read original images
source = tifffile.imread(os.path.join(img_dir, src_name)).astype(float) / 65535
noises = []
for noise_name in noise_names:
    if noise_name:
        noises.append(tifffile.imread(os.path.join(img_dir, noise_name)).astype(float) / 65535)
    else:
        noises.append(np.zeros_like(source))

# create small crop for optimization
s = np.copy(source)[ymin:ymax, xmin:xmax]  # crop sample for source image
n = []
for noise in noises:
        n.append(np.copy(noise)[ymin:ymax, xmin:xmax])   # crop sample for noise image


# define optimization function
def f(params):
    alpha1, alpha2, alpha3 = params
    return LA.norm(s - alpha1 * n[0] - alpha2 * n[1] - alpha3 * n[2])

# optimize paramteres
initial_guess = [0, 0, 0]
result = optimize.minimize(f, initial_guess, bounds=((0, 1), (0, 1), (0, 1)))

alpha = result.x
print(alpha)


def float_to_uint16(image):
    image_u16 = np.copy(image) * 65535
    image_u16[image_u16 < 0] = 0
    return np.uint16(image_u16)


if visualize:
    c = s - alpha[0] * n[0] - alpha[1] * n[1] - alpha[2] * n[2]
    c = float_to_uint16(c)


    ax1 = plt.subplot(311)
    plt.imshow(s, cmap='gray')
    plt.axis('off')
    plt.title('source')

    plt.subplot(334, sharex=ax1, sharey=ax1)
    plt.imshow(n[0], cmap='gray')
    plt.axis('off')
    plt.title('noise')

    plt.subplot(335, sharex=ax1, sharey=ax1)
    plt.imshow(n[1], cmap='gray')
    plt.axis('off')
    plt.title('noise')

    plt.subplot(336, sharex=ax1, sharey=ax1)
    plt.imshow(n[2], cmap='gray')
    plt.axis('off')
    plt.title('noise')

    plt.subplot(313, sharex=ax1, sharey=ax1)
    plt.imshow(c, cmap='gray')
    plt.axis('off')
    plt.title('corrected')

    plt.show()

# create and save new image
new_img = source - alpha[0] * noises[0] - alpha[1] * noises[1] - alpha[2] * noises[2]
new_img = float_to_uint16(new_img)

src_fname = os.path.splitext(src_name)[0]
new_name = src_fname + '_unmixed.tif'
tifffile.imsave(new_name, new_img, bigtiff=True)

print('Done.')