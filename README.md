# unmixing_pipeline:

unmixing_pipeline is a tool for unmixing Immunohistochemistry (IHC) images taken in multiple channels and rounds.


# Dependencies:

* numpy
* scipy
* scikit-image
* tiffile
* matplotlib

# Pipeline:

Providing a script with channel information, you can unmix noise channels from source channel:

![Alt text](files/1.png)

# Arguments:

* --img_dir     : help = path to the directory of images                example: --img_dir E:/50_plex/tif
* --script_file : help = csv script file name                           example: --script_file script.csv
* --default_box : help = selected box coordinates xmin_ymin_xmax_ymax   example: --default_box 16200_6100_21300_12200
* --visualize   : help = visualize the unmixing report of crop          example: --visualize

```bash
python main.py --img_dir E:/50_plex/tif --script_file script.csv --default_box 16200_6100_21300_12200 --visualize
```

if --visualized pass:

![Alt text](files/2.png)

# Returns:

1. Unmixed image will be saved with **_unmixed** suffix in `img_dir`.
2. Report of unmixed values will be saved as **script_unmixed.csv** in working directory.



