# unmixing_pipeline:

unmixing_pipeline is a tool for unmixing Immunohistochemistry (IHC) images taken in multiple channels and rounds.


## Dependencies:

* numpy
* scipy
* scikit-image
* scikit-learn
* tifffile (conda-forge)
* matplotlib

## Pipeline:

You can run the unmxing pipeline in 2 modes:
1. `main_unsupervised`: In this mode, all the channels of same round are going to be unmixed with
 __sparse spectral unmixing__ algorithm.
 
    ```bash
    python main_unsupervised.py --img_dir=/path/to/input/dir \
                                --save_dir=/path/to/save/dir \
                                --round_pattern=R \
                                --channel_pattern=C \
                                --has_brightfield=True    
    ``` 
    It will save unmixed images and generate a script of unmixed channels in the `save_dir`:
    ![Alt text](files/0.png)

2. `main_supervised`: In this mode, user can correct the script and all the channels provided in the script are going to be unmixed with
 __chemical co-localization unmixing__ algorithm.
 
    ![Alt text](files/1.png)
    
    ```bash
    python main_supervised.py --img_dir=/path/to/input/dir \
                              --save_dir=/path/to/save/dir \
                              --script_file=/path/to/script/file   
    ```
    It will save unmixed images in `save_dir`.


## Arguments:

1. `main_unsupervised`:
    
    |Argument|Help|Example|
    |---|---|---|
    |img_dir|Path to the directory of images|--img_dir=C:\images\input|
    |save_dir|Path to the directory to save unmixed images|--save_dir=C:\images\output|
    |default_box|Selected box coordinates xmin_ymin_xmax_ymax|--default_box=16200_6100_21300_12200|
    |has_brightfield|If last channel is brightfield|--has_brightfield=True|
    |round_pattern|Pattern for round idx|--round_pattern=R|
    |channel_pattern|Pattern for channel idx|--channel_pattern=C|
2. `main_supervised`:

    |Argument|Help|Example|
    |---|---|---|
    |img_dir|path to the directory of images|--img_dir=C:\images\input|
    |save_dir|path to the directory to save unmixed images|--save_dir=C:\images\output|
    |script_file|csv script file name|--script_file=script.csv|
    |default_box|selected box coordinates xmin_ymin_xmax_ymax|--default_box=16200_6100_21300_12200|
    |visualize|visualize the unmixing report of crop|--visualize|

## Notes:

* if `--visualized` passed:
    ![Alt text](files/2.png)

