## Semantic Video Segmentation by Gated Recurrent Flow Propagation
This repo contains the code for the CVPR 2018 paper "Semantic Video Segmentation by Gated Recurrent Flow Propagation" by David Nilsson and Cristian Sminchisescu. [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Nilsson_Semantic_Video_Segmentation_CVPR_2018_paper.pdf)

### Setup

Check config.py. Download all data from the cityscapes dataset and change the paths in config.py. Check that you can run python config.py without any errors.

Run misc/compile.sh to compile the bilinear warping operator. Change the include directory on line 9 if you get errors related to libcudart.

Run misc/download_pretrained_models.sh to download the models used in the paper.

### Usage

Reproduce the values in table 9 by running the following. It takes about 4 hours on a titan X GPU.
```
python -u evaluate.py --static lrr --flow flownet1 2>&1 | tee logs/log_flownet1.txt
python -u evaluate.py --static lrr --flow flownet2 2>&1 | tee logs/log_flownet2.txt
python -u evaluate.py --static lrr --flow farneback 2>&1 | tee logs/log_farneback.txt
python -u evaluate.py --static lrr --flow farneback --frames 1 2>&1 | tee logs/log_lrr_static.txt
```

Evaluation on PSP and Dilation10 as well as code for training will be added soon.

### Citation
If you use the code in your own research, please cite
```
@InProceedings{Nilsson_2018_CVPR,
author = {Nilsson, David and Sminchisescu, Cristian},
title = {Semantic Video Segmentation by Gated Recurrent Flow Propagation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

Depending on the setup you use, consider also citing PSP, LRR, Dilation, Flownet1, Flownet2 or Farnebäck.