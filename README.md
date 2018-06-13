## Semantic Video Segmentation by Gated Recurrent Flow Propagation
This repo contains the code for the CVPR 2018 paper "Semantic Video Segmentation by Gated Recurrent Flow Propagation" by David Nilsson and Cristian Sminchisescu. [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Nilsson_Semantic_Video_Segmentation_CVPR_2018_paper.pdf)

### Setup

Check config.py. Download all data from the cityscapes dataset and change the paths in config.py. Check that you can run python config.py without any errors.

Run misc/compile.sh to compile the bilinear warping operator. Change the include directory on line 9 if you get errors related to libcudart.

Download all pretrained models from [here](https://drive.google.com/open?id=1eGy7JcX1ptzxwQ6thEd2R_ix4VehLRQL) and unpack them under ./models. For instance, the file ./models/flownet1.index should exist.

### Usage

Evaluate the GRFP(LRR-4x, FlowNet2) setup on the validation set by running
```
python evaluate.py --static lrr --flow flownet2
```

Evaluation on PSP and Dilation10 as well as code for training will be added soon.

The values in table 9 can be reproduced by running the following. It takes about 4 hours on a titan X GPU.
```
python evaluate.py --static lrr --flow flownet2
python evaluate.py --static lrr --flow flownet1
python evaluate.py --static lrr --flow farneback
```


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

Depending on the setup you use, consider also citing [PSP](https://github.com/hszhao/PSPNet), [LRR](https://github.com/golnazghiasi/LRR), [Dilation](https://github.com/fyu/dilation), [FlowNet1](https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/), [FlowNet2](https://github.com/lmb-freiburg/flownet2) or [Farnebäck](https://link.springer.com/chapter/10.1007/3-540-45103-X_50).