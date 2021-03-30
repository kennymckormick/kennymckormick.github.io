---
layout: post
title: "Omni-sourced Webly-supervised Learning for Video Recognition"
date: 2020-10-27 22:00:00 +0800
description: We introduce OmniSource, a novel framework for leveraging web data to train video recognition models. Appear in ECCV2020.
img: logo/omnisource.png # Add image post (optional)
tags: [VideoRecognition]
paper: https://arxiv.org/abs/2003.13042
code: https://github.com/open-mmlab/mmaction2
dataset: https://github.com/open-mmlab/mmaction2/tree/master/tools/data/omnisource
---

# Abstract

We introduce OmniSource, a novel framework for leveraging web data to train video recognition models. OmniSource overcomes the barriers between data formats, such as images, short videos, and long untrimmed videos for webly-supervised learning. First, data samples with multiple formats, curated by task-specific data collection and automatically filtered by a teacher model, are transformed into a unified form. Then a joint-training strategy is proposed to deal with the domain gaps between multiple data sources and formats in webly-supervised learning. Several good practices, including data balancing, resampling, and cross-dataset mixup are adopted in joint training. Experiments show that by utilizing data from multiple sources and formats, OmniSource is more data-efficient in training. With only 3.5M images and 800K minutes videos crawled from the internet without human labeling (less than 2% of prior works), our models learned with OmniSource improve Top-1 accuracy of 2D- and 3D-ConvNet baseline models by 3.0% and 3.9%, respectively, on the Kinetics-400 benchmark. With OmniSource, we establish new records with different pretraining strategies for video recognition. Our best models achieve 80.4%, 80.5%, and 83.6 Top-1 accuracies on the Kinetics-400 benchmark respectively for training-from-scratch, ImageNet pre-training and IG-65M pre-training.

# Problem Background

Collecting large-scale human-labeled datasets for trimmed video recognition is costly and time-consuming. An annotator has to go through the entire untrimmed video and manually cut it into informative clips based on a specific query. As a result, while the quantity of web videos grows exponentially over the past three years, the Kinetics dataset merely grows from 300K videos to 650K videos, partially limiting the scaling-up of video architecture. To exploit the rich and diversified web data, some recent works [1, 2] explore the possibility of pre-training from massive unlabeled web images or videos with hashtags. However, these approaches typically require billions of images / dozens of millions of videos to obtain a strong pre-training model, which poses great costs and restricts its practicability. In this work, we propose a data-efficient framework (named OmniSource) for video classification which can utilize multiple sources of web data including images, trimmed videos and untrimmed videos simultaneously.


|                 | Webly-supervised pretrain [1,2]                            | Web-scale semi-supervised [3]                                | OmniSource                                                   |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Procedure       | 1. Train a model $$\mathcal{M}$$ on $$\mathcal{U}$$ <br>2. Finetune $$\mathcal{M}$$ on $$\mathcal{D_T}$$ | 1. Train a model $$\mathcal{M}$$ on $$\mathcal{D_T}$$ <br>2. Run $$\mathcal{M}$$ on $$\mathcal{U}$$ to get pseudo-labeled $$\mathcal{\hat{D}}$$ <br>3. Train a student model $$\mathcal{M'}$$ on $$\mathcal{\hat{D}}$$ <br>4. Finetune $$\mathcal{M'}$$ on $$\mathcal{D_T}$$ | 1. Train one (or more) model $$\mathcal{M}$$ on $$\mathcal{D_T}$$ <br>2. Run $$\mathcal{M}$$ on $$\bigcup_i \mathcal{U_i}$$ to get pseudo labeled $$\bigcup_i \mathcal{\hat{D_i}}$$ <br> (Samples under certain threshold are dropped)<br>3. Apply transforms $$\mathcal{T_i}$$: $$\mathcal{\hat{D_i}} \rightarrow \mathcal{D_{A,i}}$$ <br>4. Train model $$\mathcal{M'}$$ (or $$\mathcal{M}$$) on $$\mathcal{D_T} \cup \mathcal{D_A}$$ |
| $$\vert \mathcal{U} \vert$$ | 3.5B images / 65M videos                                     | 1B images / 65M videos                                       | $$\vert \mathcal{U} \vert$$: 13M images & 1.4M videos (0.4%~2%) <br>$$\mathcal{D_A}$$: 3.5M images & 0.8M videos (0.1%~1%) |

Table 1: **Difference to previous works.** $$\mathcal{U}$$ is the unlabeled web data, $$\mathcal{D_T}$$ is the target dataset. $$\mathcal{\vert U\vert}$$, $$\mathcal{\vert D_A \vert}$$ denote the scale of web data and filtered auxiliary dataset.

# OmniSource Framework

<img src="/assets/img/omnisource/pipeline.png" width="750px" class="center">

Figure 1: **OmniSource Framework.**  We first use the teacher network trained on the target dataset to filter collected task-specific web data, including images, short videos, long videos, to reduce noise and improve data quality. Specific transformations are then applied to the filtered out data corresponding to their formats. The target dataset and auxiliary web datasets are used for joint training of the student network.

## Task-Specific Data Collection

|     Source Dataset      |     Format      |    Raw Size     | Raw Storage |   Clean Size    |
| :---------------------: | :-------------: | :-------------: | :---------: | :-------------: |
|  GoogleImage (GG-img)   |      image      |       6M        |    350GB    |       2M        |
| InstagramImage (IG-img) |      image      |      7.4M       |    450GB    |      1.5M       |
| InstagramVideo (IG-vid) |  trimmed video  | 1.1M, 480K mins |   1.74TB    | 500K, 250K mins |
|       K400-untrim       | untrimmed video |    670K mins    |   2.44TB    |    500K mins    |

Table 2: **Dataset Statistics.** The web datasets we crawled for the target dataset Kinetics-400. The datasets are available at [MMAction2]([mmaction2/tools/data/omnisource at master · open-mmlab/mmaction2 (github.com)](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/omnisource)). You can fill in a [request form](https://docs.google.com/forms/d/e/1FAIpQLSd8_GlmHzG8FcDbW-OEu__G7qLgOSYZpH-i5vYVJcu7wcb_TQ/viewform?usp=sf_link) to obtain them.

<img src="/assets/img/omnisource/data_collection.png" width="750px" class="center"> 

Figure 2: **Task-Specific Data Collection.**  We simply use class names as keywords for data crawling, optionally with automatic permutation and stemming, free from additional human labor.

## Teacher Filtering

Data crawled from the web are inevitably noisy. Directly using collected web data for joint training leads to a significant accuracy drop (over 3%). To prevent irrelevant data from polluting the training set, we first train a teacher network $$\mathcal{M}$$ on the target dataset and discard web data with low confidence scores. In experiments, we use 2D teachers for web images, 3D teachers for web videos. We also find that better teacher always lead to better students.

<img src="/assets/img/omnisource/teacher_filtering.png" width="750px" class="center"> 

Figure 3: **Web Data Distribution.** The inter-class distribution of three web datasets is visualized in (a), both before and after filtering. (b) gives out samples of filtered out images and remain ones for GoogleImage. 

## Transforming to the target domain

<img src="/assets/img/omnisource/transformation.png" width="750px" class="center"> 

Figure 4: **Transformations.** We use different transformations to convert heterogeneous web data (images, untrimmed videos) to the target domain (trimmed videos). Left: Inflating images to clips, by replicating or inflating with perspective warping. Right: Extracting segments or clips from untrimmed videos, guided by confidence scores.

## Joint Training

Since the auxiliary dataset may be much larger and the domain gap may occur, the size ratio between target and auxiliary mini-batches is crucial for final performance. Besides, both intra- and cross-dataset mixup contributes to performance when models are trained from scratch. We try several resampling strategies to tailor the class distribution into a more balanced one, which also yield nontrivial improvements. 

# Experiments

## Main results

|   Arch/Dataset    |    K400     |   +GG-img   | +[GG&IG]-img |   +IG-vid   | +K400-untrim |    +All     |
| :---------------: | :---------: | :---------: | :----------: | :---------: | :----------: | :---------: |
|   TSN-3seg-R50    | 70.6 / 89.4 | 71.5 / 89.5 | 72.0 / 90.0  | 72.0 / 90.3 | 71.7 / 89.6  | 73.6 / 91.0 |
| SlowOnly-4x16-R50 | 73.8 / 90.9 | 74.5 / 91.4 | 75.2 / 91.6  | 75.2 / 91.7 | 74.5 / 91.1  | 76.6 / 92.5 |

Table 3: **Every Source Contributes.** Each source contributes to the target task. With all sources combined, the improvement can be more considerable. (Format: Top-1 Acc / Top-5 Acc)

| Architecture  |   Backbone   |    Pretrain    |  w/o. Omni  |   w. Omni   | $$\Delta$$  |
| :-----------: | :----------: | :------------: | :---------: | :---------: | :---------: |
|   TSN-3seg    |   ResNet50   |    ImageNet    | 70.6 / 89.4 | 73.6 / 91.0 | +3.0 / +1.6 |
|   TSN-3seg    |   ResNet50   | IG-1B (Image)  | 73.1 / 90.4 | 75.7 / 91.9 | +2.6 / +1.5 |
|   TSN-3seg    | Efficient-b4 |    ImageNet    | 73.3 / 91.0 | 75.2 / 92.0 | +1.9 / +1.0 |
| SlowOnly-4x16 |   ResNet50   |       -        | 72.9 / 90.9 | 76.8 / 92.5 | +3.9 / +1.6 |
| SlowOnly-4x16 |   ResNet50   |    ImageNet    | 73.8 / 90.9 | 76.6 / 92.5 | +2.8 / +1.6 |
| SlowOnly-8x8  |  ResNet101   |       -        | 76.3 / 92.6 | 80.4 / 94.4 | +4.1 / +1.8 |
| SlowOnly-8x8  |  ResNet101   |    ImageNet    | 76.8 / 92.8 | 80.5 / 94.4 | +3.7 / +1.6 |
|   irCSN-152   |  irCSN-152   | IG-65M (Video) | 82.6 / 95.3 | 83.6 / 96.0 | +1.0 / +0.7 |

Table 4: **Improvement under various experiment configurations.** OmniSource is extensively tested on various architectures with various pretraining strategies. The improvement is significant in **ALL** tested choices. Even for the SOTA setting, which uses 65M web videos for pretraining, OmniSource still improves the Top-1 accuracy by 1.0% (Format: Top-1 / Top-5 Acc)

|         Method          |   Backbone   | Pretrain |  Top-1   |  Top-5   |
| :---------------------: | :----------: | :------: | :------: | :------: |
|        TSN-7seg         | Inception-v3 | ImageNet |   73.9   |   91.1   |
|        TSM-8seg         |   ResNet50   | ImageNet |   72.8   |    -     |
|   TSN-3seg (**Ours**)   |   ResNet50   | ImageNet |   73.6   |   91.0   |
|   TSN-3seg (**Ours**)   | Efficient-b4 | ImageNet | **75.2** | **92.0** |
|      SlowOnly-8x8       |  ResNet101   |    -     |   75.9   |    -     |
|      SlowFast-8x8       |  ResNet101   |    -     |   77.9   |   93.2   |
| SlowOnly-8x8 (**Ours**) |  ResNet101   |    -     | **80.4** | **94.4** |
|        I3D-64x1         | Inception-v1 | ImageNet |   72.1   |   90.3   |
|        NL-128x1         |  ResNet101   | ImageNet |   77.7   |   93.3   |
|      SlowFast-8x8       |  ResNet101   | ImageNet |   77.9   |   93.2   |
| SlowOnly-8x8 (**Ours**) |  ResNet101   | ImageNet | **80.5** | **94.4** |
|       irCSN-32x2        |  irCSN-152   |  IG-65M  |   82.6   |   95.3   |
|  irCSN-32x2 (**Ours**)  |  irCSN-152   |  IG-65M  | **83.6** | **96.0** |

Table 5: **Comparison with Kinetics-400 state-of-the-art.**

## Transfer Learning

|      Architecture      | Pretrain | UCF101-Top1<br>(w/o. or w. Omni) | HMDB51-Top1<br>(w/o. or w. Omni) |
| :--------------------: | :------: | :------------------------------: | :------------------------------: |
|   TSN-3seg-ResNet50    | ImageNet |           91.5 / 93.3            |           63.5 / 65.9            |
| TSN-3seg-Efficient-b4  | ImageNet |           92.5 / 93.1            |           66.3 / 66.5            |
| SlowOnly-4x16-ResNet50 |    -     |           94.1 / 96.0            |           65.8 / 71.0            |
| SlowOnly-4x16-ResNet50 | ImageNet |           94.7 / 96.0            |           69.4 / 70.7            |
| SlowOnly-8x8-ResNet101 |    -     |           96.6 / 97.5            |           75.8 / 79.0            |
| SlowOnly-8x8-ResNet101 | ImageNet |           96.4 / 97.4            |           76.4 / 79.0            |

Table 6: **Detailed results of transfer learning.** We report Top-1 accuracies on the split-1 of UCF101 and HMDB51. OmniSource framework can learn better representation that transfers to other recognition tasks well, even without ImageNet pretraining.

|            Model             |   Modality    |       Pretrain        | UCF101-Top1 | HMDB51-Top1 |
| :--------------------------: | :-----------: | :-------------------: | :---------: | :---------: |
|          Two-Stream          |   RGB+Flow    |       ImageNet        |    88.0     |    59.4     |
|             TSN              |   RGB+Flow    |       ImageNet        |    94.2     |    69.4     |
|             I3D              |      RGB      |  ImageNet + Kinetics  |    95.6     |    74.8     |
|             I3D              |   RGB+Flow    |  ImageNet + Kinetics  |    98.0     |    80.7     |
|         I3D + PoTion         | RGB+Flow+Pose |  ImageNet + Kinetics  |    98.2     |    80.9     |
|          I3D + PA3D          | RGB+Flow+Pose |  ImageNet + Kinetics  |      -      |    82.1     |
|      SlowOnly-8x8-R101       |      RGB      | Kinetics + OmniSource |    97.3     |    79.0     |
| SlowOnly-8x8-R101 + I3D-Flow |   RGB+Flow    | Kinetics + OmniSource |    98.6     |    83.8     |

Table 7:  **Comparison with UCF-101 & HMDB-51 state-of-the-art.** (mean accuracies over three splits are reported). OmniSource not only outperforms RGB-Only methods, when fused with the flow stream, it surpasses all methods by a large margin, even for those which ensemble results of RGB, Flow and other modalities.

|   Architecture    |         Pretrain          | mAP on AVA v2.1 val |
| :---------------: | :-----------------------: | :-----------------: |
| SlowOnly-4x16-R50 |       Kinetics-400        |        20.1         |
| SlowOnly-4x16-R50 | Kinetics-400 + OmniSource |        21.8         |
| SlowOnly-8x8-R101 |       Kinetics-400        |        24.6         |
| SlowOnly-8x8-R101 | Kinetics-400 + OmniSource |        25.9         |

Table 8: The learned representation also transfer well to spatio-temporal action detection, leads to significant improvement on AVA validation. 

## Interpretation of OmniSource Gain

To find out why web data help, we mainly focus on the confusion pairs that web images can improve. We define the confusion score of a class pair as  $$s_{ij} = (n_{ij} + n_{ji}) / (n_{ij} + n_{ji} + n_{ii} + n_{jj})$$, $$n_{ij}$$ denotes the number of images with ground-truth label $$i$$ but being recognized as class $$j$$. Lower confusion score denotes better discriminating power between the two classes. We visualize some confusing pairs in Fig 5. The improvement can be mainly attributed to two reasons: 

(1) Web data usually focus on key objects of action. For example, we find that in those pairs with the largest confusion score reduction, there exist pairs like "drinking beer"  vs. "drinking shots", and "eating hotdog" vs. "eating chips". Training with web data leads to better object recognition ability. 

(2) Web data usually include discriminative poses, especially for those actions which last for a short time. For example, "rock scissors paper" vs. "shaking hands" has the second-largest confusion score reduction. Other examples include "sniffing" - "headbutting", "break dancing" - "robot dancing".

<img src="/assets/img/omnisource/interpretation.png" width="750px" class="center"> 

Figure 5: **Confusing pairs improved by OmniSource.** The original accuracy and change are denoted in black and in color. 

<img src="/assets/img/omnisource/eating.png" width="750px" class="center"> 

Figure 6: **Improvement on eating something.** Rows denote groundtruth and columns denote predictions. Block$$_{ij}$$ represents the difference in numbers of samples which belongs to class $$i$$ but recognized as class $$j$$ between the baseline and our model.

### Reference

[1] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In ECCV, pages 181–196, 2018.

[2] Deepti Ghadiyaram, Du Tran, and Dhruv Mahajan. Large-scale weakly-supervised pretraining for video action recognition. In CVPR, pages 12046–12055, 2019.

[3] I Zeki Yalniz, Herve Jegou, Kan Chen, Manohar Paluri, and Dhruv Mahajan. Billion-scale semi-supervised learning for image classification. arXiv preprint arXiv:1905.00546
