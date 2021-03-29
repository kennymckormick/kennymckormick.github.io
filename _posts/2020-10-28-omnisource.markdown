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

## Abstract

We introduce OmniSource, a novel framework for leveraging web data to train video recognition models. OmniSource overcomes the barriers between data formats, such as images, short videos, and long untrimmed videos for webly-supervised learning. First, data samples with multiple formats, curated by task-specific data collection and automatically filtered by a teacher model, are transformed into a unified form. Then a joint-training strategy is proposed to deal with the domain gaps between multiple data sources and formats in webly-supervised learning. Several good practices, including data balancing, resampling, and cross-dataset mixup are adopted in joint training. Experiments show that by utilizing data from multiple sources and formats, OmniSource is more data-efficient in training. With only 3.5M images and 800K minutes videos crawled from the internet without human labeling (less than 2% of prior works), our models learned with OmniSource improve Top-1 accuracy of 2D- and 3D-ConvNet baseline models by 3.0% and 3.9%, respectively, on the Kinetics-400 benchmark. With OmniSource, we establish new records with different pretraining strategies for video recognition. Our best models achieve 80.4%, 80.5%, and 83.6 Top-1 accuracies on the Kinetics-400 benchmark respectively for training-from-scratch, ImageNet pre-training and IG-65M pre-training.

## Problem Background

Collecting large-scale human-labeled datasets for trimmed video recognition is costly and time-consuming. An annotator has to go through the entire untrimmed video and manually cut it into informative clips based on a specific query. As a result, while the quantity of web videos grows exponentially over the past three years, the Kinetics dataset merely grows from 300K videos to 650K videos, partially limiting the scaling-up of video architecture. To exploit the rich and diversified web data, some recent works [1, 2] explore the possibility of pre-training from massive unlabeled web images or videos with hashtags. However, these approaches typically require billions of images / dozens of millions of videos to obtain a strong pre-training model, which poses great costs and restricts its practicability. In this work, we propose a data-efficient framework (named OmniSource) for video classification which can utilize multiple sources of web data including images, trimmed videos and untrimmed videos simultaneously. 

|                 | Webly-supervised pretrain                                    | Web-scale semi-supervised                                    | OmniSource                                                   |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Procedure       | 1. Train a model $\mathcal{M}$ on  $\mathcal{U}$ <br>2. Finetune $\mathcal{M}$ on $\mathcal{D_T}$ | 1. Train a model $\mathcal{M}$ on $\mathcal{D_T}$ <br>2. Run $\mathcal{M}$ on $\mathcal{U}$ to get pseudo-labeled $\mathcal{\hat{D}}$ <br>3. Train a student model $\mathcal{M'}$ on $\mathcal{\hat{D}}$ <br>4. Finetune $\mathcal{M'}$ on $\mathcal{D_T}$ | 1. Train one (or more) model $\mathcal{M}$ on $\mathcal{D_T}$ <br>2. Run $\mathcal{M}$ on $\bigcup_i \mathcal{U_i}$ to get pseudo labeled $\bigcup_i \mathcal{\hat{D_i}}$ <br> (Samples under certain threshold are dropped)<br>3. Apply transforms $\mathcal{T_i}$: $\mathcal{\hat{D_i}} \rightarrow \mathcal{D_{A,i}}$ <br>4. Train model $\mathcal{M'}$ (or $\mathcal{M}$) on $\mathcal{D_T} \cup \mathcal{D_A}$ |
| $|\mathcal{U}|$ | 3.5B images / 65M videos                                     | 1B images / 65M videos                                       | $\mathcal{|U|}$: 13M images & 1.4M videos (0.4%~2%) <br>$\mathcal{D_A}$: 3.5M images & 0.8M videos (0.1%~1%) |

Table 1: **Difference to previous works. ** $\mathcal{U}$ is the unlabeled web data, $\mathcal{D_T}$ is the target dataset. $\mathcal{|U|}$, $\mathcal{D_A}$ denote the scale of web data and filtered auxiliary dataset.


## OmniSource Framework



### Reference

[1] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In ECCV, pages 181–196, 2018.

[2] Deepti Ghadiyaram, Du Tran, and Dhruv Mahajan. Large-scale weakly-supervised pretraining for video action recognition. In CVPR, pages 12046–12055, 2019. 