---
layout: post
title: "Triplet Representation for 2D Human Body Understanding"
date: 2020-01-10 22:00:00 +0800
description: The design of a novel triplet representation of human body and its estimation algorithm. Appear in ICCV2019.
img: logo/triplet-representation.jpg # Add image post (optional)
tags: [HumanPose]
paper: https://arxiv.org/abs/1910.11535
dataset: https://github.com/kennymckormick/Triplet-Representation-of-human-Body/
---

This post gives a brief introduction to the paper: [TRB: A Novel Triplet Representation for Understanding 2D Human Body](http://openaccess.thecvf.com/content_ICCV_2019/papers/Duan_TRB_A_Novel_Triplet_Representation_for_Understanding_2D_Human_Body_ICCV_2019_paper.pdf). The [dataset](https://github.com/kennymckormick/Triplet-Representation-of-human-Body/) used in this paper is released.

# Abstract

Human pose and shape are two important components of 2D human body. However, how to efficiently represent both of them in images is still an open question. In this paper, we propose the Triplet Representation for Body (TRB) -- a compact 2D human body representation, with skeleton keypoints capturing human pose information and contour keypoints containing human shape information. TRB not only preserves the flexibility of skeleton keypoint representation, but also contains rich pose and human shape information. Therefore, it promises broader application areas, such as human shape editing and conditional image generation. We further introduce the challenging problem of TRB estimation, where joint learning of human pose and shape is required. We construct several large-scale TRB estimation datasets, based on popular 2D pose datasets: LSP, MPII, COCO. To effectively solve TRB estimation, we propose a two-branch network (TRB-net) with three novel techniques, namely X-structure (Xs), Directional Convolution (DC) and Pairwise Mapping (PM), to enforce multi-level message passing for joint feature learning. We evaluate our proposed TRB-net and several leading approaches on our proposed TRB datasets, and demonstrate the superiority of our method through extensive evaluations.

# The TRB representation

A comprehensive 2D human body representation should capture both human pose and shape information. Such representation is promising for applications beyond plain keypoint localization, such as media generation and editing. Before the booming of deep learning, people establish a model and fit it to images for applications like pose estimation. In those models, pictorial structure[1] use articulated rigid rectangles to represent human, while deformable structure[2] and contour people[3] add more parameters to describe human shape. In deep learning era, skeleton-keypoint is the mostly used representation for 2D human. However, it focuses on human pose and ignores human shape.

![](/assets/img/triplet-representation/traditional.png)

Due to the incompleteness of skeleton-keypoint. We proposed the Triplet Representation of human Body (**TRB**), which incorporates both 2D pose and shape, while as simple as skeleton keypoints. TRB is extended from skeleton keypoints, by adding two contour points for each skeleton point. Those contour points locate on both sides of human contour. Based on this definition, we annotate contour points on three popular pose estimation dataset: MPII[4], LSP[5] and COCO[6], extending them into TRB dataset. Some samples are displayed in the image below (Blue points denote medial contour points, green points denote lateral ones). Currently, annotations in [MPII-trb](https://github.com/kennymckormick/Triplet-Representation-of-human-Body) is released.

![](/assets/img/triplet-representation/trb-sample.png)

# Method

We first try current pose estimation approaches on TRB estimation task, and mainly have two observations. First, contour keypoint is harder to estimate than skeleton keypoint. Second, estimating TRB leads to similar of inferior performance of skeleton keypoints, despite the richer supervision used. Thus we developed TRB-Net, a two branch network for TRB estimation. Various backbones can be used in TRB-Net, indicated by MS Block here. We developed various message passing modules, denoted by MP Block, to exchange information between 2 branches.

![](/assets/img/triplet-representation/trb-net.png)

We designed three message passing modules, named X-structure, Directional Convolution and Pairwise Mapping. The X-structure is the most naïve one. It concatenates current branch feature and transformed remote branch feature to predict the refined heatmap. Directional convolution further improves X-structure. It updates feature in inside-out or outside-in order during one pass convolution. By doing so, it simulates gathering from contour to skeleton and scattering from skeleton to contour, makes feature transformation more efficient. In pairwise mapping, for each pair of keypoints, the warping between them is estimated.  A consistency loss is added on the predicted heatmap and warping. These predictions will then be ensemble to give out a better prediction of TRB.

![](/assets/img/triplet-representation/message-passing.png)

# Experiment Results & Applications

For TRB estimation, TRB-Net improves accuracy by nearly 2% comparing to the baseline, since it improves message passing between skeleton branch and contour branch. For skeleton keypoint estimation, TRB-Net also out-performs its baseline, and competitive to state-of-the-art methods on MPII and LSP. On COCO dataset, using TRB annotation and TRB-Net, similar accuracy can be obtained with only half amount of data, demonstrates the data efficiency of TRB.

![](/assets/img/triplet-representation/results.png)

TRB has rich potential in applications. In below example, we use a vunet[7] for TRB conditioned human image generation, which can disentangle human appearance and shape and then combine them again arbitarily. We manipulate contour points to change human shape of shoulder, torso and legs in these three demos respectively. This approach can be potentially used in human shape editing.


<style>
	.boxes{
        width:33%;
        float:left;
	}
	#mainDiv{
		width:100%;
		margin:auto;
	}
	img{
		max-width:100%;
	}
</style>
<div id="mainDiv">
    <div id="divOne" class="boxes">
	<img src="/assets/img/triplet-representation/shou.gif">
    </div>
    <div id="divTwo" class="boxes">
	<img src="/assets/img/triplet-representation/torso.gif">
    </div>
    <div id="divTwo" class="boxes">
	<img src="/assets/img/triplet-representation/leg.gif">
    </div>
</div>



### References

[1] Pedro F Felzenszwalb and Daniel P Huttenlocher. Pictorial structures for object recognition. International journal of computer vision, 61(1):55–79, 2005

[2] Silvia Zuffi, Oren Freifeld, and Michael J Black. From pictorial structures to deformable structures. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 3546–3553. IEEE, 2012

[3] Freifeld, O., Weiss, A., Zuffi, S., & Black, M. J. (2010, June). Contour people: A parameterized model of 2D articulated human shape. In *2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition* (pp. 639-646). IEEE.

[4] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2d human pose estimation: New benchmark and state of the art analysis. In Proceedings of the IEEE Conference on computer Vision and Pattern Recognition, pages 3686–3693, 2014.

[5] Sam Johnson and Mark Everingham. Clustered pose and nonlinear appearance models for human pose estimation. 2010.

[6] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence ´ Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[7] Patrick Esser, Ekaterina Sutter, and Bjorn Ommer. A variational u-net for conditional appearance and shape generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8857–8866, 2018.
