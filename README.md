# Computer Vision - Bayesian Semantic Image Registration

Author : Hugo Steiger  
Mentors : Marie-Odile Berger, Gilles Simon  
Full documentation (in french) : https://drive.google.com/drive/folders/1qpwtRY_KSRP7m6JFITHoyompFZOu7Wsp?usp=sharing

----------------------------------------------------------------------------------------------------------------------------------------------------------------

This repository contains elements of the work carried out for my second-year project at Mines Nancy and my second-year internship. This is my first experience in Data Science and Machine Learning, as it took place in the Computer Vision department (MAGRIT) of Nancy's LORIA computer science laboratory. The goal of the project was to develop a new method of image registration. Image registration is a CV field which consists in finding the spatial transformation between two poses of a camera, thanks to photo taken from each pose. It can also include finding the transformation between a picture (initial) and the same picture which has been rotated, translated and scaled up or down (target). Here is an example of image registration problem, where ΔX is the unknown variable :

![registration](https://user-images.githubusercontent.com/106969232/182454832-5ee6cab8-3c61-40d5-8652-bd0d2e33be88.JPG)

The new approach evaluated in this project uses a semantic segmentation of the initial and target picture to find the transformation. Basically, the transformation between the "target" and "init" pictures is infered from the transformations of the objects between "target" and "init". This solution was first explored by MAGRIT's PhD student Antoine Fond. As a bayesian model is used to find the transformation that is the best fit among the different objects (maximisation of the likelihood thansk to an EM algorithm), the project was entitled "Bayesian Semantic Image Registration". The semantic segmentation process consists in giving a label to every single pixel in the picture : it is a Machine Learning field and the state of the art solution for this probleme is a CNN. A retrained ResNet18 network was used in this project. SegNet has a similar architecture, and can be schematized as follows :

![Seg](https://user-images.githubusercontent.com/106969232/182454476-69d1312f-1b0a-437b-8e45-bb4f1f3d8bc1.JPG)

As the transformation estimation deeply depends on the semantic segmentation of the picture, an evaluation of the segmentation uncertainty is necessary. This was done using Monte Carlo Dropout sampling. This solution was introduced by Yarin Gal, a computer vision scientist, in his PhD. Dropout is the random deletion of network connections in a neural network. It is mainly used to make a training more robust, but it can also be used during forward passes of the network to estimate the stability of the model and, therefore, its uncertainty. Monte Carlo Dropout sampling produces many forward passes of the network with Dropout enabled (and therefore random connections cuts for each of them) : a variance calculus is done on the resulting segmentation maps. For each pixel, it is possible to observe how, on average, slight drops in the network affect the prediction. The greater the variance gets, the less stability there is as regards the estimation of the network. Here is a visual explanation of Monte Carlo Dropout Sampling :

![Dropout](https://user-images.githubusercontent.com/106969232/182455332-8f8ad5ff-5266-41dd-9588-8613b27f802a.JPG)

The Bayesian Semantic Image Registration strategy is the following :
- 


