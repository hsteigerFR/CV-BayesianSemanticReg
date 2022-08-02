# Computer Vision - Bayesian Semantic Image Registration

Author : Hugo Steiger  
Mentors : Marie-Odile Berger, Gilles Simon  
Full documentation (in french) : https://drive.google.com/drive/folders/1qpwtRY_KSRP7m6JFITHoyompFZOu7Wsp?usp=sharing

-------------------------------------------------------------------------------------------------------------------------------------------------------------

This repository contains elements of the work carried out for my second-year project at Mines Nancy and my second-year internship. This is my first experience in Data Science and Machine Learning, as it took place in the Computer Vision department (MAGRIT) of Nancy's LORIA computer science laboratory. The goal of the project was to develop a new method of image registration. Image registration is a CV field which consists in finding the spatial transformation between two poses of a camera, thanks to photo taken from each pose. It can also include finding the transformation between a picture (initial) and the same picture which has been rotated, translated and scaled up or down (target). Here is an example of image registration problem, where ΔX is the unknown variable :

![registration](https://user-images.githubusercontent.com/106969232/182454832-5ee6cab8-3c61-40d5-8652-bd0d2e33be88.JPG)

The new approach evaluated in this project uses a semantic segmentation of the initial and target picture to find the transformation. Basically, the transformation between the "target" and "init" pictures is infered from the transformations of the objects between "target" and "init". This solution was first explored by MAGRIT's PhD student Antoine Fond. As a bayesian model is used to find the transformation that is the best fit among the different objects (maximisation of the likelihood thansk to an EM algorithm), the project was entitled "Bayesian Semantic Image Registration". The semantic segmentation process consists in giving a label to every single pixel in the picture : it is a Machine Learning field and the state of the art solution for this probleme is a CNN. A retrained ResNet18 network was used in this project. The segmentation for the class Car can be schematized as follows :

![Seg](https://user-images.githubusercontent.com/106969232/182463353-644d4739-3e7b-4fe3-9cdd-66560fe6664b.JPG)

As the transformation estimation deeply depends on the semantic segmentation of the picture, an evaluation of the segmentation uncertainty is necessary. This was done using Monte Carlo Dropout sampling. This solution was introduced by Yarin Gal, a computer vision scientist, in his PhD. Dropout is the random deletion of network connections in a neural network. It is mainly used to make a training more robust, but it can also be used during forward passes of the network to estimate the stability of the model and, therefore, its uncertainty. Monte Carlo Dropout sampling produces many forward passes of the network with Dropout enabled (and therefore random connections cuts for each of them) : a variance calculus is done on the resulting segmentation maps. For each pixel, it is possible to observe how, on average, slight drops in the network affect the prediction. The greater the variance gets, the less stability there is as regards the estimation of the network. Here is a visual explanation of Monte Carlo Dropout Sampling :

![Dropout](https://user-images.githubusercontent.com/106969232/182455332-8f8ad5ff-5266-41dd-9588-8613b27f802a.JPG)

The Bayesian Semantic Image Registration strategy is the following :
- The "init" picture is supposed to have a GT truth. Soft clusters, i.e. gaussian mixture models (GMMs), are built to cover each object.
- The semantic segmentation is calculated over the "target" picture.
- If the gaussian mixture models (density f) are aligned with the semantic segmentation (probability p), then f * p will be maximum. This is what the likelihood of the model will measure. The GMMs have transformation parameters that will be optimized during the algorithm, to maximise the likelihood : these parameters should equal the transformation between pictures if the algorithm works fine. 
- This likelihood is maximised iteratively using an Expectation Maximisation algorithm (more details in the Drive documents above)
- The uncertainty of the target picture segmentation is used in this algorithm to weight the impact of each segmentation pixel on the registration (the less uncertain, the highest weight).
- In the end, the true and predicted transformations are compared using picture alignement.

An illustration of this strategy can be found below : 

![Strategy](https://user-images.githubusercontent.com/106969232/182458327-7d3e45a6-5b24-4435-bbff-5fb9e22a1dc2.JPG)

Here is an example of the process, each frame corresponds to an EM iteration :

![demo2](https://user-images.githubusercontent.com/106969232/182462037-96255559-f0d1-4adc-97af-783ae3767823.gif)

One semantic class is used in this example : Car. Here is the evolution of the GMM for the Car class :

![gaussian](https://user-images.githubusercontent.com/106969232/182461088-27c66432-5909-45f3-bca2-c7e5673c4381.gif)

The transformation is not perfectly estimated with a single class. If a second class is used, i.e. the Person class, the result is better. The code in this repository proposes the situation with two classes : Car and Person. Here is the result of the process with two classes :

![demo](https://user-images.githubusercontent.com/106969232/182461847-eb1b4cb4-4876-411a-9326-33bda6d8b6d9.gif)

In the end, here are some examples of uncertainty calculus. The left column corresponds to the raw calculus and the right column to the refined uncertainty :

![var](https://user-images.githubusercontent.com/106969232/182464089-7dd3d5d4-1525-4621-9e02-e14711162ee6.JPG)

As regards the limitations of the algorithm, it requires a lot of computation power and the calculus can be really slow. The use of a high level programming language such as Matlab may not be adapted as it slows down the process even more. The number of points that is used in the EM algorithm, if too high, makes the algorithm unstable and some points have to be filtered out. The algorithm is stable for segmentation probabilities above 0.8, but is really prone to fall into local extrema if the tranformation between the two pictures is too important. In order to deal with this issue, a first approximation of the transformation can be done with EdgeBoxes (Yolov5 algorithm for instance), to start the algorithm with this transformation rather than the identity. A study was also led with a "Window" class for building facades registration.

-------------------------------------------------------------------------------------------------------------------------------------------------------------
