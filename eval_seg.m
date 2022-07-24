"""This script enables to test the semantic segmentation on a picture.""";
"""The uncertainty of the prediction can also be computed thanks to Monte Carlo""";
"""Dropout sampling.""";

load("FCN.mat")
I = imread("study.png");
I = imresize(I,[720,960]);

classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];

% Use model and show the segmentation :
[labels,maxscores,allscores] = semanticseg(I,net);
U = zeros(720,960,3);
U(:,:,3) = allscores(:,:,9);
U(:,:,1) = allscores(:,:,10);
imshow(U)

% Show model uncertainty
[Ym,V] = uncertainty_eval(I,net,5);

% Optional : Show the clustering as well as the probabilities
%{
U = zeros(720,960,3);
for i = 1 : 720
    for j= 1:960
        if (labels(i,j) == "Car")
            U(i,j,3) = allscores(i,j,9);
        end
        
        if (labels(i,j) == "Pedestrian")
            U(i,j,1) = allscores(i,j,10);
        end
    end
end
imshow(U)
%}
