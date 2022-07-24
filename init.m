"""The target picture is built thanks to this script. Then, the CNN model""";
"""detects classes in the target picture""";
I = imread(fullfile(pwd,"base.png"));
T2 = [-5; -50]; % Here is the target translation
s2 = 1.2; % Here is the target scale

% The image registration is working if, at the end of the algorithm,
% T = T2, s = s2

% The brightness of the target picture is altered to make it different from
% the initial picture's
I = I + 50;
I = imtranslate(I,T2');
tform = affine2d([s2 0 0; 0 s2 0; 0 0 1]);
I = imwarp(I,tform,'OutputView', imref2d( size(I) ));

% The target picture is saved
imwrite(I,fullfile(pwd,"study.png"))
Ic = zeros(1,720*960);
Jc = zeros(1,720*960);

%The CNN model detecting Pedestrian and Cars is loaded and evaluated.
load("FCN.mat"); 
[labels,maxscores,allscores] = semanticseg(I,net);

I2 = imread(fullfile(pwd,"ver.png"));
I2 = imtranslate(I2,T2');
I2 = imwarp(I2,tform,'OutputView', imref2d( size(I2) ));

%CNN evaluation for each class
P = double(allscores(:,:,10)); %Piéton
P2 = double(allscores(:,:,9)); %Voiture

%To try it with the GT instead of the CNN evaluation :
%{
P = double(I2(:,:,1)/255); %Pedestrian
P2 = double(I2(:,:,3)/255); %Car
%}

% Pixels where the probabilities of Pedestrian and Car are below 10^-2 are
% filtered out. They will not be considered during the EM algorithm.

counter = 1;
for i = 1 : 720
    for j = 1 : 960
        if (P(i,j)>8*10^-1 || P2(i,j)>8*10^-1)
            Ic(counter) = i;
            Jc(counter) = j;
            counter = counter +1;
        end
    end
end
Ic = Ic(1,1:counter-1);
Jc = Jc(1,1:counter-1);

