"""The gaussians are initialized thanks to the ground truth information of base.png""";
"""Indeed, for base.png i.e. the initial picture, is GT is supposed to be known.""";
I = imread(fullfile(pwd,"ver.png")); %ver.png is the GT info of base.png

% Pedestrian
% A GMM is obtained to represent the Pedestrian class thanks to an EM soft clustering algorithm
% over the GT.
X=zeros(1,720*960);
Y=zeros(1,720*960);
counter = 1;
% Points belonging to the GT Pedestrian class are retrieved and for each point,
% x-y coordinates are placed respectively in X and Y arrays
for i = 1 : 720
    for j = 1 : 960
        if (I(i,j,1)==255)
            X(counter) = j-1;
            Y(counter) = i-1;
            counter = counter + 1;
        end
    end
end
X = X(1,1:counter-1);
Y = Y(1,1:counter-1);

% GMM calculus thanks to EM. The code is from Mo Chen (sth4nth@gmail.com).
% 100 gaussians are initially used for the GMM soft clustering, but this 
% algorithm automatically removes the less interesting gaussians.
[~,model,llh] = mixGaussEm([X' Y']',100);

% GMM parameters
U = model.mu;
V = model.Sigma;
w = model.w;

% K is the final number of gaussians used to represent the Pedestrian class
K = size(U,2);

% Car : same process
X2=zeros(1,720*960);
Y2=zeros(1,720*960);
counter = 1;
for i = 1 : 720
    for j = 1 : 960
        if (I(i,j,3)==255)
            X2(counter) = j-1;
            Y2(counter) = i-1;
            counter = counter + 1;
        end
    end
end
X2 = X2(1,1:counter-1);
Y2 = Y2(1,1:counter-1);

[~,model,llh2] = mixGaussEm([X2' Y2']',100);
U2 = model.mu;
V2 = model.Sigma;
w2 = model.w;
K2 = size(U2,2);

% Optional : The GMM decomposition is displayed for the Pedestrian and Car
% classes
%{
Lm = zeros(720,960);
for k = 1 : K
    for i = 1 : 720 
        for j = 1 : 960
            Lm(i,j) = Lm(i,j) + w(k)*gaussian([0;0],1,U(:,k),V(:,:,k),[j-1; i-1]);
        end
    end
end

Lm2 = zeros(720,960);
for k = 1 : K2
    for i = 1 : 720 
        for j = 1 : 960
            Lm2(i,j) = Lm2(i,j) + w2(k)*gaussian([0;0],1,U2(:,k),V2(:,:,k),[j-1; i-1]);
        end
    end
end

I2 = imread('base.png');
subplot(2,1,1);
imshow(I2);
hold on;
h = imagesc(Lm);
set(h,'AlphaData', 0.8*ones(720,960));
title(sprintf("K = %d, Log-vraisemblance =  %2.4f", K, llh(end)  ));
hold off;

subplot(2,1,2);
imshow(I2);
hold on;
h = imagesc(Lm2);
set(h,'AlphaData', 0.8*ones(720,960));
title(sprintf("K = %d, Log-vraisemblance = %2.4f",  K2, llh2(end) ));
hold off;
%}