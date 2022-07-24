"""An image registration is carried out with x/y translations and scale factor as variabes.""";
"""Choice here : the scale factor s is applied before the translation T.""";
"""The resulting system of equations is non linear and has no explicit solutions. fsolve is used to solve it.""";
ver_ter; % Creates initial GMMs to represent Pedestrian and Car
init;
clc;
% Constants
H = 720; %Image height
W = 960; %Image width
n = H*W; %Image size (number of pixels)
N = 100; %Max iteration count

% Variables 
T = [0;0]; %Translation
s = 1; %Scale factor
Q = zeros(1,N); %Log-likelihood array

Z = zeros(H,W,K); %Pedestrian latent variables
Z2 = zeros(H,W,K2); %Car latent variables

Zo = zeros(H,W); %Outlier / background latent variables

G = zeros(H,W,K); %Pedestrian GMM
G2 = zeros(H,W,K2); %Car GMM

% Normalization paramaters
alpha = 1;
lambda = alpha /(H*W);

% Weights for each of the GMMs. w for Pedestrian, w2 for Cars. wo is for
% Outliers.
wo = w;
wo2 = w2;

% The initial input picture is displayed, as well as the target picture.
% At first, they are not registered with each other as the EM algorithm
% has not started yet.
f =figure();
I2 = imread("study.png");
imshow(I2);
hold on;
I = imread("base.png");
I = imtranslate(I,T');
tform = affine2d([s 0 0; 0 s 0; 0 0 1]);
I = imwarp(I,tform,'OutputView', imref2d( size(I) ));
h2 = imshow(I);
set(h2,'AlphaData', 0.5*ones(720,960));
hold off;
frame = getframe(f); 
im = frame2im(frame); 
[imind,cm] = rgb2ind(im,256); 
imwrite(imind,cm,"results/full.gif",'gif', 'Loopcount',inf,'DelayTime',0.01); 
close(f);

%Loop
for t = 1 : N
    
    %%% Expectation-maximization algorithm - Step E

    %-Gaussians probabilities for the Car class and Pedestrian class are calculated
    % for each coordinates where the CNN evaluation for these classes is above a given
    % threshold.
    for i = 1 : length(Ic)
        for k= 1 : K
            G(Ic(i),Jc(i),k) = gaussian(T,s,U(:,k),V(:,:,k),[Jc(i)-1;Ic(i)-1]);
        end
        
        for k2 = 1 : K2
            G2(Ic(i),Jc(i),k2) = gaussian(T,s,U2(:,k2),V2(:,:,k2),[Jc(i)-1;Ic(i)-1]);
        end
    end
    
    %-The latent variables can be calculated thanks to the update formula.
    % The common denominator to a given couple of coordinates (i,j) is calculated first.
    d = zeros(H,W);

    for i = 1 : length(Ic)
        for k = 1 : K
            d(Ic(i),Jc(i)) = d(Ic(i),Jc(i)) + w(k)*G(Ic(i),Jc(i),k)*P(Ic(i),Jc(i));
        end
        
        for k2 = 1 : K2
            d(Ic(i),Jc(i)) = d(Ic(i),Jc(i)) + w2(k2)*G2(Ic(i),Jc(i),k2)*P2(Ic(i),Jc(i));
        end
        
        d(Ic(i),Jc(i)) = d(Ic(i),Jc(i)) + lambda;
    end
    
    % Then, each numerator is calculated (a given numerator depends on i,j,the class and the n° in 
    % the GMM), and divided by the previously calculated d.

    for i = 1 : length(Ic)
        for k = 1 : K
            Z(Ic(i),Jc(i),k) = w(k)*G(Ic(i),Jc(i),k)*P(Ic(i),Jc(i))/d(Ic(i),Jc(i));
        end
        
        for k2 = 1 : K2
            Z2(Ic(i),Jc(i),k2) = w2(k2)*G2(Ic(i),Jc(i),k2)*P2(Ic(i),Jc(i))/d(Ic(i),Jc(i));
        end
        
        Zo(Ic(i),Jc(i)) = lambda/d(Ic(i),Jc(i));
    end
    
    %%%Step M
    % The non-linear system, depending on the newly calculated latent variables, is solved with fsolve
    f = @(x) systemP(x(1),x(2),x(3),U,V,U2,V2,Z,Z2,Ic,Jc);
    S = fsolve(f,[T(1),T(2),s],optimoptions('fsolve','Algorithm','levenberg-marquardt','Display','iter'));
    T(1)= S(1);
    T(2) = S(2);
    s = S(3);
    
    % The likelihood Q(t) [for iteration t] is calculated. It depends on the Pedestrian class
    % but also the Car class :
    for i = 1 : length(Ic)
        Q(t) = Q(t) + Zo(Ic(i),Jc(i))*log(lambda);
        for k = 1 : K
            Q(t) = Q(t) + Z(Ic(i),Jc(i),k)*(log(w(k)) + log(G(Ic(i),Jc(i),k)+10^-5) + log(P(Ic(i),Jc(i))+10^-5));
        end
        
        for k2 = 1 : K2
            Q(t) = Q(t) + Z2(Ic(i),Jc(i),k2)*(log(w2(k2)) + log(G2(Ic(i),Jc(i),k2)+10^-5) + log(P2(Ic(i),Jc(i))+10^-5));
        end
    end
    % Q is shown
    Q
    
    % As latent variables have changed in the E step, the normalization
    % factor needs to be updated.
    d2 = 0;
    n2 = zeros(1,K);
    for k = 1 : K 
        d2 = d2 + (w(k)-1);
        for i = 1 : length(Ic)
            d2 = d2 + Z(Ic(i),Jc(i),k);
            n2(k) = n2(k) + Z(Ic(i),Jc(i),k);
        end
    end
    
    n22 = zeros(1,K2);
    for k2 = 1 : K2
        d2 = d2 + (w2(k2)-1);
        for i = 1 : length(Ic)
            d2 = d2 + Z2(Ic(i),Jc(i),k2);
            n22(k2) = n22(k2) + Z2(Ic(i),Jc(i),k2);
        end
    end
    alpha = sum(Zo,'all')/d2;
    lambda = alpha /(H*W);
    
    % Optional : the weights for each gaussian within the GMMs are updated
    %{
    for k = 1 : K 
        w(k) = abs((wo(k)-1 + n2(k))/d2);
    end
    
    for k2 = 1 : K2 
        w2(k) = abs((wo2(k)-1 + n22(k2))/d2);
    end
    %}
    
    % For each iteration of the EM algorithm, the alignement with the
    % target picture is shown and saved.
    f =figure();
    I2 = imread("study.png");
    imshow(I2);
    hold on;

    I = imread("base.png");
    I = imtranslate(I,T');
    tform = affine2d([s 0 0; 0 s 0; 0 0 1]);
    I = imwarp(I,tform,'OutputView', imref2d( size(I) ));
    h2 = imshow(I);
    set(h2,'AlphaData', 0.5*ones(720,960));
    frame = getframe(f); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    imwrite(imind,cm,"results/full.gif",'gif','WriteMode','append','DelayTime',0.01); 
    close(f);
    
    % The pedestrian GMM is shown and saved
    temp = zeros(720,960);
    for k =1: K
        temp = temp + w(k)*G(:,:,k);
    end
    f = figure();
    imagesc(temp);
    axis equal;
    title(sprintf("%d",t));
    frame = getframe(f); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    if t == 1 
        imwrite(imind,cm,"results/G_ped.gif",'gif', 'Loopcount',inf,'DelayTime',0.01); 
    else 
        imwrite(imind,cm,"results/G_ped.gif",'gif','WriteMode','append','DelayTime',0.01); 
    end 
    close(f);
    
    % The car GMM is shown and saved
    temp = zeros(720,960);
    for k2 =1: K2
        temp = temp + w2(k2)*G2(:,:,k2);
    end
    f = figure();
    imagesc(temp);
    axis equal;
    title(sprintf("%d",t));
    frame = getframe(f); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    if t == 1 
        imwrite(imind,cm,"results/G_car.gif",'gif', 'Loopcount',inf,'DelayTime',0.01); 
    else 
        imwrite(imind,cm,"results/G_car.gif",'gif','WriteMode','append','DelayTime',0.01); 
    end 
    close(f);
    
    
    % The pedestrian related latent variables are shown and saved as well
    temp = zeros(720,960);
    for k =1: K
        temp = temp + Z(:,:,k);
    end
    f = figure();
    imagesc(temp);
    axis equal;
    title(sprintf("%d",t));
    frame = getframe(f); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    if t == 1 
        imwrite(imind,cm,"results/Z_ped.gif",'gif', 'Loopcount',inf,'DelayTime',0.01); 
    else 
        imwrite(imind,cm,"results/Z_ped.gif",'gif','WriteMode','append','DelayTime',0.01); 
    end 
    close(f);
    
    % The car related latent variables are shown and saved as well
    temp = zeros(720,960);
    for k2 =1: K2
        temp = temp + Z2(:,:,k2);
    end
    f = figure();
    imagesc(temp);
    axis equal;
    title(sprintf("%d",t));
    frame = getframe(f); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    if t == 1 
        imwrite(imind,cm,"results/Z_car.gif",'gif', 'Loopcount',inf,'DelayTime',0.01); 
    else 
        imwrite(imind,cm,"results/Z_car.gif",'gif','WriteMode','append','DelayTime',0.01); 
    end 
    close(f);
    
    % The Outlier / background related latent variables are shown and saved as well
    f = figure();
    imagesc(Zo);
    axis equal;
    title(sprintf("%d",t));
    frame = getframe(f); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    if t == 1 
        imwrite(imind,cm,"results/Z_out.gif",'gif', 'Loopcount',inf,'DelayTime',0.01); 
    else 
        imwrite(imind,cm,"results/Z_out.gif",'gif','WriteMode','append','DelayTime',0.01); 
    end 
    close(f);
    
% At the end of the iteration, the current transformation and the expected transformation
% are shown
fprintf("Expected parameters : [%3.3f,%3.3f,%3.3f]\n",[T2',s2])
fprintf("Current parameters : [%3.3f,%3.3f,%3.3f]\n\n",[T',s])
 
end

% Final alignement / registration check
figure;
I2 = imread("study.png");
h=imshow(I2);
hold on;

I = imread("base.png");
I = imtranslate(I,T');
tform = affine2d([s 0 0; 0 s 0; 0 0 1]);
I = imwarp(I,tform,'OutputView', imref2d( size(I) ));
h2 = imshow(I);
set(h2,'AlphaData', 0.5*ones(720,960));
hold off;

%Obtained and expected parameters
fprintf("Expected parameters : [%3.3f,%3.3f,%3.3f]\n",[T2',s2])
fprintf("Obtained parameters : [%3.3f,%3.3f,%3.3f]\n",[T',s])

