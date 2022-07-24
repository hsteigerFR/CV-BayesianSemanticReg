function [Ym,V] = uncertainty_eval(I,net,n)
    """This function uses Monte Carlo Dropout Sampling to calculate the uncertainty""";
    """of the CNN prediction for a given image I""";
    
    % The activation map for I at layer "drop7" is calculated
    E = activations(net, I, 'drop7');

    tic
    Y = single(zeros(720,960,11,n));
    
    % The Monte Carlo Dropout Sampling consists in using Dropout on a
    % network forward pass n times. Then the mean and the covariance matrix
    % of the different outputs are calculated. The variance and covariance
    % between classes is a measure of the model uncertainty.
    for u = 1 : n
        % From layer "drop7", each layer result until the output layer is
        % calculated, integrating the droping of some weights.
        u
        P = binornd(1,0.5*ones(22,30,4096));
        U=E.*P;

        L = zeros(22,30,11);
        W_L = net.Layers(39).Weights;
        B_L = net.Layers(39).Bias;

        for k = 1 : 11
            for c = 1 : 4096
                L(:,:,k) = L(:,:,k) + conv2(U(:,:,c),W_L(:,:,c,k));
            end
            L(:,:,k) = L(:,:,k) + B_L(:,:,k);
        end

        W_M = net.Layers(40).Weights;
        B_M = net.Layers(40).Bias;
        M  = zeros(736,992,11);

        for k = 1 : 11
            for i = 1 : 22
                for j = 1 : 30
                    for c = 1 : 11
                        M((i-1)*32+1:(i-1)*32+64,(j-1)*32+1:(j-1)*32+64,k) = M((i-1)*32+1:(i-1)*32+64,(j-1)*32+1:(j-1)*32+64,k) + L(i,j,c)*W_M(:,:,c,k);
                    end
                    M((i-1)*32+1:(i-1)*32+64,(j-1)*32+1:(j-1)*32+64,k) = M((i-1)*32+1:(i-1)*32+64,(j-1)*32+1:(j-1)*32+64,k) + B_M(:,:,k)*ones(64,64);
                end
            end
        end

        N = zeros(720,960,11);
        win = centerCropWindow2d(size(M(:,:,1)),[720 960]);
        for i = 1 : 11
            N(:,:,i) = imcrop(M(:,:,i),win);
        end

        temp = zeros(11,1);
        % For each iteration u, the output is saved in Y(:,:,:,u).
        for i = 1 : 720
            for j = 1 :960
                temp = softmax(reshape(N(i,j,:),[11 1]));
                Y(i,j,:,u) = single(reshape(temp,[1 1 11]));
            end
        end
    end

    % The metrics, that is to the mean Ym of probability maps over all
    % iterations and V, the covariance matrix of classes for each pixel,
    % are calculated thanks to what was saved in array Y.
    Ym = zeros(720,960,11);
    for u = 1 : n
        Ym = Ym + Y(:,:,:,u);
    end
    Ym = (1/n)*Ym;

    V = zeros(720,960,11,11);

    Ym_temp = zeros(1,11);
    Vtemp = zeros(11,11);

    for i = 1 : 720
        for j = 1 : 960
            Ym_temp = reshape(Ym(i,j,:),[1 11]);
            Vtemp = 1*diag(ones(1,11))-(Ym_temp.')*Ym_temp;
            for u = 1 : n
                Vtemp = Vtemp + (1/n)*(reshape(Y(i,j,:,u),[11 1]))*reshape(Y(i,j,:,u),[1 11]);
            end
            V(i,j,:,:) = Vtemp;
        end
    end
  
    % Optional : show some key plots
    fig = figure;

    subplot(4,2,1);
    imshow(Ym(:,:,9),[0 1]);
    title("Mean over 'Car'; Iterations = " + int2str(n));

    subplot(4,2,2);
    imshow(Ym(:,:,10),[0 1]);
    title("Mean over 'Pedestrian'");

    subplot(4,2,3);
    imshow(V(:,:,9,9),[]);
    title("Variance over 'Car'");

    subplot(4,2,4);
    imshow(V(:,:,10,10),[]);
    title("Variance over 'Pedestrian'");
    
    cov_1 = zeros(720,960);
    for i = 1 : 11
        if (i ~= 9)
            cov_1(:,:) = cov_1(:,:) + V(:,:,9,i);
        end
    end
    cov_1(:,:) = (1/10)*cov_1(:,:);
    subplot(4,2,5);
    imshow(abs(cov_1),[]);
    title("Covariances over 'Car'");

    cov_2 = zeros(720,960);
    for i = 1 : 11
        if (i ~= 10)
            cov_2(:,:) = cov_2(:,:) + V(:,:,10,i);
        end
    end
    cov_2(:,:) = (1/10)*cov_2(:,:);
    subplot(4,2,6);
    imshow(abs(cov_2),[]);
    title("Covariances over 'Pedestrian'");
    
    subplot(4,2,7);
    imshow(abs(V(:,:,9,10)),[]);
    title("Covariance over 'Car' - 'Pedestrian'");
    
    toc
end

