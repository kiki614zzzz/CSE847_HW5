% Load USPS dataset
load USPS.mat

% Data pre-processing
A = double(A); % Convert to double precision
A = A - mean(A,1); % Mean subtraction
A = A ./ std(A,[],1); % Standardization

% SVD for PCA
[U,S,V] = svd(A,'econ');

% Reconstruction error for different number of principal components
recon_error = zeros(1,4);
for i = 1:4
    % Select p principal components
    p = [10,50,100,200];
    Up = U(:,1:p(i));
    Sp = S(1:p(i),1:p(i));
    Vp = V(:,1:p(i));
    
    % Reconstruct images
    Ap = Up * Sp * Vp';
    recon_error(i) = norm(A - Ap,'fro') / norm(A,'fro');
    
    % Display reconstructed images
    for j = 1:2
        img = reshape(A(j,:),16,16)';
        img_recon = reshape(Ap(j,:),16,16)';
        subplot(2,4,i+(j-1)*4)
        imshow([img,img_recon])
        title(['p = ',num2str(p(i))])
    end
end

% Display total reconstruction error
disp('Total Reconstruction Error:')
disp(recon_error)
