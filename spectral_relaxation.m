% Generate random dataset
X = randn(1000, 2);

% K-means
K = 3; % Number of clusters
max_iter = 100; % Maximum number of iterations
centroids = X(randperm(size(X, 1), K), :); % Initialize centroids
for i = 1:max_iter
    % Assign each point to nearest centroid
    [~, labels] = pdist2(centroids, X, 'euclidean', 'Smallest', 1);
    % Update centroids
    for j = 1:K
        centroids(j, :) = mean(X(labels == j, :), 1);
    end
end
% Plot results
figure; hold on;
scatter(X(:,1), X(:,2), 10, labels, 'filled');
scatter(centroids(:,1), centroids(:,2), 50, 'k', 'filled');
title('K-means Clustering');

% Spectral relaxed k-means
W = pdist2(X, X, 'euclidean');
sigma = median(W(:));
D = diag(sum(W, 2));
L = D^(-1/2) * W * D^(-1/2);
[V, ~] = eig(L);
V = V(:, 2:K+1);
V = normr(V);
max_iter = 100; % Maximum number of iterations
centroids = V(randperm(size(V, 1), K), :); % Initialize centroids
for i = 1:max_iter
    % Assign each point to nearest centroid
    [~, labels] = pdist2(centroids, V, 'euclidean', 'Smallest', 1);
    % Update centroids
    for j = 1:K
        centroids(j, :) = mean(V(labels == j, :), 1);
    end
end
% Plot results
figure; hold on;
scatter(X(:,1), X(:,2), 10, labels, 'filled');
scatter(centroids(:,1), centroids(:,2), 50, 'k', 'filled');
title('Spectral Relaxed K-means Clustering');
