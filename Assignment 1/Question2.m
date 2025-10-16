%% EECE5644 - Assignment 1 Question 2
% 4-class Gaussian mixture, MAP and ERM classifiers
clear; close all; clc;
rng(1);

%% Parameters
mu(:,:,1) = [0 0];
mu(:,:,2) = [3.5 0.5];
mu(:,:,3) = [0.5 3.0];
mu(:,:,4) = [3.0 3.0]; % My own 4 distinct Gaussian class conditional pdfs p(x|L=j), j∈{1,2,3,4}

Sigma(:,:,1) = [0.8 0.2; 0.2 0.6];
Sigma(:,:,2) = [0.6 -0.15; -0.15 0.5];
Sigma(:,:,3) = [0.5 0; 0 0.5];
Sigma(:,:,4) = [1.0 0.3; 0.3 0.9]; % Standard deviation

priors = [0.25 0.25 0.25 0.25]; % The class prior is 0.25
K = 4; N = 10000;

%% Generate data
labels = randsample(1:K, N, true, priors);
X = zeros(N,2);
for j=1:K
    idx = find(labels==j);
    X(idx,:) = mvnrnd(mu(:,:,j), Sigma(:,:,j), numel(idx));
end

px_given = zeros(N,K); % Compute class conditional pdfs
for j=1:K
    px_given(:,j) = mvnpdf(X, mu(:,:,j), Sigma(:,:,j));
end

%% Part A: MAP classification (0-1 loss)
post_unnorm = px_given .* priors; % For each sample x, calculate the unnormalized posterior
[~, pred_map] = max(post_unnorm, [], 2); % MAP Decision
conf_map = zeros(K,K); % Count confusion matrix
for j=1:K
    idx = find(labels==j);
    for i=1:K
        conf_map(i,j) = sum(pred_map(idx)==i) / numel(idx);
    end
end

error_rate_map = mean(pred_map ~= labels);
figure;

% Make sure it is a column vector
pred_map = pred_map(:);
labels = labels(:);
N = length(labels);

colors = repmat([0 1 0], N, 1);   % All points are initially green
incorrect_idx = (pred_map ~= labels);

% Print debug information
fprintf('Total samples: %d | incorrect: %d\n', N, sum(incorrect_idx));

% Coloring error samples
colors(incorrect_idx, :) = ones(sum(incorrect_idx),1) * [1 0 0];

markers = {'o','s','^','d'};
hold on;
for j=1:K
    idx = find(labels==j);
    scatter(X(idx,1), X(idx,2), 12, colors(idx,:), markers{j}, 'filled');
end
title('Part A: MAP classification (green=correct, red=incorrect)');
xlabel('x1'); ylabel('x2'); grid on; hold off;
saveas(gcf, 'partA_scatter_map_matlab.png');

figure;
heatmap(1:K, 1:K, conf_map, 'CellLabelFormat','%.3f');
xlabel('True Label L=j'); ylabel('Decision D=i');
title('Confusion Matrix P(D=i|L=j) - MAP');
saveas(gcf, 'partA_confusion_map_matlab.png');

%% Part B: ERM classification with loss matrix
Lambda = [0 10 10 100; 1 0 10 100; 1 1 0 100; 1 1 1 0]; % Use the loss matrix Λ given in the question

posterior = post_unnorm ./ sum(post_unnorm, 2); % Calculate the posterior probability P(L=j|x) for a given sample x
R = posterior * Lambda.'; % expected losses per decision
[~, pred_erm] = min(R, [], 2);

conf_erm = zeros(K,K);
for j=1:K
    idx = find(labels==j);
    for i=1:K
        conf_erm(i,j) = sum(pred_erm(idx)==i) / numel(idx);
    end
end % Estimate empirical risk (average loss) using the sample mean

% Empirical average risks
losses_map = arrayfun(@(n)Lambda(pred_map(n), labels(n)), 1:N);
losses_erm = arrayfun(@(n)Lambda(pred_erm(n), labels(n)), 1:N);
emp_risk_map = mean(losses_map);
emp_risk_erm = mean(losses_erm);

figure;

pred_map = pred_map(:);
labels = labels(:);
N = length(labels);

colors = repmat([0 1 0], N, 1);   
incorrect_idx = (pred_map ~= labels);

fprintf('Total samples: %d | incorrect: %d\n', N, sum(incorrect_idx));

colors(incorrect_idx, :) = ones(sum(incorrect_idx),1) * [1 0 0];

hold on;
for j=1:K
    idx = find(labels==j);
    scatter(X(idx,1), X(idx,2), 12, colors(idx,:), markers{j}, 'filled');
end
title('Part B: ERM classification (green=correct, red=incorrect)');
xlabel('x1'); ylabel('x2'); grid on; hold off;
saveas(gcf, 'partB_scatter_erm_matlab.png');

figure;
heatmap(1:K, 1:K, conf_erm, 'CellLabelFormat','%.3f');
xlabel('True Label L=j'); ylabel('Decision D=i');
title('Confusion Matrix P(D=i|L=j) - ERM');
saveas(gcf, 'partB_confusion_erm_matlab.png');

fprintf('MAP error rate = %.4f\n', error_rate_map);
fprintf('MAP empirical risk (Lambda) = %.4f\n', emp_risk_map);
fprintf('ERM empirical risk = %.4f\n', emp_risk_erm);
