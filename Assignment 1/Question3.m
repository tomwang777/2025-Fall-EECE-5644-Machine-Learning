%% EECE5644 - Assignment 1 Question 3
% Gaussian Classifiers
clear; close all; clc;

%% Parameters
data_dir = './data';
% Choose regularization factor scaling alpha (0 < alpha < 1)
alpha = 1e-3;    

%% Wine Quality
red_file = fullfile(data_dir,'winequality-red.csv');
white_file = fullfile(data_dir,'winequality-white.csv');

if ~exist(red_file,'file') || ~exist(white_file,'file')
    error('Please download winequality-red.csv and winequality-white.csv into ./data/ first.');
end

% read CSVs
opts = detectImportOptions(red_file,'Delimiter',';');
T_red = readtable(red_file, opts);
opts = detectImportOptions(white_file,'Delimiter',';');
T_white = readtable(white_file, opts);

% Combine red and white datasets into one dataset and use "quality" (0-10) as label
T_all = [T_red; T_white];
X_wine = table2array(T_all(:,1:end-1));   % features (N x d)
y_wine = table2array(T_all(:,end));       % labels (N x 1)

% Normalize features (z-score) before Gaussian fitting
mu_feat = mean(X_wine);
std_feat = std(X_wine);
Xw = (X_wine - mu_feat) ./ (std_feat + eps);

% Unique classes in wine (quality scores)
classes_w = unique(y_wine);
K_w = numel(classes_w);
N_w = size(Xw,1);
d_w = size(Xw,2);

% Estimate class priors, means, covariances
priors_w = zeros(K_w,1);
means_w = zeros(K_w,d_w);
covs_w = zeros(d_w,d_w,K_w);
for k=1:K_w
    idx = (y_wine == classes_w(k));
    priors_w(k) = sum(idx)/N_w;
    Xk = Xw(idx,:);
    means_w(k,:) = mean(Xk,1);
    covs_w(:,:,k) = cov(Xk,1);   % population cov with normalization by N (matlab cov with flag 1 gives normalization by N)
    s = svd(covs_w(:,:,k));    % ensure numerical stability & regularize
    rk = sum(s > 1e-12);
    if rk == 0
        lambda_k = alpha * mean(s); % compute suggested lambda per hint: lambda = alpha * trace(C)/rank(C)
    else
        lambda_k = alpha * sum(s)/max(1,rk);
    end
    covs_w(:,:,k) = covs_w(:,:,k) + lambda_k * eye(d_w);
end

% Use log-likelihood form to classify all training samples via MAP
logpost_w = zeros(N_w, K_w);
for k=1:K_w
    mu_k = means_w(k,:);
    Sigma_k = covs_w(:,:,k);      % use log Gaussian
    L = chol(Sigma_k,'lower');    % lower triangular
    invL = L'\eye(d_w);
    Sigma_inv = invL*invL';
    logdetSigma = 2*sum(log(diag(L)));
    diff = bsxfun(@minus, Xw, mu_k); % N x d
    quad = sum((diff * Sigma_inv) .* diff, 2); % N x 1
    loglike = -0.5*( quad + logdetSigma + d_w*log(2*pi) );
    logpost_w(:,k) = log(priors_w(k) + eps) + loglike;
end

[~, pred_w_idx] = max(logpost_w, [], 2);
pred_w = classes_w(pred_w_idx);

% Confusion matrix and error prob
C_w = confusionmat(y_wine, pred_w, 'Order', classes_w);
% convert to P(D=i | L=j) style: columns true label j, rows decision i
conf_prob_w = zeros(K_w,K_w);
for j=1:K_w
    idx = (y_wine == classes_w(j));
    if sum(idx)>0
        for i=1:K_w
            conf_prob_w(i,j) = sum(pred_w(idx) == classes_w(i)) / sum(idx);
        end
    end
end
err_w = mean(pred_w ~= y_wine);

fprintf('\\nWine dataset (all data) results:\\n');
fprintf('N = %d, d = %d, #classes = %d\\n', N_w, d_w, K_w);
fprintf('Training MAP error = %.4f\\n', err_w);

% Save confusion heatmap and a PCA scatter
figure('visible','off'); imagesc(conf_prob_w); colorbar; colormap('parula');
xticks(1:K_w); yticks(1:K_w);
xticklabels(cellstr(num2str(classes_w))); yticklabels(cellstr(num2str(classes_w)));
xlabel('True label (quality)'); ylabel('Decision');
title('Wine: Confusion P(D=i | L=j) - MAP (columns true label)');
saveas(gcf,'wine_confusion_map.png');

% PCA visualization
[coeff,score,~] = pca(Xw);
pc2d = score(:,1:2);
figure('visible','off'); gscatter(pc2d(:,1),pc2d(:,2),y_wine);
xlabel('PC1'); ylabel('PC2'); title('Wine: PCA projection (PC1 vs PC2) colored by quality'); legend('Location','bestoutside');
saveas(gcf,'wine_pca_2d.png');

figure('visible','off'); scatter(pc2d(:,1), pc2d(:,2), 6, double(pred_w == y_wine)); colormap([1 0 0; 0 1 0]); colorbar; title('Wine: green correct / red incorrect'); saveas(gcf,'wine_pca_correctness.png');

%% Part II: HAR dataset
har_file = fullfile(data_dir,'HAR_all.csv');
if ~exist(har_file,'file')
    error('Please place HAR_all.csv (561 features + label column) into ./data/');
end
Thar = readmatrix(har_file);  % assume numeric matrix
X_har = Thar(:,1:end-1);
y_har = Thar(:,end);
classes_h = unique(y_har);
K_h = numel(classes_h);
N_h = size(X_har,1);
d_h = size(X_har,2);

% Standardize features per column
mu_h = mean(X_har,1); sd_h = std(X_har,[],1);
Xh = bsxfun(@rdivide, bsxfun(@minus, X_har, mu_h), sd_h + eps);

% Estimate priors, means, covs per class for regularization
priors_h = zeros(K_h,1);
means_h = zeros(K_h,d_h);
covs_h = zeros(d_h,d_h,K_h);
for k=1:K_h
    idx = (y_har == classes_h(k));
    priors_h(k) = sum(idx)/N_h;
    Xk = Xh(idx,:);
    means_h(k,:) = mean(Xk,1);
    % Use N-1 normalization via cov to generate sample covariance
    Ck = cov(Xk); 
    s = svd(Ck);
    rk = sum(s > 1e-12);
    if rk == 0
        lambda_k = alpha * mean(s);
    else
        lambda_k = alpha * sum(s)/max(1,rk);
    end
    covs_h(:,:,k) = Ck + lambda_k * eye(d_h);
end

% Compute log posteriors for HAR
logpost_h = zeros(N_h, K_h);
for k=1:K_h
    mu_k = means_h(k,:);
    Sigma_k = covs_h(:,:,k);
    % Use pseudo-inverse / chol if possible
    % Try chol; Use pinv with regularization if fails
    try
        L = chol(Sigma_k,'lower');
        invL = L'\eye(d_h);
        Sigma_inv = invL*invL';
        logdetSigma = 2*sum(log(diag(L)));
    catch
        Sigma_inv = pinv(Sigma_k);
        logdetSigma = log(det(Sigma_k) + eps);
    end
    diff = bsxfun(@minus, Xh, mu_k);
    quad = sum((diff * Sigma_inv) .* diff, 2);
    loglike = -0.5*( quad + logdetSigma + d_h*log(2*pi) );
    logpost_h(:,k) = log(priors_h(k) + eps) + loglike;
end

[~, pred_h_idx] = max(logpost_h, [], 2);
pred_h = classes_h(pred_h_idx);

% Confusion and error
C_h = confusionmat(y_har, pred_h, 'Order', classes_h);
conf_prob_h = zeros(K_h,K_h);
for j=1:K_h
    idx = (y_har == classes_h(j));
    if sum(idx)>0
        for i=1:K_h
            conf_prob_h(i,j) = sum(pred_h(idx) == classes_h(i)) / sum(idx);
        end
    end
end
err_h = mean(pred_h ~= y_har);

fprintf('\\nHAR dataset results:\\n');
fprintf('N = %d, d = %d, #classes = %d\\n', N_h, d_h, K_h);
fprintf('Training MAP error = %.4f\\n', err_h);

figure('visible','off'); imagesc(conf_prob_h); colorbar; colormap('parula');
xticks(1:K_h); yticks(1:K_h);
xticklabels(cellstr(num2str(classes_h))); yticklabels(cellstr(num2str(classes_h)));
xlabel('True label L=j'); ylabel('Decision D=i'); title('HAR: Confusion P(D=i|L=j) - MAP'); saveas(gcf,'har_confusion_map.png');

% PCA on HAR  (Use a subset for speed to visualize per-class clusters)
[coeff_h, score_h, ~] = pca(Xh);
pc_h_2d = score_h(:,1:2);
sample_vis = randperm(N_h, min(3000,N_h)); % sample subset if large
figure('visible','off'); gscatter(pc_h_2d(sample_vis,1), pc_h_2d(sample_vis,2), y_har(sample_vis));
xlabel('PC1'); ylabel('PC2'); title('HAR: PCA (PC1 vs PC2) colored by activity label'); legend('Location','eastoutside');
saveas(gcf,'har_pca_2d.png');

% Save numeric results
save('question3_results.mat', 'err_w', 'conf_prob_w', 'classes_w', 'err_h', 'conf_prob_h', 'classes_h');

fprintf('\\nSaved results and figures into current folder.\\n');
