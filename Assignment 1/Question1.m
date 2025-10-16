%% EECE5644 - Assignment 1 Question 1
% True Bayes, Naive Bayes (identity covariance), Fisher LDA
% ROC curves and empirical min error computation
clear; close all; clc;
rng(0); 

%% Parameters from the question
N = 10000;p0 = 0.65;p1 = 0.35;

m0 = [-0.5; -0.5; -0.5];
C0 = [1.0, -0.5, 0.3;
     -0.5, 1.0, -0.5;
      0.3, -0.5, 1.0];

m1 = [1.0; 1.0; 1.0];
C1 = [1.0,  0.3, -0.2;
       0.3, 1.0,  0.3;
      -0.2, 0.3,  1.0];

%% Generate class labels
u = rand(1, N)>= p0;
y = double(u >= p0);  
% 0 or 1 labels
N0 = sum(y==0);
N1 = sum(y==1);

fprintf('Generated N0 = %d, N1 = %d\n', N0, N1);  % Category prior

%% Generate samples
X = zeros(3, N);
idx0 = find(y==0);
idx1 = find(y==1);
X(:,idx0) = mvnrnd(m0, C0, N0)'; % Corresponding to N0 Data Generation
X(:,idx1) = mvnrnd(m1, C1, N1)'; % Corresponding to N1 Data Generation

%% Multivariate Gaussian pdf
mvn_pdf = @(X,mu,Sigma) ...
    exp(-0.5*sum((Sigma\ (X - mu)).*(X - mu),1)) / ...
    ((2*pi)^(size(X,1)/2)*sqrt(det(Sigma)));

%% Compute true conditional pdfs
p_x_given_0 = zeros(1,N);
p_x_given_1 = zeros(1,N);
for i=1:N
    p_x_given_0(i) = mvn_pdf(X(:,i), m0, C0);
    p_x_given_1(i) = mvn_pdf(X(:,i), m1, C1);
end

%% True Bayes scores
scores_true = p_x_given_1 ./ (p_x_given_0 + eps);


%% Part A: True Bayes classifier
res_true = compute_roc_and_min_error(scores_true, y, p0, p1);
thr_min_true = res_true.thr(res_true.idx_min);
fprintf('\nTrue Bayes:\n');
fprintf('  min empirical error = %.6f, threshold = %.6f\n', res_true.min_error, thr_min_true);
fprintf('  TPR = %.4f, FPR = %.4f\n', res_true.TPR(res_true.idx_min), res_true.FPR(res_true.idx_min));

gamma_theory = p0 / p1; % Likelihood ratio test
pred_theory = scores_true >= gamma_theory;
TP_theory = sum(pred_theory==1 & y==1) / sum(y==1);
FP_theory = sum(pred_theory==1 & y==0) / sum(y==0);  % Get TPR&FPR to show ROC curve
P_error_theory = FP_theory*p0 + (1-TP_theory)*p1;
fprintf('  theoretical gamma = %.4f, empirical error at gamma = %.6f\n', gamma_theory, P_error_theory);

%% Part B: Naive Bayes (identity covariance)
I3 = eye(3);
p0_nb = zeros(1,N); p1_nb = zeros(1,N);
for i=1:N
    p0_nb(i) = mvn_pdf(X(:,i), m0, I3);
    p1_nb(i) = mvn_pdf(X(:,i), m1, I3);
end
scores_nb = p1_nb ./ (p0_nb + eps);
res_nb = compute_roc_and_min_error(scores_nb, y, p0, p1);  % Examining the impact of model mismatch
fprintf('\nNaive Bayes (I):\n');
fprintf('  min empirical error = %.6f, threshold = %.6f\n', res_nb.min_error, res_nb.thr(res_nb.idx_min));

%% Part C: Fisher LDA
X0 = X(:,y==0); X1 = X(:,y==1);
m0_hat = mean(X0,2);
m1_hat = mean(X1,2); % Data estimation class mean
Sigma0_hat = cov(X0');
Sigma1_hat = cov(X1'); % Within-class covariance
S_w = Sigma0_hat + Sigma1_hat; % Equal weight
w_lda = S_w \ (m1_hat - m0_hat); % Fisher projection vector
scores_lda = w_lda' * X; % Projection score
res_lda = compute_roc_and_min_error(scores_lda, y, p0, p1);
fprintf('\nFisher LDA:\n');
fprintf('  min empirical error = %.6f, threshold = %.6f\n', res_lda.min_error, res_lda.thr(res_lda.idx_min));

%% Plot ROC curves
figure('Position',[100 100 1800 600]); 

subplot(1,3,1);
plot(res_true.FPR, res_true.TPR, 'b-', 'LineWidth', 1.5); hold on;
scatter(res_true.FPR(res_true.idx_min), res_true.TPR(res_true.idx_min), 60, 'r', 'filled');
title('True Bayes');
xlabel('FPR'); ylabel('TPR');
grid on; axis square; % True Bayes ROC Curve

subplot(1,3,2);
plot(res_nb.FPR, res_nb.TPR, 'g-', 'LineWidth', 1.5); hold on;
scatter(res_nb.FPR(res_nb.idx_min), res_nb.TPR(res_nb.idx_min), 60, 'r', 'filled');
title('Naive Bayes');
xlabel('FPR'); ylabel('TPR');
grid on; axis square;  % Naive Bayes ROC Curve

subplot(1,3,3);
plot(res_lda.FPR, res_lda.TPR, 'm-', 'LineWidth', 1.5); hold on;
scatter(res_lda.FPR(res_lda.idx_min), res_lda.TPR(res_lda.idx_min), 60, 'r', 'filled');
title('Fisher LDA');
xlabel('FPR'); ylabel('TPR');
grid on; axis square;  % Fisher LDA ROC Curve

sgtitle('ROC Curves Comparison');
saveas(gcf, 'roc_all_subplots.png');% Comparison summary chart


%% Function to compute ROC and empirical min error
function result = compute_roc_and_min_error(scores, y, p0, p1)
    [scores_sorted, ~] = sort(scores, 'descend');
    thresholds = [max(scores)+1, scores_sorted, min(scores)-1];
    n_thr = length(thresholds);
    TPR = zeros(1, n_thr);
    FPR = zeros(1, n_thr);
    for i=1:n_thr
        pred = scores >= thresholds(i);
        TP = sum(pred==1 & y==1);
        FP = sum(pred==1 & y==0);
        TPR(i) = TP / sum(y==1);
        FPR(i) = FP / sum(y==0);
    end
    P_error = FPR*p0 + (1-TPR)*p1;
    [min_error, idx_min] = min(P_error);
    result.thr = thresholds;
    result.TPR = TPR;
    result.FPR = FPR;
    result.P_error = P_error;
    result.idx_min = idx_min;
    result.min_error = min_error;
end
