%% EECE5644 - Assignment 2 Question 1
% Bayesian optimization
% Maximum likelihood parameter estimation technique
clear; clc; close all;
rng(1);

%% Parameters
P = [0.6, 0.4];
C = [0.75 0; 0 1.25];
m01 = [-0.9; -1.1];
m02 = [0.8; 0.75];
m11 = [-1.1; 0.9];
m12 = [0.9; -0.75];

N_train = [50, 500, 5000];
N_val = 10000;

%% Generate datasets
[X_val, L_val] = generate_data(N_val, P, m01, m02, m11, m12, C);

%% Part 1: True Bayes Classifier
p_x_L0 = 0.5*gpdf(X_val,m01,C) + 0.5*gpdf(X_val,m02,C);
p_x_L1 = 0.5*gpdf(X_val,m11,C) + 0.5*gpdf(X_val,m12,C);

post1 = P(2)*p_x_L1 ./ (P(1)*p_x_L0 + P(2)*p_x_L1);
decision = post1 > 0.5;

% Error estimate
P_error = mean(decision ~= L_val);

fprintf('Min-P(error) (Bayes) ≈ %.4f\n', P_error);

% ROC curve
thresholds = linspace(0,1,200);
TPR = zeros(size(thresholds)); FPR = zeros(size(thresholds));
for i=1:length(thresholds)
    t = thresholds(i);
    pred = post1 > t;
    TP = sum(pred==1 & L_val==1);
    FP = sum(pred==1 & L_val==0);
    FN = sum(pred==0 & L_val==1);
    TN = sum(pred==0 & L_val==0);
    TPR(i) = TP/(TP+FN);
    FPR(i) = FP/(FP+TN);
end

figure;
plot(FPR,TPR,'b-','LineWidth',1.5); hold on;
[~,idx_min] = min(abs(thresholds-0.5));
scatter(FPR(idx_min),TPR(idx_min),60,'r','filled');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('Bayes Classifier ROC Curve');
grid on; axis square;
saveas(gcf,'Q1_ROC_Bayes.png');

%% Optional: Decision Boundary Visualization
[x1,x2] = meshgrid(linspace(-3,3,200), linspace(-3,3,200));
gridX = [x1(:), x2(:)];
p0 = 0.5*gpdf(gridX,m01,C)+0.5*gpdf(gridX,m02,C);
p1 = 0.5*gpdf(gridX,m11,C)+0.5*gpdf(gridX,m12,C);
post = P(2)*p1 ./ (P(1)*p0 + P(2)*p1);
decision_map = reshape(post>0.5,size(x1));

figure;
gscatter(X_val(:,1),X_val(:,2),L_val,'br','ox'); hold on;
contour(x1,x2,decision_map,[0.5 0.5],'k','LineWidth',2);
title('Bayes Decision Boundary (Validation Data)');
xlabel('x_1'); ylabel('x_2');
axis equal; grid on;
saveas(gcf,'Q1_BayesBoundary.png');
%% PART 2: Logistic Regression
fprintf('\n--- PART 2: Logistic Regression ---\n');
for N = [50, 500, 5000]
    fprintf('\nTraining size = %d\n', N);
    [X_train, L_train] = generate_data(N, P, m01, m02, m11, m12, C);
    % linear
    w_lin = train_logistic(X_train, L_train, 'linear');
    err_lin = evaluate_logistic(X_val, L_val, w_lin, 'linear');
    % quadratic
    w_quad = train_logistic(X_train, L_train, 'quadratic');
    err_quad = evaluate_logistic(X_val, L_val, w_quad, 'quadratic');
    fprintf('Linear logistic error = %.3f | Quadratic logistic error = %.3f\n', ...
            err_lin, err_quad);
end
%% Training
Z = feature_map(X_train, 'linear');  % using mode 'linear' or 'quadratic'
y = L_train;

opts = optimset('MaxIter', 1e3, 'Display','off');
w_init = zeros(size(Z,2),1);
[w_opt, fval] = fminsearch(@(w) nll_loss(w,Z,y), w_init, opts);
%% Prediction and error rate
Z_val = feature_map(X_val, 'linear');
activation = Z_val * w_opt;
prob = 1 ./ (1 + exp(-activation));
pred = prob > 0.5;
err = mean(pred ~= L_val);
fprintf('Validation error (linear) = %.3f\n', err);
%% Decision Boundary Visualization
[x1,x2] = meshgrid(linspace(-3,3,200), linspace(-3,3,200));
gridX = [x1(:), x2(:)];
Zg = feature_map(gridX, 'linear');
activatione = Zg * w_opt;
prob = 1 ./ (1 + exp(-activatione));
contour(x1,x2,reshape(post>0.5,size(x1)),[1 1]*0.5,'k','LineWidth',2);
hold off; 
gcf; 
saveas(gcf, 'decision_boundary.png');
%% Function for Gaussian PDF
function p = gpdf(x, m, C)
%   x: N x d matrix (N samples)
%   m: 1 x d or d x 1 mean vector
%   C: d x d covariance matrix
%   p: N x 1 vector of pdf values

    if iscolumn(m)
        m = m';    % convert column mean to row
    end
    [N, d] = size(x);
    % ensure m is 1xd
    if numel(m) ~= d
        error('Mean dimension does not match data dimension');
    end

    % subtract mean (vectorized)
    xc = bsxfun(@minus, x, m);   % N x d

    invC = inv(C);
    detC = det(C);
    if detC <= 0
        error('Covariance matrix must be positive definite');
    end

    quad = sum((xc * invC) .* xc, 2);   % N x 1
    const = 1 / ((2*pi)^(d/2) * sqrt(detC));
    p = const * exp(-0.5 * quad);
end
%% Function to sample data
function [X, L] = generate_data(N, P, m01, m02, m11, m12, C)
    L = rand(N,1) > P(1); % class label (0/1)
    X = zeros(N,2);
    for i = 1:N
        if L(i)==0
            if rand<0.5
                X(i,:) = mvnrnd(m01, C);
            else
                X(i,:) = mvnrnd(m02, C);
            end
        else
            if rand<0.5
                X(i,:) = mvnrnd(m11, C);
            else
                X(i,:) = mvnrnd(m12, C);
            end
        end
    end
end

%% Feature Extension Function
function Z = feature_map(X, mode)  % using mode = 'linear' or 'quadratic'
    if strcmp(mode,'linear')
        Z = [ones(size(X,1),1), X];  % [1, x1, x2]
    elseif strcmp(mode,'quadratic')
        x1 = X(:,1); x2 = X(:,2);
        Z = [ones(size(X,1),1), x1, x2, x1.^2, x1.*x2, x2.^2];
    else
        error('Unknown mode');
    end
end
%% Sigmoid and Negative Log-Likelihood
function J = nll_loss(w,Z,y)
    h = 1./(1+exp(-Z*w));
    eps_ = 1e-8;  % To avoid log(0)
    J = -sum(y.*log(h+eps_) + (1-y).*log(1-h+eps_));
end
%% Training Function
function w = train_logistic(X, L, mode)
    Z = feature_map(X, mode);
    w0 = zeros(size(Z,2),1);
    opts = optimset('MaxIter',1000,'Display','off');
    w = fminsearch(@(w)nll_loss(w,Z,L), w0, opts);
end
%% Evaluation Function
function err = evaluate_logistic(X_val, L_val, w, mode)
    Z_val = feature_map(X_val, mode);
    prob = 1./(1+exp(-Z_val*w));
    pred = prob>0.5;
    err = mean(pred~=L_val);
end




