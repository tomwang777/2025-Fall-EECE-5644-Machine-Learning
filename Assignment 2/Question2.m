%% EECE5644 - Assignment 2 Question 2
% ML Estimator
% MLP Estimator
clear; clc; close all;
rng(1);
%% Generate data
N_train = 1000; % Choose appropriate N for training
N_validate = 5000; % Choose appropriate N for validation
[xTrain, yTrain, xValidate, yValidate] = hw2q2(N_train, N_validate); 

% Transpose y vectors to be N x 1
yTrain = yTrain';
yValidate = yValidate';

%% Construct design matrix
Z_train = feature_map_cubic(xTrain);
Z_validate = feature_map_cubic(xValidate);

%% ML Estimator (Compute once)
w_ml = ML_estimator(Z_train, yTrain);

%% MAP Estimator Experiment (Iterate over gamma)
m = 4; % 10^-m
n = 4; % 10^n
gamma_values = logspace(-m, n, 50); % 50 points log-spaced
mse_map = zeros(size(gamma_values)); 

for i = 1:length(gamma_values)
    gamma_param = gamma_values(i);
    
    % Train MAP model
    w_map = MAP_estimator(Z_train, yTrain, gamma_param);
    
    % Evaluate on Validation Set
    y_pred = Z_validate * w_map;
    
    % Calculate Mean Squared Error (MSE)
    errors = yValidate - y_pred;
    mse_map(i) = mean(errors.^2);
end

%% ML Model Evaluation (For comparison)
y_pred_ml = Z_validate * w_ml;
mse_ml = mean((yValidate - y_pred_ml).^2);
fprintf('ML Model Validation MSE: %.4f\n', mse_ml);
fprintf('MAP Model Validation MSE: %.4f\n', mse_map);

%% Visualization and Analysis
figure(3);
loglog(gamma_values, mse_map, 'b-'); % Use log-log scale for regularization analysis
hold on;
loglog(gamma_values, ones(size(gamma_values)) * mse_ml, 'r--'); % Plot ML MSE as baseline
xlabel('\gamma (Hyperparameter)');
ylabel('Average-Squared Error on Validation Set');
title('MAP Performance vs. Regularization Hyperparameter \gamma');
legend('MAP Model MSE', 'ML Model MSE');
grid on;
%% Key Fuction Realization
function Z = feature_map_cubic(X)
% X is a 2 x N matrix (x1 and x2 features, N samples)
% Z is an N x 10 design matrix

% Check if X is 2 x N or N x 2, transpose if necessary (assuming data is 2 x N as in hw2q2.m)
if size(X, 1) ~= 2
    X = X'; % Transpose to make it 2 x N
end

N = size(X, 2);
x1 = X(1, :);
x2 = X(2, :);

% 10 Features: [1, x1, x2, x1^2, x1x2, x2^2, x1^3, x1^2x2, x1x2^2, x2^3]
Z = [
    ones(1, N); % 1. Bias
    x1; % 2. x1
    x2; % 3. x2
    x1.^2; % 4. x1^2
    x1 .* x2; % 5. x1*x2
    x2.^2; % 6. x2^2
    x1.^3; % 7. x1^3
    (x1.^2) .* x2; % 8. x1^2*x2
    x1 .* (x2.^2); % 9. x1*x2^2
    x2.^3; % 10. x2^3
];

% Transpose Z to be N x 10 for standard regression form
Z = Z'; 
end
%% Estimator Realization Function
function w_ml = ML_estimator(Z_train, y_train)
% ML estimator
%   Z_train: N x d design matrix
%   y_train: N x 1 output vector

    % Closed-form MLE solution
    w_ml = (Z_train' * Z_train) \ (Z_train' * y_train);
end

function w_map = MAP_estimator(Z_train, y_train, gamma_param)
% MAP estimator (Bayesian Ridge Regression)
%   gamma_param: regularization hyperparameter
%   Equivalent to prior variance = gamma^-1

    d = size(Z_train, 2);
    I = eye(d);

    % Closed-form MAP solution
    w_map = (Z_train' * Z_train + (1/gamma_param) * I) \ (Z_train' * y_train);
end
