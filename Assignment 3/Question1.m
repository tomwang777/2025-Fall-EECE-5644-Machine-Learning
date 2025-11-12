%% EECE5644 - Assignment 3  Question 1
% MLP Classifier
% Cross-Validation for Multi-class Classification
clear; clc; close all;


%% Define Data Distribution Parameters
rng(1); % Reproducibility

% Number of classes and dimensions
C = 4;
n = 3; % 3-dimensional data
priors = ones(1, C) / C; % Uniform priors

% Define mean vectors for each class
mu(:, 1) = [0; 0; 0];
mu(:, 2) = [3; 3; 0];
mu(:, 3) = [0; 3; 3];
mu(:, 4) = [3; 0; 3];

% Adjust the parameters to define covariance matrices by adding some overlap for 10-20% error
Sigma(:, :, 1) = [1.2, 0.3, 0.1; 0.3, 1.2, 0.2; 0.1, 0.2, 1.2];
Sigma(:, :, 2) = [1.3, -0.2, 0.15; -0.2, 1.3, 0.1; 0.15, 0.1, 1.3];
Sigma(:, :, 3) = [1.4, 0.2, -0.15; 0.2, 1.4, 0.25; -0.15, 0.25, 1.4];
Sigma(:, :, 4) = [1.3, 0.15, 0.2; 0.15, 1.3, -0.1; 0.2, -0.1, 1.3];

fprintf('Data Distribution Defined:\n');
fprintf('- Classes: %d\n', C);
fprintf('- Dimensions: %d\n', n);
fprintf('- Priors: Uniform (0.25 each)\n\n');

%% Generate Datasets
N_train = [100, 500, 1000, 5000, 10000];
N_test = 100000;


% Generate test dataset
[X_test, y_test] = generateData(N_test, priors, mu, Sigma);

% Generate training dataset
X_train = cell(length(N_train), 1);
y_train = cell(length(N_train), 1);
for i = 1:length(N_train)
    [X_train{i}, y_train{i}] = generateData(N_train(i), priors, mu, Sigma);
    fprintf('  Training set %d: %d samples\n', i, N_train(i));
end
fprintf('  Test set: %d samples\n\n', N_test);

%% Theoretically Optimal Classifier
P_error_optimal = evaluateOptimalClassifier(X_test, y_test, priors, mu, Sigma); % MAP with True PDF
fprintf(' Theoretical Optimal P(error) = %.4f (%.2f%%)\n\n', P_error_optimal, P_error_optimal*100);

%% MLP Training with Cross-Validation
P_options = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30];
K_folds = 10; % Use 10-fold cross-validation
num_reinit = 5; % Number of random reinitializations

P_error_mlp = zeros(length(N_train), 1);
P_optimal_selected = zeros(length(N_train), 1);
cv_results = cell(length(N_train), 1);

fprintf('Training MLPs with Cross-Validation...\n');
fprintf('Perceptron options: [%s]\n', num2str(P_options));
fprintf('K-fold: %d, Random reinitializations: %d\n\n', K_folds, num_reinit);

for i = 1:length(N_train)
    fprintf(' Training Set %d (N=%d) \n', i, N_train(i));
    
    % Cross-validation for model order selection
    cv_errors = zeros(length(P_options), 1);
    
    for p_idx = 1:length(P_options)
        P = P_options(p_idx);
        fold_errors = zeros(K_folds, 1);
        
        % Create K-fold partition
        cv_partition = cvpartition(y_train{i}, 'KFold', K_folds);
        
        for k = 1:K_folds
            % Split data
            X_tr = X_train{i}(:, cv_partition.training(k));
            y_tr = y_train{i}(cv_partition.training(k));
            X_val = X_train{i}(:, cv_partition.test(k));
            y_val = y_train{i}(cv_partition.test(k));
            
            % Train MLP
            best_net = [];
            best_perf = inf;
            
            for r = 1:num_reinit
                net = trainMLP(X_tr, y_tr, P, C);
                
                % Evaluate on training set
                y_pred_tr = predictMLP(net, X_tr);
                perf = mean(y_pred_tr ~= y_tr);
                
                if perf < best_perf
                    best_perf = perf;
                    best_net = net;
                end
            end
            
            % Evaluate on validation set
            y_pred_val = predictMLP(best_net, X_val);
            fold_errors(k) = mean(y_pred_val ~= y_val);
        end
        
        cv_errors(p_idx) = mean(fold_errors);
    end
    
    % Select best P
    [~, best_idx] = min(cv_errors);
    P_best = P_options(best_idx);
    P_optimal_selected(i) = P_best;
    
    cv_results{i}.P_options = P_options;
    cv_results{i}.cv_errors = cv_errors;
    cv_results{i}.P_best = P_best;
    
    fprintf('  Cross-validation complete. Best P = %d (CV error = %.4f)\n', ...
        P_best, cv_errors(best_idx));
    
    % Train final MLP with best P on full training set
    fprintf('  Training final MLP with P=%d...\n', P_best);
    best_net_final = [];
    best_perf_final = inf;
    
    for r = 1:num_reinit
        net = trainMLP(X_train{i}, y_train{i}, P_best, C);
        
        % Evaluate on training set
        y_pred_tr = predictMLP(net, X_train{i});
        perf = mean(y_pred_tr ~= y_train{i});
        
        if perf < best_perf_final
            best_perf_final = perf;
            best_net_final = net;
        end
    end
    
    % Evaluate on test set
    y_pred_test = predictMLP(best_net_final, X_test);
    P_error_mlp(i) = mean(y_pred_test ~= y_test);
    
    fprintf('  Test P(error) = %.4f (%.2f%%)\n\n', P_error_mlp(i), P_error_mlp(i)*100);
end

%% Results Visualization
fprintf('Final Results \n\n');
fprintf('Theoretical Optimal P(error): %.4f (%.2f%%)\n\n', P_error_optimal, P_error_optimal*100);
fprintf('MLP Results:\n');
fprintf('%-15s %-15s %-20s\n', 'N_train', 'P_optimal', 'Test P(error)');
fprintf('%-15s %-15s %-20s\n', '-------', '---------', '-------------');
for i = 1:length(N_train)
    fprintf('%-15d %-15d %.4f (%.2f%%)\n', N_train(i), P_optimal_selected(i), ...
        P_error_mlp(i), P_error_mlp(i)*100);
end

% Plot 1: Test Error vs Training Set Size
figure('Position', [100, 100, 900, 600]);
semilogx(N_train, P_error_mlp * 100, 'bo-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
hold on;
semilogx(N_train, P_error_optimal * 100 * ones(size(N_train)), 'r--', 'LineWidth', 2);
grid on;
xlabel('Number of Training Samples', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Probability of Error (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('MLP Classification Performance vs Training Set Size', 'FontSize', 14, 'FontWeight', 'bold');
legend('MLP Classifier', 'Theoretical Optimal', 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 11);

% Plot 2: Selected Number of Perceptrons
figure('Position', [150, 150, 900, 600]);
semilogx(N_train, P_optimal_selected, 'gs-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'g');
grid on;
xlabel('Number of Training Samples', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Optimal Number of Perceptrons (P)', 'FontSize', 12, 'FontWeight', 'bold');
title('Model Complexity Selection via Cross-Validation', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

% Plot 3: Cross-Validation Curves for each training set
figure('Position', [200, 200, 1200, 800]);
for i = 1:length(N_train)
    subplot(2, 3, i);
    plot(cv_results{i}.P_options, cv_results{i}.cv_errors * 100, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    plot(cv_results{i}.P_best, cv_results{i}.cv_errors(cv_results{i}.P_options == cv_results{i}.P_best) * 100, ...
        'r*', 'MarkerSize', 15, 'LineWidth', 2);
    grid on;
    xlabel('Number of Perceptrons (P)', 'FontSize', 10);
    ylabel('CV Error (%)', 'FontSize', 10);
    title(sprintf('N_{train} = %d', N_train(i)), 'FontSize', 11, 'FontWeight', 'bold');
    legend('CV Error', sprintf('Best P=%d', cv_results{i}.P_best), 'Location', 'best');
end
sgtitle('Cross-Validation Results for Model Order Selection', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('\nAll plots generated successfully!\n');

%% Functions

function [X, y] = generateData(N, priors, mu, Sigma)
    % Generate multi-class Gaussian data
    C = length(priors);
    n = size(mu, 1);
    
    % Determine number of samples per class
    N_per_class = mnrnd(N, priors);
    
    X = zeros(n, N);
    y = zeros(1, N);
    
    idx = 1;
    for c = 1:C
        N_c = N_per_class(c);
        X(:, idx:idx+N_c-1) = mvnrnd(mu(:, c), Sigma(:, :, c), N_c)';
        y(idx:idx+N_c-1) = c;
        idx = idx + N_c;
    end
    
    perm = randperm(N);
    X = X(:, perm);
    y = y(perm);
end

function P_error = evaluateOptimalClassifier(X, y_true, priors, mu, Sigma)
    % MAP classifier
    N = size(X, 2);
    C = length(priors);
    
    % Compute class posteriors
    posteriors = zeros(C, N);
    for c = 1:C
        posteriors(c, :) = priors(c) * mvnpdf(X', mu(:, c)', Sigma(:, :, c))';
    end
    
    % Normalization
    posteriors = posteriors ./ sum(posteriors, 1);
    
    % MAP decision
    [~, y_pred] = max(posteriors, [], 1);
    
    % Compute error
    P_error = mean(y_pred ~= y_true);
end

function net = trainMLP(X, y, P, C)
    % Train MLP with one hidden layer
    % X: features (n x N)
    % y: labels (1 x N)
    % P: number of hidden perceptrons
    % C: number of classes
    
    % Prepare data for neural network
    y_onehot = full(ind2vec(y, C));
    
    % Create network
    net = patternnet(P, 'trainscg'); 
    
    % Configure network
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.trainParam.epochs = 300;
    net.trainParam.max_fail = 20;
    
    net.layers{1}.transferFcn = 'poslin'; % Smooth activation with RELU
    net.layers{2}.transferFcn = 'softmax'; % Softmax output
    
    % Disable validation during training
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    
    % Train
    net = train(net, X, y_onehot);
end

function y_pred = predictMLP(net, X)
    % Predict class labels
    outputs = net(X);
    [~, y_pred] = max(outputs, [], 1);

end
