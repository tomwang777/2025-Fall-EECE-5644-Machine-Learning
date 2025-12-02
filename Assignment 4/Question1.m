%% EECE5644 - Assignment 4 Question 1
% SVM vs MLP Classifier Comparison with K-Fold Cross-Validation
clear; close all; clc;

%% Define Data Distribution and Generate Data
rng(42); % Reproducibility

% Parameters
r_minus1 = 2;
r_plus1 = 4;
sigma = 1;
n_train = 1000;
n_test = 10000;

fprintf('Data Generation\n');
fprintf('Class -1: r = %.1f, Class +1: r = %.1f, sigma = %.1f\n', r_minus1, r_plus1, sigma);
fprintf('Training samples: %d, Test samples: %d\n\n', n_train, n_test);

% Generate training data
[X_train, y_train] = generateData(n_train, r_minus1, r_plus1, sigma);

% Generate test data
[X_test, y_test] = generateData(n_test, r_minus1, r_plus1, sigma);

% Visualize training data
figure('Position', [100, 100, 900, 700]);
idx_neg = (y_train == -1);
idx_pos = (y_train == 1);
scatter(X_train(1, idx_neg), X_train(2, idx_neg), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
scatter(X_train(1, idx_pos), X_train(2, idx_pos), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.6);

% Plot true circular boundaries
theta = linspace(0, 2*pi, 200);
plot(r_minus1*cos(theta), r_minus1*sin(theta), 'b--', 'LineWidth', 2);
plot(r_plus1*cos(theta), r_plus1*sin(theta), 'r--', 'LineWidth', 2);

grid on;
xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
title('Training Data Distribution (Two Concentric Circles)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Class -1 (r=2)', 'Class +1 (r=4)', 'True boundary -1', 'True boundary +1', ...
    'Location', 'best', 'FontSize', 10);
axis equal;
set(gca, 'FontSize', 11);

%% Use K-Fold Cross-Validation to do the SVM Hyperparameter Selection 
fprintf('SVM Training\n');

K_folds = 10;
fprintf('K-fold cross-validation: %d folds\n', K_folds);

% Hyperparameter grid for SVM
kernel_widths = [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10]; % Gaussian Kernal
box_constraints = [0.1, 0.5, 1, 5, 10, 50, 100];

% Grid search with cross-validation
best_svm_error = inf;
best_svm_params = struct();
svm_cv_results = zeros(length(kernel_widths), length(box_constraints));

cv_partition = cvpartition(y_train, 'KFold', K_folds);

for i = 1:length(kernel_widths)
    for j = 1:length(box_constraints)
        kw = kernel_widths(i);
        bc = box_constraints(j);
        
        fold_errors = zeros(K_folds, 1);
        
        for k = 1:K_folds
            % Split data
            X_tr = X_train(:, cv_partition.training(k));
            y_tr = y_train(cv_partition.training(k));
            X_val = X_train(:, cv_partition.test(k));
            y_val = y_train(cv_partition.test(k));
            
            % Train SVM
            svm_model = fitcsvm(X_tr', y_tr', ...
                'KernelFunction', 'gaussian', ...
                'KernelScale', kw, ...
                'BoxConstraint', bc, ...
                'Standardize', false);
            
            % Predict on validation set
            y_pred = predict(svm_model, X_val');
            fold_errors(k) = mean(y_pred ~= y_val');
        end
        
        % Average error across folds
        avg_error = mean(fold_errors);
        svm_cv_results(i, j) = avg_error;
        
        % Track best parameters
        if avg_error < best_svm_error
            best_svm_error = avg_error;
            best_svm_params.kernel_width = kw;
            best_svm_params.box_constraint = bc;
        end
    end
end

fprintf('\nBest SVM Parameters (via CV):\n');
fprintf('  Kernel Width: %.2f\n', best_svm_params.kernel_width);
fprintf('  Box Constraint: %.2f\n', best_svm_params.box_constraint);
fprintf('  CV Error: %.4f (%.2f%%)\n\n', best_svm_error, best_svm_error*100);

% Train final SVM with best parameters
fprintf('Training final SVM with optimized hyperparameters...\n');
svm_final = fitcsvm(X_train', y_train', ...
    'KernelFunction', 'gaussian', ...
    'KernelScale', best_svm_params.kernel_width, ...
    'BoxConstraint', best_svm_params.box_constraint, ...
    'Standardize', false);

% Evaluate on test set
y_pred_svm = predict(svm_final, X_test');
svm_test_error = mean(y_pred_svm ~= y_test');
fprintf('SVM Test Error: %.4f (%.2f%%)\n\n', svm_test_error, svm_test_error*100);

%% MLP Hyperparameter Selection using K-Fold Cross-Validation
fprintf('MLP Training\n');

% Hyperparameter options for MLP
perceptron_options = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50];% Numbers of perceptrons
fprintf('Testing %d perceptron configurations...\n', length(perceptron_options));

% Grid search with cross-validation
best_mlp_error = inf;
best_mlp_params = struct();
mlp_cv_results = zeros(length(perceptron_options), 1);
num_reinit = 5; % Random reinitializations to avoid local optima

for i = 1:length(perceptron_options)
    P = perceptron_options(i);
    
    fold_errors = zeros(K_folds, 1);
    
    for k = 1:K_folds
        % Split data
        X_tr = X_train(:, cv_partition.training(k));
        y_tr = y_train(cv_partition.training(k));
        X_val = X_train(:, cv_partition.test(k));
        y_val = y_train(cv_partition.test(k));
        
        % Train MLP with multiple reinitializations
        best_fold_net = [];
        best_fold_perf = inf;
        
        for r = 1:num_reinit
            net = trainMLP(X_tr, y_tr, P);
            
            % Evaluate on training data
            y_pred_tr = predictMLP(net, X_tr);
            perf = mean(y_pred_tr ~= y_tr);
            
            if perf < best_fold_perf
                best_fold_perf = perf;
                best_fold_net = net;
            end
        end
        
        % Evaluate best network on validation set
        y_pred_val = predictMLP(best_fold_net, X_val);
        fold_errors(k) = mean(y_pred_val ~= y_val);
    end
    
    % Average error across folds
    avg_error = mean(fold_errors);
    mlp_cv_results(i) = avg_error;
    
    % Track best parameters
    if avg_error < best_mlp_error
        best_mlp_error = avg_error;
        best_mlp_params.num_perceptrons = P;
    end
    
    if mod(i, 3) == 0
        fprintf('  Tested %d/%d configurations...\n', i, length(perceptron_options));
    end
end

fprintf('\nBest MLP Parameters (via CV):\n');
fprintf('  Number of Perceptrons: %d\n', best_mlp_params.num_perceptrons);
fprintf('  CV Error: %.4f (%.2f%%)\n\n', best_mlp_error, best_mlp_error*100);

% Train final MLP with best parameters
fprintf('Training final MLP with optimized hyperparameters...\n');
best_mlp_final = [];
best_final_perf = inf;

for r = 1:num_reinit
    net = trainMLP(X_train, y_train, best_mlp_params.num_perceptrons);
    
    % Evaluate on training data
    y_pred_tr = predictMLP(net, X_train);
    perf = mean(y_pred_tr ~= y_train);
    
    if perf < best_final_perf
        best_final_perf = perf;
        best_mlp_final = net;
    end
end

% Evaluate on test set
y_pred_mlp = predictMLP(best_mlp_final, X_test);
mlp_test_error = mean(y_pred_mlp ~= y_test);
fprintf('MLP Test Error: %.4f (%.2f%%)\n\n', mlp_test_error, mlp_test_error*100);

%% Result
fprintf('Results\n\n');
fprintf('%-25s %-25s %-20s\n', 'Classifier', 'Best Hyperparameters', 'Test Error (%)');
fprintf('%-25s σ=%.2f, C=%.2f        %.4f (%.2f%%)\n', 'SVM (Gaussian)', ...
    best_svm_params.kernel_width, best_svm_params.box_constraint, ...
    svm_test_error, svm_test_error*100);
fprintf('%-25s P=%d                   %.4f (%.2f%%)\n', 'MLP (1 hidden layer)', ...
    best_mlp_params.num_perceptrons, mlp_test_error, mlp_test_error*100);
fprintf('\n');

%% Visualization

% Figure 2: Cross-Validation Results - SVM
figure('Position', [150, 150, 1000, 700]);
imagesc(box_constraints, kernel_widths, svm_cv_results * 100);
colorbar;
colormap(jet);
hold on;

% Mark best parameters
[~, best_i] = min(abs(kernel_widths - best_svm_params.kernel_width));
[~, best_j] = min(abs(box_constraints - best_svm_params.box_constraint));
plot(box_constraints(best_j), kernel_widths(best_i), 'w*', 'MarkerSize', 20, 'LineWidth', 3);

xlabel('Box Constraint (C)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Kernel Width (σ)', 'FontSize', 12, 'FontWeight', 'bold');
title('SVM Cross-Validation Error (%) - Hyperparameter Grid Search', ...
    'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);
set(gca, 'XScale', 'log');

% Figure 3: Cross-Validation Results - MLP
figure('Position', [200, 200, 900, 600]);
plot(perceptron_options, mlp_cv_results * 100, 'bo-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
hold on;
plot(best_mlp_params.num_perceptrons, best_mlp_error * 100, 'r*', 'MarkerSize', 20, 'LineWidth', 3);
grid on;
xlabel('Number of Perceptrons', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Cross-Validation Error (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('MLP Cross-Validation Error vs Number of Perceptrons', 'FontSize', 14, 'FontWeight', 'bold');
legend('CV Error', sprintf('Best (P=%d)', best_mlp_params.num_perceptrons), ...
    'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 11);

% Figure 4: SVM Decision Boundary on Test Data
figure('Position', [250, 250, 1200, 500]);

subplot(1, 2, 1);
plotDecisionBoundary(svm_final, X_test, y_test, 'SVM');
title(sprintf('SVM Decision Boundary (Test Error: %.2f%%)', svm_test_error*100), ...
    'FontSize', 12, 'FontWeight', 'bold');

subplot(1, 2, 2);
plotDecisionBoundaryMLP(best_mlp_final, X_test, y_test, 'MLP');
title(sprintf('MLP Decision Boundary (Test Error: %.2f%%)', mlp_test_error*100), ...
    'FontSize', 12, 'FontWeight', 'bold');

sgtitle('Classification Boundaries on Test Data', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 5: Error Distribution on Test Set
figure('Position', [300, 300, 1000, 600]);

% Compute errors for each sample
svm_correct = (y_pred_svm == y_test');
mlp_correct = (y_pred_mlp == y_test');

subplot(1, 2, 1);
scatter(X_test(1, svm_correct), X_test(2, svm_correct), 20, 'g', 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
scatter(X_test(1, ~svm_correct), X_test(2, ~svm_correct), 40, 'r', 'filled', 'MarkerFaceAlpha', 0.7);
theta = linspace(0, 2*pi, 200);
plot(r_minus1*cos(theta), r_minus1*sin(theta), 'b--', 'LineWidth', 1.5);
plot(r_plus1*cos(theta), r_plus1*sin(theta), 'r--', 'LineWidth', 1.5);
grid on;
xlabel('x_1', 'FontSize', 11);
ylabel('x_2', 'FontSize', 11);
title(sprintf('SVM: Correct (green) vs Errors (red)\nTest Error: %.2f%%', svm_test_error*100), ...
    'FontSize', 11, 'FontWeight', 'bold');
legend('Correct', 'Errors', 'True boundary', '', 'Location', 'best');
axis equal;

subplot(1, 2, 2);
scatter(X_test(1, mlp_correct), X_test(2, mlp_correct), 20, 'g', 'filled', 'MarkerFaceAlpha', 0.4);
hold on;
scatter(X_test(1, ~mlp_correct), X_test(2, ~mlp_correct), 40, 'r', 'filled', 'MarkerFaceAlpha', 0.7);
plot(r_minus1*cos(theta), r_minus1*sin(theta), 'b--', 'LineWidth', 1.5);
plot(r_plus1*cos(theta), r_plus1*sin(theta), 'r--', 'LineWidth', 1.5);
grid on;
xlabel('x_1', 'FontSize', 11);
ylabel('x_2', 'FontSize', 11);
title(sprintf('MLP: Correct (green) vs Errors (red)\nTest Error: %.2f%%', mlp_test_error*100), ...
    'FontSize', 11, 'FontWeight', 'bold');
legend('Correct', 'Errors', 'True boundary', '', 'Location', 'best');
axis equal;

% Figure 6: Comparison Bar Chart
figure('Position', [350, 350, 800, 600]);
models = {'SVM', 'MLP'};
errors = [svm_test_error, mlp_test_error] * 100;
bar(errors, 'FaceColor', [0.3 0.5 0.8]);
hold on;
text(1:2, errors, arrayfun(@(x) sprintf('%.2f%%', x), errors, 'UniformOutput', false), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Test Error (%)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', models, 'FontSize', 12);
title('Test Performance Comparison: SVM vs MLP', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
ylim([0, max(errors)*1.2]);

fprintf('All visualizations generated successfully!\n');

%% Functions

function [X, y] = generateData(N, r_minus1, r_plus1, sigma)
    % Generate data according to the distribution
    
    % Equal priors for both classes
    N_minus1 = round(N / 2);
    N_plus1 = N - N_minus1;
    
    % Class -1: r = r_minus1
    theta_minus1 = unifrnd(-pi, pi, 1, N_minus1);
    X_minus1 = r_minus1 * [cos(theta_minus1); sin(theta_minus1)] + ...
               sigma * randn(2, N_minus1);
    y_minus1 = -ones(1, N_minus1);
    
    % Class +1: r = r_plus1
    theta_plus1 = unifrnd(-pi, pi, 1, N_plus1);
    X_plus1 = r_plus1 * [cos(theta_plus1); sin(theta_plus1)] + ...
              sigma * randn(2, N_plus1);
    y_plus1 = ones(1, N_plus1);
    
    % Combine and shuffle
    X = [X_minus1, X_plus1];
    y = [y_minus1, y_plus1];
    
    perm = randperm(N);
    X = X(:, perm);
    y = y(perm);
end

function net = trainMLP(X, y, P)
    % Train MLP with one hidden layer
    % X: features (2 x N)
    % y: labels (1 x N), values in {-1, +1}
    % P: number of hidden perceptrons
    
    % Convert labels to {1, 2} for neural network
    y_nn = (y + 3) / 2; 
    y_onehot = full(ind2vec(y_nn, 2));
    
    % Create network
    net = patternnet(P, 'trainscg');
    
    % Configure network
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.trainParam.epochs = 300;
    net.trainParam.max_fail = 20;
    
    % Activation functions
    % Using tansig for hidden layer
    net.layers{1}.transferFcn = 'tansig'; 
    net.layers{2}.transferFcn = 'softmax';
    
    % Disable internal validation
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    
    % Train
    net = train(net, X, y_onehot);
end

function y_pred = predictMLP(net, X)
    % Predict class labels
    outputs = net(X);
    [~, y_pred_nn] = max(outputs, [], 1);
    
    % Convert back to {-1, +1}
    y_pred = 2 * y_pred_nn - 3; % 1 -> -1, 2 -> +1
end

function plotDecisionBoundary(svm_model, X_test, y_test, model_name)
    % Plot decision boundary for SVM
    idx_neg = (y_test == -1);
    idx_pos = (y_test == 1);
    
    % Create grid
    x1_range = linspace(min(X_test(1,:))-1, max(X_test(1,:))+1, 200);
    x2_range = linspace(min(X_test(2,:))-1, max(X_test(2,:))+1, 200);
    [X1_grid, X2_grid] = meshgrid(x1_range, x2_range);
    X_grid = [X1_grid(:), X2_grid(:)];
    
    % Predict on grid
    y_grid = predict(svm_model, X_grid);
    y_grid = reshape(y_grid, size(X1_grid));
    
    % Plot decision boundary
    contourf(X1_grid, X2_grid, y_grid, 1, 'LineWidth', 1.5);
    colormap([0.7 0.7 1; 1 0.7 0.7]);
    hold on;
    
    % Plot data points
    scatter(X_test(1, idx_neg), X_test(2, idx_neg), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(X_test(1, idx_pos), X_test(2, idx_pos), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
    
    % Plot true boundaries
    theta = linspace(0, 2*pi, 200);
    plot(2*cos(theta), 2*sin(theta), 'k--', 'LineWidth', 2);
    plot(4*cos(theta), 4*sin(theta), 'k--', 'LineWidth', 2);
    
    grid on;
    xlabel('x_1', 'FontSize', 11);
    ylabel('x_2', 'FontSize', 11);
    legend('', '', 'Class -1', 'Class +1', 'True boundary', '', 'Location', 'best');
    axis equal;
    set(gca, 'FontSize', 10);
end

function plotDecisionBoundaryMLP(net, X_test, y_test, model_name)
    % Plot decision boundary for MLP
    idx_neg = (y_test == -1);
    idx_pos = (y_test == 1);
    
    % Create grid
    x1_range = linspace(min(X_test(1,:))-1, max(X_test(1,:))+1, 200);
    x2_range = linspace(min(X_test(2,:))-1, max(X_test(2,:))+1, 200);
    [X1_grid, X2_grid] = meshgrid(x1_range, x2_range);
    X_grid = [X1_grid(:)'; X2_grid(:)'];
    
    % Predict on grid
    y_grid = predictMLP(net, X_grid);
    y_grid = reshape(y_grid, size(X1_grid));
    
    % Plot decision boundary
    contourf(X1_grid, X2_grid, y_grid, 1, 'LineWidth', 1.5);
    colormap([0.7 0.7 1; 1 0.7 0.7]);
    hold on;
    
    % Plot data points
    scatter(X_test(1, idx_neg), X_test(2, idx_neg), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(X_test(1, idx_pos), X_test(2, idx_pos), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
    
    % Plot true boundaries
    theta = linspace(0, 2*pi, 200);
    plot(2*cos(theta), 2*sin(theta), 'k--', 'LineWidth', 2);
    plot(4*cos(theta), 4*sin(theta), 'k--', 'LineWidth', 2);
    
    grid on;
    xlabel('x_1', 'FontSize', 11);
    ylabel('x_2', 'FontSize', 11);
    legend('', '', 'Class -1', 'Class +1', 'True boundary', '', 'Location', 'best');
    axis equal;
    set(gca, 'FontSize', 10);
end