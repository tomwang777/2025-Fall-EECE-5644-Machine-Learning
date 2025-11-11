%% EECE5644 - Assignment 3 Question 2
% GMM Model Order Selection
% K-Fold Cross-Validation
clear; clc; close all;

%% Define True GMM Parameters
rng(42); % Reproduce

% GMM with 4 components, 2-Dimensional data
n_dim = 2;
n_components_true = 4;

% Component weights
true_weights = [0.25, 0.30, 0.25, 0.20];

% Mean vectors - Components 1 & 2 overlap
true_means = zeros(n_dim, n_components_true);
true_means(:, 1) = [0; 0];    
true_means(:, 2) = [4; 4];      % Component 2 (overlaps with 1)
true_means(:, 3) = [8; 2];     
true_means(:, 4) = [4; 8];    

% Covariance matrices - Components 1 & 2 overlap
true_covs = zeros(n_dim, n_dim, n_components_true);
true_covs(:, :, 1) = [2.0, 0.5; 0.5, 2.0];   
true_covs(:, :, 2) = [2.5, -0.4; -0.4, 2.5]; % Component 2 (overlaps with 1)
true_covs(:, :, 3) = [1.0, 0.2; 0.2, 1.0];   
true_covs(:, :, 4) = [1.5, 0.3; 0.3, 1.5];   

% True GMM parameters
fprintf('Number of components: %d\n', n_components_true);
fprintf('Data dimensionality: %d\n\n', n_dim);

fprintf('Component weights: [%.2f, %.2f, %.2f, %.2f]\n\n', true_weights);

for i = 1:n_components_true
    fprintf('Component %d:\n', i);
    fprintf('  Mean: [%.2f, %.2f]\n', true_means(:, i));
    fprintf('  Covariance:\n');
    fprintf('    [%.2f, %.2f]\n', true_covs(1, :, i));
    fprintf('    [%.2f, %.2f]\n', true_covs(2, :, i));
    
    % Calculate eigenvalues
    eigvals = eig(true_covs(:, :, i));
    fprintf('  Eigenvalues: [%.2f, %.2f], Avg: %.2f\n', eigvals(1), eigvals(2), mean(eigvals));
    
    if i == 2
        % Check overlap between components 1 and 2
        dist = norm(true_means(:, 1) - true_means(:, 2));
        avg_eig_1 = mean(eig(true_covs(:, :, 1)));
        avg_eig_2 = mean(eig(true_covs(:, :, 2)));
        fprintf('  Distance to Component 1: %.2f\n', dist);
        fprintf('  Sum of avg eigenvalues: %.2f (comparable for overlap)\n', avg_eig_1 + avg_eig_2);
    end
    fprintf('\n');
end

%% Experimental Setup
N_samples = [10, 100, 1000];  % Dataset sample sizes
n_experiments = 100;           % Number of repetitions
K_folds = 10;                  % K-fold cross-validation
M_candidates = 1:10;           % Candidate model orders

fprintf('Dataset sizes: [%s]\n', num2str(N_samples));
fprintf('Number of experiments: %d\n', n_experiments);
fprintf('K-fold cross-validation: %d folds\n', K_folds);
fprintf('Candidate model orders: %d to %d components\n\n', M_candidates(1), M_candidates(end));

%% Visualization of True GMM
fprintf('Generating visualization of true GMM...\n');
X_vis = generateGMMData(5000, true_weights, true_means, true_covs);

figure('Position', [100, 100, 900, 700]);
scatter(X_vis(1, :), X_vis(2, :), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;

% Plot component means and covariance ellipses
colors = {'r', 'm', 'g', 'c'};
for i = 1:n_components_true
    plot(true_means(1, i), true_means(2, i), 'x', 'Color', colors{i}, ...
        'MarkerSize', 15, 'LineWidth', 3);
    plotGaussian2D(true_means(:, i), true_covs(:, :, i), colors{i}, 2);
end

grid on;
xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
title('True GMM Distribution (4 Components with Overlap)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Data samples', 'Component means', 'Location', 'best');
axis equal;
set(gca, 'FontSize', 11);

%% Run Experiments
fprintf('\n Experiments Start \n');

% Store results: selected models
selected_models = zeros(length(N_samples), n_experiments);
best_log_likelihoods = zeros(length(N_samples), n_experiments);
all_cv_scores = cell(length(N_samples), n_experiments);

for n_idx = 1:length(N_samples)
    N = N_samples(n_idx);
    fprintf('\n Dataset Size: N = %d \n', N);
    
    for exp = 1:n_experiments
        if mod(exp, 10) == 0
            fprintf('  Experiment %d/%d\n', exp, n_experiments);
        end
        
        % Generate dataset
        X = generateGMMData(N, true_weights, true_means, true_covs);
        
        % Perform K-fold cross-validation for each candidate model order
        cv_log_likelihood = zeros(length(M_candidates), 1);
        
        for m_idx = 1:length(M_candidates)
            M = M_candidates(m_idx);
            
            % K-fold cross-validation
            fold_log_likelihoods = zeros(K_folds, 1);
            
            % Create fold indices
            indices = crossvalind('Kfold', N, K_folds);
            
            for k = 1:K_folds
                % Split data into training and validation
                test_idx = (indices == k);
                train_idx = ~test_idx;
                
                X_train = X(:, train_idx);
                X_val = X(:, test_idx);
                
                % Handle small folds 
                if size(X_train, 2) < M || size(X_val, 2) < 1
                    fold_log_likelihoods(k) = -inf;
                    continue;
                end
                
                % Train GMM using EM algorithm
                try
                    gmm_model = fitgmdist(X_train', M, ...
                        'RegularizationValue', 0.01, ...
                        'Replicates', 3, ...
                        'Options', statset('MaxIter', 500, 'Display', 'off'));
                    
                    % Evaluate log-likelihood on validation set
                    fold_log_likelihoods(k) = sum(log(pdf(gmm_model, X_val') + 1e-10));
                catch
                    % If GMM fitting fails, assign very low likelihood
                    fold_log_likelihoods(k) = -inf;
                end
            end
            
            % Average log-likelihood across folds
            valid_folds = fold_log_likelihoods > -inf;
            if sum(valid_folds) > 0
                cv_log_likelihood(m_idx) = mean(fold_log_likelihoods(valid_folds));
            else
                cv_log_likelihood(m_idx) = -inf;
            end
        end
        
        % Select best model order
        [best_ll, best_idx] = max(cv_log_likelihood);
        selected_models(n_idx, exp) = M_candidates(best_idx);
        best_log_likelihoods(n_idx, exp) = best_ll;
        all_cv_scores{n_idx, exp} = cv_log_likelihood;
    end
    
    fprintf('  Completed %d experiments for N=%d\n', n_experiments, N);
end


%% 5. Analyze and Report Results
fprintf('Results\n\n');

% Selection frequencies for each dataset size
selection_frequencies = zeros(length(N_samples), length(M_candidates));

for n_idx = 1:length(N_samples)
    fprintf('Dataset Size N = %d:\n', N_samples(n_idx));
    fprintf('%-15s %-15s %-15s\n', 'Model Order', 'Count', 'Frequency (%)');
    fprintf('%-15s %-15s %-15s\n', '-----------', '-----', '-------------');
    
    for m_idx = 1:length(M_candidates)
        M = M_candidates(m_idx);
        count = sum(selected_models(n_idx, :) == M);
        freq = count / n_experiments * 100;
        selection_frequencies(n_idx, m_idx) = freq;
        
        if count > 0
            fprintf('%-15d %-15d %-15.2f\n', M, count, freq);
        end
    end
    
    % Statistics
    mean_selected = mean(selected_models(n_idx, :));
    std_selected = std(selected_models(n_idx, :));
    mode_selected = mode(selected_models(n_idx, :));
    
    fprintf('\nStatistics:\n');
    fprintf('  Mean selected order: %.2f\n', mean_selected);
    fprintf('  Std deviation: %.2f\n', std_selected);
    fprintf('  Most frequently selected: %d (%.1f%% of experiments)\n', ...
        mode_selected, max(selection_frequencies(n_idx, :)));
    fprintf('\n');
end

%% Visualizations

% Plot 2: Selection Frequency Heatmap
figure('Position', [150, 150, 1000, 600]);
imagesc(M_candidates, 1:length(N_samples), selection_frequencies);
colorbar;
colormap(hot);
caxis([0 100]);

color_char = 'k'; % Default color (frequency <= 50)
if selection_frequencies(n_idx, m_idx) > 50
    color_char = 'w'; % Frequency > 50，change it to white
end

% Add text annotations
for n_idx = 1:length(N_samples)
    for m_idx = 1:length(M_candidates)
        if selection_frequencies(n_idx, m_idx) > 0
            text(m_idx, n_idx, sprintf('%.1f%%', selection_frequencies(n_idx, m_idx)), ...
     'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
     'Color', color_char);
        end
    end
end

xlabel('Model Order (Number of Components)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Dataset Size', 'FontSize', 12, 'FontWeight', 'bold');
yticks(1:length(N_samples));
yticklabels(arrayfun(@(x) sprintf('N=%d', x), N_samples, 'UniformOutput', false));
title(sprintf('Model Order Selection Frequency (%d Experiments)', n_experiments), ...
    'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

% Plot 3: Bar Chart of Selection Frequencies
figure('Position', [200, 200, 1200, 700]);
for n_idx = 1:length(N_samples)
    subplot(1, 3, n_idx);
    bar(M_candidates, selection_frequencies(n_idx, :), 'FaceColor', [0.2 0.4 0.8]);
    hold on;
    
    % Highlight true model order
    true_order_idx = find(M_candidates == n_components_true);
    if ~isempty(true_order_idx)
        bar(n_components_true, selection_frequencies(n_idx, true_order_idx), ...
            'FaceColor', [0.8 0.2 0.2]);
    end
    
    grid on;
    xlabel('Model Order', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Selection Frequency (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('N = %d', N_samples(n_idx)), 'FontSize', 12, 'FontWeight', 'bold');
    xlim([0.5, length(M_candidates) + 0.5]);
    ylim([0, max(selection_frequencies(n_idx, :)) * 1.1 + 5]);
    
    if n_idx == 1
        legend('Selected Models', 'True Order (M=4)', 'Location', 'best');
    end
end
sgtitle(sprintf('GMM Model Order Selection Results (%d Experiments)', n_experiments), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Plot 4: Box plots of selected model orders
figure('Position', [250, 250, 900, 600]);
boxplot(selected_models', 'Labels', arrayfun(@(x) sprintf('N=%d', x), N_samples, 'UniformOutput', false));
hold on;
plot([0.5, length(N_samples) + 0.5], [n_components_true, n_components_true], ...
    'r--', 'LineWidth', 2);
grid on;
ylabel('Selected Model Order', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Dataset Size', 'FontSize', 12, 'FontWeight', 'bold');
title('Distribution of Selected Model Orders Across Experiments', 'FontSize', 14, 'FontWeight', 'bold');
legend('True Order (M=4)', 'Location', 'best');
set(gca, 'FontSize', 11);

% Plot 5: Average CV Log-Likelihood Curves
figure('Position', [300, 300, 1200, 700]);
for n_idx = 1:length(N_samples)
    subplot(1, 3, n_idx);
    
    % Compute average CV scores across experiments
    avg_cv_scores = zeros(length(M_candidates), 1);
    std_cv_scores = zeros(length(M_candidates), 1);
    
    for m_idx = 1:length(M_candidates)
        scores = zeros(n_experiments, 1);
        for exp = 1:n_experiments
            scores(exp) = all_cv_scores{n_idx, exp}(m_idx);
        end
        valid_scores = scores(scores > -inf);
        if ~isempty(valid_scores)
            avg_cv_scores(m_idx) = mean(valid_scores);
            std_cv_scores(m_idx) = std(valid_scores);
        else
            avg_cv_scores(m_idx) = nan;
            std_cv_scores(m_idx) = nan;
        end
    end
    
    % Plot with error bars
    errorbar(M_candidates, avg_cv_scores, std_cv_scores, 'b-o', ...
        'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    hold on;
    
    % Mark the most frequently selected order
    mode_order = mode(selected_models(n_idx, :));
    mode_idx = find(M_candidates == mode_order);
    if ~isempty(mode_idx) && ~isnan(avg_cv_scores(mode_idx))
        plot(mode_order, avg_cv_scores(mode_idx), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
    end
    
    grid on;
    xlabel('Model Order', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Average CV Log-Likelihood', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('N = %d', N_samples(n_idx)), 'FontSize', 12, 'FontWeight', 'bold');
    
    if n_idx == 1
        legend('Mean ± Std', 'Most Selected', 'Location', 'best');
    end
end
sgtitle('Cross-Validation Performance vs Model Order', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\nAll visualizations generated successfully!\n');

%% Functions

function X = generateGMMData(N, weights, means, covs)
    % Generate data from GMM
    n_components = length(weights);
    n_dim = size(means, 1);
    
    % Sample component assignments
    component_samples = randsample(n_components, N, true, weights);
    
    % Generate samples
    X = zeros(n_dim, N);
    for i = 1:N
        c = component_samples(i);
        X(:, i) = mvnrnd(means(:, c), covs(:, :, c))';
    end
end

function plotGaussian2D(mu, Sigma, color, nstd)
    % Plot 2D Gaussian confidence ellipse
    [V, D] = eig(Sigma);
    
    % Generate ellipse points
    theta = linspace(0, 2*pi, 100);
    circle = nstd * [cos(theta); sin(theta)];
    
    % Transform circle to ellipse
    ellipse = V * sqrt(D) * circle + mu;
    
    plot(ellipse(1, :), ellipse(2, :), 'Color', color, 'LineWidth', 2);
end