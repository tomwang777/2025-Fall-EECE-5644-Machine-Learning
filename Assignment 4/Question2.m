%% EECE5644 - Assignment 4 Question 2
% GMM-Based Color Image Segmentation with Cross-Validation
clear; close all; clc;

%% Load and Preprocess Image
img = imread('69015.jpg');

fprintf('Image loaded successfully.\n');
fprintf('Image size: %d x %d x %d\n', size(img, 1), size(img, 2), size(img, 3));

% Downsample if image is too large
max_dim = 200; % Maximum dimension
[h, w, ~] = size(img);
if max(h, w) > max_dim
    scale_factor = max_dim / max(h, w);
    img = imresize(img, scale_factor);
    fprintf('Image downsampled to: %d x %d (scale factor: %.2f)\n', ...
        size(img, 1), size(img, 2), scale_factor);
end

% Display original image
figure('Position', [100, 100, 1200, 400]);
subplot(1, 3, 1);
imshow(img);
title('Original Image', 'FontSize', 12, 'FontWeight', 'bold');

%% Create 5-Dimensional Feature Vectors
fprintf('\nCreating 5-dimensional feature vectors\n');

[h, w, c] = size(img);
n_pixels = h * w;

% Initialize feature matrix: [row, col, R, G, B]
features = zeros(n_pixels, 5);

% Create pixel coordinates and color values
[col_idx, row_idx] = meshgrid(1:w, 1:h);
features(:, 1) = row_idx(:);        % Row index
features(:, 2) = col_idx(:);        % Column index
temp_red = double(img(:, :, 1));
features(:, 3) = temp_red(:); % Red channel
features(:, 3) = features(:, 3);
temp_green = double(img(:, :, 2));
features(:, 4) = temp_green(:); % Green channel
features(:, 4) = features(:, 4);
temp_blue = double(img(:, :, 3));
features(:, 5) = temp_blue(:); % Blue channel
features(:, 5) = features(:, 5);

fprintf('Feature matrix created: %d pixels x 5 features\n', n_pixels);

% Normalize each feature to [0, 1]
fprintf('Normalizing features to [0, 1]...\n');
for i = 1:5
    min_val = min(features(:, i));
    max_val = max(features(:, i));
    if max_val > min_val
        features(:, i) = (features(:, i) - min_val) / (max_val - min_val);
    end
end

fprintf('Features normalized. All feature vectors fit in 5D unit hypercube.\n');

%% Model Order Selection using K-Fold Cross-Validation

K_folds = 5; % Use 5-fold CV
M_candidates = 2:8; % Test 2 to 8 components
n_replicates = 3; % Number of random initializations for GMM fitting

fprintf('K-fold cross-validation: %d folds\n', K_folds);
fprintf('Candidate model orders: %d to %d components\n', M_candidates(1), M_candidates(end));
fprintf('GMM fitting replicates: %d\n\n', n_replicates);

% Perform cross-validation
cv_log_likelihoods = zeros(length(M_candidates), 1);
cv_indices = crossvalind('Kfold', n_pixels, K_folds);

fprintf('Running cross-validation...\n');
for m_idx = 1:length(M_candidates)
    M = M_candidates(m_idx);
    fprintf('  Testing M = %d components...', M);
    
    fold_log_likelihoods = zeros(K_folds, 1);
    
    for k = 1:K_folds
        % Split data
        val_idx = (cv_indices == k);
        train_idx = ~val_idx;
        
        X_train = features(train_idx, :);
        X_val = features(val_idx, :);
        
        % Train GMM on training fold
        try
            gmm_model = fitgmdist(X_train, M, ...
                'RegularizationValue', 0.01, ...
                'Replicates', n_replicates, ...
                'Options', statset('MaxIter', 200, 'Display', 'off'));
            
            % Evaluate log-likelihood on validation fold
            fold_log_likelihoods(k) = sum(log(pdf(gmm_model, X_val) + 1e-10));
        catch
            fold_log_likelihoods(k) = -inf;
        end
    end
    
    % Average log-likelihood across folds
    valid_folds = fold_log_likelihoods > -inf;
    if sum(valid_folds) > 0
        cv_log_likelihoods(m_idx) = mean(fold_log_likelihoods(valid_folds));
        fprintf(' CV log-likelihood: %.2f\n', cv_log_likelihoods(m_idx));
    else
        cv_log_likelihoods(m_idx) = -inf;
        fprintf(' Failed\n');
    end
end

% Select best model order
[best_ll, best_idx] = max(cv_log_likelihoods);
best_M = M_candidates(best_idx);

fprintf('\nBest model order selected: M = %d components\n', best_M);
fprintf('Best CV log-likelihood: %.2f\n\n', best_ll);

% Plot CV results
figure('Position', [150, 150, 800, 500]);
plot(M_candidates, cv_log_likelihoods, 'bo-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
hold on;
plot(best_M, best_ll, 'r*', 'MarkerSize', 20, 'LineWidth', 3);
grid on;
xlabel('Number of Components (M)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Average CV Log-Likelihood', 'FontSize', 12, 'FontWeight', 'bold');
title('Model Order Selection via Cross-Validation', 'FontSize', 14, 'FontWeight', 'bold');
legend('CV Log-Likelihood', sprintf('Best M=%d', best_M), 'Location', 'best', 'FontSize', 11);
set(gca, 'FontSize', 11);

%% Fit Final GMM with Best Model Order
fprintf('Training GMM with M = %d components on full dataset\n', best_M);

% Train final GMM with more replicates for best results
gmm_final = fitgmdist(features, best_M, ...
    'RegularizationValue', 0.01, ...
    'Replicates', 5, ...
    'Options', statset('MaxIter', 300, 'Display', 'off'));

fprintf('Final GMM training complete.\n\n');

% Display GMM parameters
fprintf('GMM Component Weights:\n');
for i = 1:best_M
    fprintf('  Component %d: %.4f\n', i, gmm_final.ComponentProportion(i));
end
fprintf('\n');

%% Segment Image using GMM
fprintf('Image Segmentation\n');

% Compute posterior probabilities for each pixel
posteriors = posterior(gmm_final, features);

% Assign each pixel to component with highest posterior
[~, pixel_labels] = max(posteriors, [], 2);

% Reshape labels to image dimensions
label_image = reshape(pixel_labels, h, w);

fprintf('Segmentation complete.\n');
fprintf('Segments identified: %d\n\n', length(unique(pixel_labels)));

%% Visualize Segmentation Results
fprintf('Visualization\n');

% Create grayscale label image with contrast
% Distribute grayscale values uniformly between min and max
label_image_gray = label_image;
unique_labels = unique(label_image);
n_labels = length(unique_labels);
gray_values = linspace(0, 255, n_labels);

label_image_display = zeros(h, w, 'uint8');
for i = 1:n_labels
    label_image_display(label_image == unique_labels(i)) = gray_values(i);
end

% Display segmentation results
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
imshow(img);
title('Original Image', 'FontSize', 12, 'FontWeight', 'bold');

subplot(1, 3, 2);
imshow(label_image_display);
title(sprintf('GMM Segmentation (M=%d)', best_M), 'FontSize', 12, 'FontWeight', 'bold');

% Create colored segmentation for better visualization
subplot(1, 3, 3);
imagesc(label_image);
colormap(jet(best_M));
colorbar;
title(sprintf('Segmentation Labels (M=%d)', best_M), 'FontSize', 12, 'FontWeight', 'bold');
axis image;
axis off;

% Create a figure showing each segment separately
figure('Position', [200, 200, 1200, 800]);
n_cols = ceil(sqrt(best_M));
n_rows = ceil(best_M / n_cols);

for i = 1:best_M
    subplot(n_rows, n_cols, i);
    
    % Create binary mask for this component
    mask = (label_image == i);
    
    % Create RGB image showing only this segment
    segment_img = img;
    for c = 1:3
        channel = segment_img(:, :, c);
        channel(~mask) = 0;
        segment_img(:, :, c) = channel;
    end
    
    imshow(segment_img);
    title(sprintf('Segment %d (%.1f%% pixels)', i, sum(mask(:))/n_pixels*100), ...
        'FontSize', 10, 'FontWeight', 'bold');
end
sgtitle('Individual Segments', 'FontSize', 14, 'FontWeight', 'bold');

% Overlay segmentation boundaries on original image
figure('Position', [250, 250, 800, 600]);
imshow(img);
hold on;

% Find boundaries between segments
boundaries = zeros(h, w);
for i = 1:h-1
    for j = 1:w-1
        if label_image(i, j) ~= label_image(i+1, j) || ...
           label_image(i, j) ~= label_image(i, j+1)
            boundaries(i, j) = 1;
        end
    end
end

% Overlay boundaries in yellow
[by, bx] = find(boundaries);
plot(bx, by, 'y.', 'MarkerSize', 3);
title(sprintf('Segmentation Boundaries (M=%d)', best_M), 'FontSize', 14, 'FontWeight', 'bold');
hold off;
%% Summary Statistics
fprintf('Results\n');
fprintf('Image size: %d x %d\n', h, w);
fprintf('Total pixels: %d\n', n_pixels);
fprintf('Number of segments: %d\n', best_M);
fprintf('\nSegment sizes:\n');

for i = 1:best_M
    n_pixels_segment = sum(pixel_labels == i);
    percentage = n_pixels_segment / n_pixels * 100;
    fprintf('  Segment %d: %d pixels (%.2f%%)\n', i, n_pixels_segment, percentage);
end

fprintf('\nGMM Component Characteristics:\n');
for i = 1:best_M
    fprintf('  Component %d:\n', i);
    fprintf('    Weight: %.4f\n', gmm_final.ComponentProportion(i));
    fprintf('    Mean RGB (normalized): [%.3f, %.3f, %.3f]\n', ...
        gmm_final.mu(i, 3), gmm_final.mu(i, 4), gmm_final.mu(i, 5));
end

fprintf('\nAll visualizations generated successfully!\n');
fprintf('\n=== SEGMENTATION COMPLETE ===\n');