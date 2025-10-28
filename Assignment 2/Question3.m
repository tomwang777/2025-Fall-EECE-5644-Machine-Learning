%% EECE5644 - Assignment 2 Question 3
% MAP Estimation
% Objective Function
clear; clc; close all;
rng(1);
%% Set Constants and Grids
sigma_noise = 0.3; % Measurement noise standard deviation
sigma_prior = [0.25; 0.25]; % Prior standard deviations 
K_values = [1, 2, 3, 4]; % K values to iterate over 

% True Location
r_true = rand(1); 
theta_true = 2 * pi * rand(1);
x_true = [r_true * cos(theta_true); r_true * sin(theta_true)]; % [x_T; y_T]

% Grid for Contour Plot (from -2 to 2) 
x_range = linspace(-2, 2, 100); 
y_range = linspace(-2, 2, 100);
[X_grid, Y_grid] = meshgrid(x_range, y_range);

Z_levels = 1:2:30; 

figure;
for k_idx = 1:length(K_values)
    K = K_values(k_idx);
    
%% Generate K landmarks and measurements
    
    % Landmark Location
    angles = linspace(0, 2*pi, K+1);
    angles = angles(1:K); 
    landmark_locs = [cos(angles); sin(angles)]; % 2xK matrix
    
    % Generate measurement value
    r_measurements = zeros(K, 1);
    for i = 1:K
        d_true = norm(x_true - landmark_locs(:, i));
        r_i = -1; % Initialize negative to enter loop
        while r_i < 0
            n_i = sigma_noise * randn(1); % Generate Gaussian noise
            r_i = d_true + n_i;
        end
        r_measurements(i) = r_i;
    end
    
    % Calculate grid search and objective function
    Z_objective = zeros(size(X_grid)); % Store objective function values
    
    for m = 1:size(X_grid, 1)
        for n = 1:size(X_grid, 2)
            x_candidate = [X_grid(m, n); Y_grid(m, n)];
            % Call the objective function
            Z_objective(m, n) = map_objective(x_candidate, r_measurements, ...
                                               landmark_locs, sigma_noise, sigma_prior);
        end
    end
    
%% Plotting and visualization
    
    subplot(2, 2, k_idx); % 2x2 layout
    
    % Plot equilevel contours 
    contour(X_grid, Y_grid, Z_objective, Z_levels, 'LineWidth', 1.5); 
    
    % Superimpose true location 
    hold on;
    plot(x_true(1), x_true(2), 'k+', 'MarkerSize', 10, 'LineWidth', 2); 
    
    % Superimpose landmarks 
    plot(landmark_locs(1, :), landmark_locs(2, :), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'b'); 
    
    % Estimate MAP location
    % x_map_opt = fminsearch(@(x) map_objective(x, r_measurements, landmark_locs, sigma_noise, sigma_prior), [0; 0]);
    % plot(x_map_opt(1), x_map_opt(2), 'rx', 'MarkerSize', 12, 'LineWidth', 2); 
    
    title(sprintf('MAP Objective Contours (K=%d)', K));
    xlabel('x'); ylabel('y');
    axis equal;
    grid on;
    hold off;
end
%% Calculate the objective function value at any point x on the grid
function J = map_objective(x_candidate, r_measurements, landmark_locs, sigma_noise, sigma_prior)
% J = map_objective(x_candidate, r_measurements, landmark_locs, sigma_noise, sigma_prior)
%   x_candidate: [x; y] - 2x1 candidate position vector
%   r_measurements: Kx1 vector of range measurements (r_i)
%   landmark_locs: 2xK matrix of landmark coordinates [x1...xK; y1...yK]
%   sigma_noise: scalar, standard deviation of range noise (sigma_i for all i)
%   sigma_prior: [sigma_x; sigma_y] - 2x1 vector of prior standard deviations

K = length(r_measurements);
sigma_sq = sigma_noise^2; 
J = 0; % Initialize objective function value

% Likelihood term
for i = 1:K
    % True distance
    d_Ti = norm(x_candidate - landmark_locs(:, i)); 
    
    J = J + (r_measurements(i) - d_Ti)^2 / sigma_sq; 
end

% Position Prior
% C_prior_inv = [1/sigma_x^2, 0; 0, 1/sigma_y^2]
sigma_x_sq = sigma_prior(1)^2;
sigma_y_sq = sigma_prior(2)^2;

% x' * C_prior_inv * x = (x^2 / sigma_x^2) + (y^2 / sigma_y^2)
prior_term = (x_candidate(1)^2 / sigma_x_sq) + (x_candidate(2)^2 / sigma_y_sq);

J = J + prior_term;
end