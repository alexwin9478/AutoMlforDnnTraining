% Script to read CSV data and plot Pareto front with scatter plot
% add matlab2tikz to your path for export to latex / tikz
addpath(genpath('C:\Git\matlab2tikz'));
clear; clc;

% preprocessing needs to be done here: make csv into xlsx and convert data
% from text (delimter: comma) to columns and then mark the string fields
% data = readtable('Data\optuna_results_nopBO_Search_GE_05012025_journal.csv'); % ,'Sheet','RunsExtractMainExp');
data = readtable('data\eval_climb_v1\automl_results\optuna_results_val_mse_SOC prediction.csv');

type = "OptunaRun_Climb_v1";
data_nmbr = 07082025;
save_plots_sw = 1;

% preprocessing
% cut out all running or incomplete runs
data = rmmissing(data,1, 'DataVariables','value');

% Separate the columns (replace with your column indices if necessary)
trials = table2array(data(:,1)); % First objective, run number
max_size = size(trials,1);
for i=1:max_size  
    trials(i,1) = i-1; % correct the trials to be consistent, no missing rows)
end
mse = table2array(data(:,2)); % Second objective, objective value
% learnable_params = table2array(data(:,3)); % Third objective, number of learnable parameters
% FLOPS = table2array(data(:,20)); % FLOPS

% Separate the columns (replace with your column indices if necessary)
x = trials; % First objective, run number
y1 = mse; % Second objective, objective value
% y2 = learnable_params; % parameter for analysis
% y3 = FLOPS;

% Plot Pareto front
%paretoPlotSingleObjective(x, y1);
paretoPlotStaircase(x, y1);

if save_plots_sw    
    save_plots(gcf, data_nmbr, type)
end

function paretoPlotSingleObjective(trials, y)
    % PARETOPLOTSINGLEOBJECTIVE Plots the Pareto front for a single optimization variable.
    %
    % Inputs:
    %   trials - Vector of trial numbers.
    %   y      - Vector of optimization values (e.g., error or loss).

    fntsze = 12; 

    % Identify Pareto-efficient solutions
    paretoIdx = isParetoEfficientSingle(y);
    paretoTrials = trials(paretoIdx);
    paretoValues = y(paretoIdx);

    % Plot all solutions
    figure;
    hold on;

    % Plot all solutions (non-Pareto)
    scatter(trials(~paretoIdx), y(~paretoIdx), 70, 'blue', 'filled', 'DisplayName', 'Solutions');
    % Plot Pareto front solutions
    scatter(paretoTrials, paretoValues, 100, 'red', 'x', 'LineWidth', 2, 'DisplayName', 'Pareto Front');
    % Highlight the best Pareto solution
    [~, bestIdx] = min(paretoValues);
    scatter(paretoTrials(bestIdx), paretoValues(bestIdx), 150, 'green', 'o', 'LineWidth', 3, 'DisplayName', 'Best Pareto Solution');

    % Set labels, legend, and title with font size and LaTeX interpreter
    xlabel('Trial Number', 'FontSize', fntsze, 'Interpreter', 'latex');
    ylabel('MSE', 'FontSize', fntsze, 'Interpreter', 'latex');
    legend('Location', 'best', 'FontSize', fntsze, 'Interpreter', 'latex');

    % Adjust axis appearance
    set(gca, 'FontSize', fntsze, 'TickLabelInterpreter', 'latex');
    grid on; box on;
    hold off;
end

% function paretoIdx = isParetoEfficientSingle(y)
%     % IS_PARETOEFFICIENTSINGLE Determines Pareto-efficient solutions for a single objective.
%     %
%     % Inputs:
%     %   y - Vector of objective values.
%     %
%     % Outputs:
%     %   paretoIdx - Logical vector indicating Pareto-efficient solutions.
% 
%     nSolutions = length(y);
%     paretoIdx = true(nSolutions, 1);
% 
%     for i = 1:nSolutions
%         if paretoIdx(i)
%             % Check if any other solution is strictly better (lower y value)
%             paretoIdx(i) = ~any(y < y(i)); % Replace with `>` if maximizing
%         end
%     end
% end
function isEff = isParetoEfficientSingle(Y)
    % ISPAR ETOEFFICIENT  Find the Pareto-efficient rows of Y.
    %
    %   isEff = isParetoEfficient(Y) returns a logical column vector
    %   of length(size(Y,1)), with true for rows that lie on the
    %   Pareto front (assuming minimization of each column).
    %
    %   Example:
    %     Y = [1 4; 2 3; 3 2; 4 1; 2.5 2.5];
    %     idx = isParetoEfficient(Y);
    %     paretoPts = Y(idx,:);  % = [1 4; 2 3; 3 2; 4 1]
    
    [N, M] = size(Y);
    isEff = true(N,1);
    for i = 1:N
        if ~isEff(i)
            continue;
        end
        % For every other point j ≠ i:
        %   check if Y(j,:) <= Y(i,:) elementwise AND any strict <
        dominated = false;
        for j = 1:N
            if j==i, continue; end
            % j weakly better in all objectives?
            if all(Y(j,:) <= Y(i,:)) && any(Y(j,:) < Y(i,:))
                dominated = true;
                break;
            end
        end
        if dominated
            isEff(i) = false;
        end
    end
end


function paretoPlotStaircase(trials, y)
    % PARETOPLOTSTAIRCASE Plots the Pareto front in a staircase format for a single optimization variable.
    %
    % Inputs:
    %   trials - Vector of trial numbers.
    %   y      - Vector of optimization values (e.g., error or loss).

    fntsze = 12; 

    % Identify Pareto-efficient solutions
    paretoIdx = isParetoEfficientSingle(y);
    paretoTrials = trials(paretoIdx);
    paretoValues = y(paretoIdx);

    % Sort Pareto front for staircase plot
    [paretoValues, sortIdx] = sort(paretoValues, 'ascend'); % Replace with 'descend' if maximizing
    paretoTrials = paretoTrials(sortIdx);

    % Add steps to create the staircase effect
    xSteps = [paretoTrials(1:end-1), paretoTrials(2:end)]';
    ySteps = [paretoValues(2:end), paretoValues(2:end)]';

    xStaircase = [paretoTrials(1); xSteps(:)];
    yStaircase = [paretoValues(1); ySteps(:)];

    % Plot all solutions
    figure;
    hold on;

    % Plot all solutions (non-Pareto)
    scatter(trials(~paretoIdx), y(~paretoIdx), 70, 'blue', 'filled', 'DisplayName', 'Solutions');
    % Plot Pareto front solutions
    scatter(paretoTrials, paretoValues, 100, 'red', 'x', 'LineWidth', 2, 'DisplayName', 'Pareto Front');
    % Draw staircase line
    plot(xStaircase, yStaircase, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Pareto Staircase');

    % Highlight the best Pareto solution
    if length(paretoValues) > 0
        [~, bestIdx] = min(paretoValues);
        scatter(paretoTrials(bestIdx), paretoValues(bestIdx), 200, 'green', 'o', 'LineWidth', 3, 'DisplayName', 'Best Pareto Solution');
    end

    % Set labels, legend, and title with font size and LaTeX interpreter
    xlabel('Trial Number', 'FontSize', fntsze, 'Interpreter', 'latex');
    ylabel('Objective Value', 'FontSize', fntsze, 'Interpreter', 'latex');
    legend('Location', 'best', 'FontSize', fntsze, 'Interpreter', 'latex');

    % Adjust axis appearance
    set(gca, 'FontSize', fntsze, 'TickLabelInterpreter', 'latex');
    grid on; box on;
    hold off;
end


function save_plots(gcf, data_nmbr, type, resolution)
    if (~exist('resolution', 'var'))
        resolution = 200;
    end

    figFileName="Plots/"+ sprintf('%04d',data_nmbr) + type;
    
    savefig(figFileName);
    saveas(gcf,figFileName,"jpg");
    if resolution > 0
        cleanfigure('targetResolution', resolution)
    end
    matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false, 'width','\figW','height','\figH');
end
