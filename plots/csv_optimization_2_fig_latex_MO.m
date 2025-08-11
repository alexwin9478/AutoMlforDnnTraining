% Script to read CSV data and plot Pareto front with scatter plot
% add matlab2tikz to your path for export to latex / tikz
addpath(genpath('C:\Git\matlab2tikz'));
clear; clc;

% preprocessing needs to be done here: make csv into xlsx and convert data
% from text (delimter: comma) to columns and then mark the string fields
data = readtable('Data\optuna_results_nopBO_Search_Learn_03012025_journal.csv'); % ,'Sheet','RunsExtractMainExp');

type = "OptunaRun_MO_1";
data_nmbr = 02012024;
save_plots_sw = 0;

% preprocessing
% cut out all running or incomplete runs
data = rmmissing(data,1, 'DataVariables','values_0');

% Separate the columns (replace with your column indices if necessary)
trials = table2array(data(:,1)); % First objective, run number
max_size = size(trials,1);
for i=1:max_size  
    trials(i,1) = i-1; % correct the trials to be consistent, no missing rows)
end
GE = table2array(data(:,2)); % Second objective, objective value
learnable_params = table2array(data(:,3)); % Third objective, number of learnable parameters
FLOPS = table2array(data(:,20)); % FLOPS

% Separate the columns (replace with your column indices if necessary)
x = trials; % First objective, run number
y1 = GE; % Second objective, objective value
y2 = learnable_params; % parameter for analysis
y3 = FLOPS;

objectiveValues = [y1, y3];
variableNames = {'GE', 'FLOPS'};
% Plot Pareto front
paretoPlotMultiObjectiveWithBestNormalized(objectiveValues, variableNames);
% paretoPlotMultiObjectiveWithBestColored(objectiveValues, variableNames);

if save_plots_sw    
    save_plots(gcf, data_nmbr, type)
end

paretoPlotStaircase(x, y1); % for GE
paretoPlotStaircase(x, y3); % for FLOPS

function paretoPlotMultiObjectiveWithBestNormalized(objectiveValues, variableNames)
    % PARETOPLOTMULTIOBJECTIVEWITHBESTNORMALIZED Plots the Pareto front with all solutions and highlights the best solutions.
    %
    % Inputs:
    %   objectiveValues - A matrix of size [nSolutions, 2] with objective values.
    %   variableNames   - Cell array of names for the two objectives.

    fntsze = 12; 

    if nargin < 2
        variableNames = {'Objective 1', 'Objective 2'};
    end

    % Identify Pareto-efficient solutions
    paretoIdx = isParetoEfficient(objectiveValues);
    paretoSet = objectiveValues(paretoIdx, :);
    dominatedSet = objectiveValues(~paretoIdx, :);

    % Find the best solution on the Pareto front with normalization
    bestSolution = findBestParetoSolutionNormalized(paretoSet);

    % Find the best solution for each individual objective
    [~, bestObj1Idx] = min(paretoSet(:, 1)); % Best for Objective 1
    bestObj1Solution = paretoSet(bestObj1Idx, :);

    [~, bestObj2Idx] = min(paretoSet(:, 2)); % Best for Objective 2
    bestObj2Solution = paretoSet(bestObj2Idx, :);

    % Plot all solutions
    figure;
    hold on;

    % Plot all solutions (dominated solutions)
    scatter(dominatedSet(:, 1), dominatedSet(:, 2), 70, 'blue', 'filled', 'DisplayName', 'Solutions');
    % Plot Pareto front solutions
    scatter(paretoSet(:, 1), paretoSet(:, 2), 100, 'red', 'x', 'LineWidth', 2, 'DisplayName', 'Pareto Front');
    % Highlight the best Pareto solution
    scatter(bestSolution(1), bestSolution(2), 150, 'green', 'x', 'LineWidth', 3, 'DisplayName', 'Best Pareto Solution - Shortest Euclidian Distance to Ideal Points');
    % Highlight the best solutions for each individual objective
    scatter(bestObj1Solution(1), bestObj1Solution(2), 120, 'magenta', 'o', 'LineWidth', 2, 'DisplayName',  ['Ideal Point for ', variableNames{1}]);
    scatter(bestObj2Solution(1), bestObj2Solution(2), 120, 'cyan', 'o', 'LineWidth', 2, 'DisplayName',  ['Ideal Point for ', variableNames{2}]);

    % Set labels, legend, and title with font size and LaTeX interpreter
    xlabel(variableNames{1}, 'FontSize', fntsze, 'Interpreter', 'latex');
    % xlabel('\cNox', 'FontSize', fntsze, 'Interpreter', 'latex');
    ylabel(variableNames{2}, 'FontSize', fntsze, 'Interpreter', 'latex');
    legend('Location', 'best', 'FontSize', fntsze, 'Interpreter', 'latex');

    % Adjust axis appearance
    set(gca, 'FontSize', fntsze, 'TickLabelInterpreter', 'latex');
    grid on; box on;
    hold off;
end

function bestSolution = findBestParetoSolutionNormalized(objectiveValues)
    % FINDBESTPARETOSOLUTIONNORMALIZED Finds the best solution on the Pareto front using normalized values.
    %
    % Inputs:
    %   objectiveValues - A matrix of size [nSolutions, 2] containing the objective values.
    %
    % Outputs:
    %   bestSolution    - The solution on the Pareto front closest to the normalized ideal point.

    % Identify Pareto-efficient solutions
    paretoIdx = isParetoEfficient(objectiveValues);
    paretoSet = objectiveValues(paretoIdx, :);

    % Compute the range of each objective
    ranges = max(objectiveValues, [], 1) - min(objectiveValues, [], 1);

    % Normalize the Pareto solutions
    normalizedParetoSet = (paretoSet - min(objectiveValues, [], 1)) ./ ranges;

    % Ideal point in normalized space
    idealPoint = zeros(1, size(objectiveValues, 2));

    % Calculate the Euclidean distance to the ideal point for each normalized Pareto solution
    distances = sqrt(sum((normalizedParetoSet - idealPoint).^2, 2));

    % Find the index of the Pareto solution with the minimum distance
    [~, bestIdx] = min(distances);

    % Extract the best solution
    bestSolution = paretoSet(bestIdx, :);
end

function paretoIdx = isParetoEfficient(objectiveValues)
    % IS_PARETOEFFICIENT Determines Pareto-efficient solutions.
    %
    % Inputs:
    %   objectiveValues - Matrix of size [nSolutions, 2].
    %
    % Outputs:
    %   paretoIdx - Logical vector indicating Pareto-efficient solutions.

    nSolutions = size(objectiveValues, 1);
    paretoIdx = true(nSolutions, 1);

    for i = 1:nSolutions
        if paretoIdx(i)
            % Check if current solution is dominated by any other solution
            dominates = all(objectiveValues <= objectiveValues(i, :), 2) & any(objectiveValues < objectiveValues(i, :), 2);
            paretoIdx(i) = ~any(dominates); % Only keep if no one dominates this solution
        end
    end
end

function save_plots(gcf, data_nmbr, type, resolution)
    if (~exist('resolution', 'var'))
        resolution = 200;
    end

    figFileName="Plots/"+ sprintf('%04d',data_nmbr) + type;
    
    savefig(figFileName);
    saveas(gcf,figFileName,"jpg");
    % saveas(gcf,figFileName,"epsc");
    % saveas(gcf,figFileName,"pdf");
    if resolution > 0
        cleanfigure('targetResolution', resolution)
    end
    matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false, 'width','\figW','height','\figH');
    % axis_code = 'yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}, scaled ticks=false,';
    % matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false, 'width','\figW','height','\figH','extraAxisOptions',axis_code); % 'externalData',true
    %export_fig(figFileName,'-eps');

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

function paretoIdx = isParetoEfficientSingle(y)
    % IS_PARETOEFFICIENTSINGLE Determines Pareto-efficient solutions for a single objective.
    %
    % Inputs:
    %   y - Vector of objective values.
    %
    % Outputs:
    %   paretoIdx - Logical vector indicating Pareto-efficient solutions.

    nSolutions = length(y);
    paretoIdx = true(nSolutions, 1);

    for i = 1:nSolutions
        if paretoIdx(i)
            % Check if any other solution is strictly better (lower y value)
            paretoIdx(i) = ~any(y < y(i)); % Replace with `>` if maximizing
        end
    end
end

