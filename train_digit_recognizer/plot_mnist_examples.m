%% Draw a labeled 3-by-5 chart of MNIST training images
% The 15 source images are included in the mnist_samples folder.

clear; clc;

sampleFolder = fullfile(fileparts(mfilename('fullpath')), 'mnist_samples');
outputFile = fullfile(fileparts(mfilename('fullpath')), 'mnist_examples_matlab.png');
files = dir(fullfile(sampleFolder, 'mnist_*.png'));
assert(numel(files) == 15, ...
    'Expected 15 MNIST sample PNGs in the mnist_samples folder.');

% Sort names so the chart order stays consistent across runs.
[~, order] = sort({files.name});
files = files(order);

fig = figure('Color', 'white', 'Name', 'MNIST training examples');
tiledlayout(fig, 3, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:numel(files)
    imagePath = fullfile(files(k).folder, files(k).name);
    digitImage = imread(imagePath);
    token = regexp(files(k).name, 'digit_(\d)\.png$', 'tokens', 'once');
    assert(~isempty(token), 'Could not read digit label from %s.', files(k).name);
    digitLabel = token{1};

    nexttile;
    imshow(digitImage, []);
    title(['Label: ' digitLabel], 'FontSize', 10, 'FontWeight', 'normal');
end
sgtitle('Examples from the MNIST training set', 'FontWeight', 'bold');
exportgraphics(fig, outputFile, 'Resolution', 200);
fprintf('Saved chart to %s\n', outputFile);

%% Plot training and validation accuracy by training-set size
% Use the measured values saved by run_mnist_experiments.py.
projectFolder = fileparts(mfilename('fullpath'));
resultsFile = fullfile(projectFolder, 'mnist_experiment_results.json');
results = jsondecode(fileread(resultsFile));
sizeResults = results.training_size_experiment.rows;
trainingImages = [sizeResults.training_examples];
trainingAccuracy = 100 * [sizeResults.train_accuracy];
validationAccuracy = 100 * [sizeResults.validation_accuracy];

fig = figure('Color', 'white', 'Name', 'Accuracy by training-set size', ...
    'Position', [100 100 1378 700]);
plot(1:numel(trainingImages), trainingAccuracy, '-o', ...
    'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', [0.28 0.47 0.38], ...
    'Color', [0.28 0.47 0.38], ...
    'DisplayName', 'Training accuracy');
hold on;
plot(1:numel(trainingImages), validationAccuracy, '-s', ...
    'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', [0.90 0.48 0.35], ...
    'Color', [0.90 0.48 0.35], ...
    'DisplayName', 'Validation accuracy');
hold off;
ylim([75 100]);
yticks(75:5:100);
xticks(1:numel(trainingImages));
xticklabels(compose('%dk', round(trainingImages / 1000)));
ylabel('Accuracy (%)');
xlabel('Training images');
lgd = legend('Location', 'southeast', 'Orientation', 'horizontal');
lgd.NumColumns = 2;
lgd.Box = 'off';
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.GridColor = [0.85 0.85 0.85];
ax.GridAlpha = 0.7;
ax.LineWidth = 1.2;
ax.FontSize = 15;
ax.Position = [0.095 0.15 0.89 0.80];
ax.Layer = 'top';
exportgraphics(fig, fullfile(projectFolder, 'training_size_experiment_matlab.png'), ...
    'Resolution', 200);

%% Draw the saved CNN confusion matrix

confusionMatrix = [ ...
    972 0 0 0 1 2 1 2 2 0; ...
    0 1128 2 0 0 1 2 2 0 0; ...
    1 0 1017 4 0 0 0 5 5 0; ...
    1 0 2 1001 0 3 0 2 1 0; ...
    1 0 2 0 969 0 1 3 0 6; ...
    2 0 0 4 0 882 2 1 1 0; ...
    5 3 0 1 1 7 941 0 0 0; ...
    0 1 6 1 0 0 0 1017 1 2; ...
    1 0 6 9 0 7 0 2 946 3; ...
    0 3 1 6 6 6 0 9 0 978];

fig = figure('Color', 'white', 'Name', 'Saved CNN confusion matrix', ...
    'Position', [100 100 1140 980]);
imagesc(confusionMatrix);
axis square;
caxis([0 max(confusionMatrix(:))]);
blueStops = [247 251 255; 222 235 247; 198 219 239; 158 202 225; ...
    107 174 214; 66 146 198; 33 113 181; 8 81 156; 8 48 107] / 255;
blueMap = interp1(linspace(0, 1, size(blueStops, 1)), blueStops, ...
    linspace(0, 1, 256));
colormap(gca, blueMap);
cb = colorbar;
cb.Label.String = 'Number of images';
cb.FontSize = 14;
cb.Ticks = 0:200:1000;
digitLabels = 0:9;
xticks(1:10);
yticks(1:10);
xticklabels(digitLabels);
yticklabels(digitLabels);
xlabel('Predicted digit');
ylabel('True digit');
title('Saved CNN confusion matrix on MNIST test set');
ax = gca;
ax.FontSize = 14;
ax.LineWidth = 1.2;
ax.TickLength = [0.01 0.01];
ax.Position = [0.105 0.12 0.72 0.79];
cb.Position = [0.85 0.12 0.035 0.79];
for row = 1:10
    for column = 1:10
        if confusionMatrix(row, column) > max(confusionMatrix(:)) / 2
            labelColor = 'white';
        else
            labelColor = [0.13 0.19 0.25];
        end
        text(column, row, num2str(confusionMatrix(row, column)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', labelColor, 'FontSize', 10, 'FontName', 'Arial');
    end
end
exportgraphics(fig, fullfile(projectFolder, 'cnn_confusion_matrix_matlab.png'), ...
    'Resolution', 200);
fprintf('Saved training-size and confusion-matrix plots in %s\n', projectFolder);
