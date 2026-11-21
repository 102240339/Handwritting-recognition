%% Draw a labeled 3-by-5 chart of MNIST training images
% The 15 source images are included in the mnist_samples folder. The chart
% is saved as mnist_examples_matlab.png in the current folder.

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
