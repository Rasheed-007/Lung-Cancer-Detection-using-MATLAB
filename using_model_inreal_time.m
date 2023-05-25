% Load the trained KNN model
load('lung_cancer_knn_model.mat', 'mdl');

% Prompt the user to select the image file
[filename, pathname] = uigetfile('*.dcm', 'Select the image file');

% Check if a file was selected
if isequal(filename, 0) || isequal(pathname, 0)
    fprintf('No file was selected. Exiting...\n');
    return;
end

% Construct the full image path
imagePath = fullfile(pathname, filename);

% Read the input image
inputImage = dicomread(imagePath);
% Preprocess the image if required
% e.g., resizing, normalization, feature extraction
% Preprocess the input image to match the training data format
inputImage = double(inputImage(:)');

% Predict using the trained KNN model
label = predict(mdl, inputImage);

% Display the prediction
if label == 1
    fprintf('The image is predicted to be cancerous (lung cancer positive).\n');
else
    fprintf('The image is predicted to be non-cancerous (lung cancer negative).\n');
end
