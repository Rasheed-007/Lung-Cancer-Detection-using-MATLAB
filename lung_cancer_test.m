% Load the trained KNN model
load('lung_cancer_knn_model.mat', 'mdl');

% Set the input DICOM folder path
testFolder = "E:\ENG Sem 6\Projects\Lung cancer\dataset\test_dataset_lung_cancer"; % Folder containing test DICOM files

% Read test DICOM files from the folder
testFiles = dir(fullfile(testFolder, '*.dcm'));

% Initialize the predictions
predictions = [];

% Process test DICOM files
for i = 1:numel(testFiles)
    dicomPath = fullfile(testFolder, testFiles(i).name);
    dicomImage = dicomread(dicomPath);
    % Preprocess the image if required
    % e.g., resizing, normalization, feature extraction
    % Preprocess the test image to match the training data format
    testImage = double(dicomImage(:)');
    
    % Predict using the trained KNN model
    label = predict(mdl, testImage);
    predictions = [predictions; label];
end

% Display the predictions
for i = 1:numel(testFiles)
    fprintf('Image: %s, Prediction: %d\n', testFiles(i).name, predictions(i));
end
