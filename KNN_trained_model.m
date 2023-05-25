% Set the input DICOM folder paths
positiveFolder = "E:\ENG Sem 6\Projects\Lung cancer\dataset\lung _cancer_positive_images"; % Folder containing positive DICOM files
negativeFolder = "E:\ENG Sem 6\Projects\Lung cancer\dataset\lung_cancer_negative_images"; % Folder containing negative DICOM files

% Read positive DICOM files from the folder
positiveFiles = dir(fullfile(positiveFolder, '*.dcm'));

% Read negative DICOM files from the folder
negativeFiles = dir(fullfile(negativeFolder, '*.dcm'));

% Initialize the training data and labels
X = [];
Y = [];

% Process positive DICOM files
for i = 1:numel(positiveFiles)
    dicomPath = fullfile(positiveFolder, positiveFiles(i).name);
    dicomImage = dicomread(dicomPath);
    % Preprocess the image if required
    % e.g., resizing, normalization, feature extraction
    % Add the preprocessed image to the training data
    X = [X; double(dicomImage(:)')];
    % Assign the label 1 (positive) to the image
    Y = [Y; 1];
end

% Process negative DICOM files
for i = 1:numel(negativeFiles)
    dicomPath = fullfile(negativeFolder, negativeFiles(i).name);
    dicomImage = dicomread(dicomPath);
    % Preprocess the image if required
    % e.g., resizing, normalization, feature extraction
    % Add the preprocessed image to the training data
    X = [X; double(dicomImage(:)')];
    % Assign the label 0 (negative) to the image
    Y = [Y; 0];
end

% Train the KNN model
k = 5; % Number of neighbors
mdl = fitcknn(X, Y, 'NumNeighbors', k);

% Save the trained model
save('lung_cancer_knn_model.mat', 'mdl');
