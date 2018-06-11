ham_dir = 'e:\\Users\\Lughead\\Downloads\\spamassassin\\ham';
spam_dir = 'e:\\Users\\Lughead\\Downloads\\spamassassin\\spam';
dataset_filename = 'spamassassin_dataset.mat';

% load dataset from file if it exists
if exist(dataset_filename) == 2
    fprintf('Loading data from %s\n', dataset_filename);
    load(dataset_filename);  % loads variable called 'dataset'
elseif exist(dataset_filename) == 0
    % create and save dataset
    ham_list = readdir(ham_dir);
    ham_list = ham_list(3:2003);
    spam_list = readdir(spam_dir);
    spam_list = spam_list(3:2003);
    % dataset = zeros((length(ham_list) + length(spam_list) - 4), 1900);
    dataset = [];
    for idx = 1:numel(ham_list)
      if (strcmp(ham_list{idx}, '..') == 1 || strcmp(ham_list{idx}, '.') == 1)
        continue;
      end
      % read file
      fprintf('Reading ham file # %d\n', idx);
      fflush(stdout);
      file_name = strcat(ham_dir, "\\", ham_list{idx});
      file_contents = readFile(file_name);
      word_indices =  processEmail(file_contents);
      features = emailFeatures(word_indices);
      dataset = [dataset; features' 0];
    end

    for idx = 1:numel(spam_list)
      if (strcmp(spam_list{idx}, '..') == 1 || strcmp(spam_list{idx}, '.') == 1)
        continue;
      end
      % read file
      fprintf('Reading spam file # %d\n', idx);
      fflush(stdout);
      file_name = strcat(spam_dir, "\\", spam_list{idx});
      file_contents = readFile(file_name);
      word_indices =  processEmail(file_contents);
      features = emailFeatures(word_indices);
      dataset = [dataset; features' 1];
    end

    % save dataset
    fprintf('Saving dataset to %s\n', dataset_filename);
    save dataset dataset_filename;
end

% create training, validation, and test sets
% randomize rows
shuffled_dataset = dataset(randperm(size(dataset)), :);
% Create X, Xval, y, yval, Xtest, ytest
%train_max = int32(length(dataset) * .6);
%validation_max = train_max + int32(length(dataset) * .2);
train_max = int32(length(shuffled_dataset) / 2);
X = shuffled_dataset(1:train_max, 1:end-1);
y = shuffled_dataset(1:train_max, end);
%Xval = dataset((train_max)+1:validation_max, 1:end-1);
%yval = dataset((train_max)+1:validation_max, end);
Xtest = shuffled_dataset((train_max+1):end, 1:end-1);
ytest = shuffled_dataset((train_max+1):end, 1:end-1);
fprintf('Size X: \n');
size(X)
fprintf('Size y: \n');
size(y);
%fprintf('Size of Xval\n');
%size(Xval);
%fprintf('Size of yval\n');
%size(yval);
fprintf('Size of Xtest\n');
size(Xtest);
fprintf('Size of ytest\n');
size(ytest);
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);
p = svmPredict(model, X);
fprintf('Training accuracy: %d\n', mean(double(p == y)) * 100);
p = svmPredict(model, Xtest);
fprintf('Test accuracy: %d\n', mean(double(p == y)) * 100);
