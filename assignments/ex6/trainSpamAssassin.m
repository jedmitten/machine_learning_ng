ham_dir = 'e:\\Users\\Lughead\\Downloads\\spamassassin\\ham';
spam_dir = 'e:\\Users\\Lughead\\Downloads\\spamassassin\\spam';
dataset_filename = 'spamassassin_dataset.mat';

% load dataset from file if it exists
if exist(dataset_filename) == 2
    fprintf('Loading data from %s\n', dataset_filename);
    dataset = load(dataset_filename);
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

% create X and y
X = dataset(:, 1:end-1);
y = dataset(:, end);
fprintf('Size X: %s\n', size(X));
fprint('Size y: %s\n', size(y));
C = 0.1;
sigma = 0.3;
model = svm