function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% use batch gradient descent
% h(x) = theta' * x for each x in training set
% theta is a (n x 1) col vector
% x is a (1 x n) row vector. X is a (m x n) collection of row vectors
% update all theta simultaneously

% from Mentor recommendations list, computing h with design matrix is X * theta
% X = 97 x 2
% theta = 2 x 1
% h = 97x1
h = X * theta;

errors = (h - y) .^ 2;
sum_sq_errors = sum(errors);

% now calculate cost
J = (1 / (2 * m)) * sum_sq_errors;


% =========================================================================

end
