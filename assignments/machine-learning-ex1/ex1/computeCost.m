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
temp_theta = theta' 
prediction = zeros(m, 1);
for j = 1:m
  x = X(j, :)';
  prediction(j) = prediction(j) + theta' * x;
end;

J = 1 / 2*m * ((prediction .- y) .^ 2);


% =========================================================================

end
