function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;

errors = (h - y) .^ 2;
sum_sq_errors = sum(errors);

% now calculate cost
J = (1 / (2 * m)) * sum_sq_errors;

% now regularize
sum_theta = sum(theta(2:end) .^ 2);
J = J + ((lambda / (2 * m)) * sum_theta);

% calculate gradient
grad = (1 / m) * (X' * (h - y));
regularization = (lambda / m) .* theta(2:end);
grad_tmp = grad(2:end) + regularization;
  
grad = [grad(1);grad_tmp];

% =========================================================================

grad = grad(:);

end
