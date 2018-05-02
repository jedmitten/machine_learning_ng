function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta);
[J, grad] = costFuntion(theta, X, y);
theta_sq = theta(2:n) .^ 2;
% update basic cost with regularization term
J = J + (lambda / (2 * m)) .* sum(theta_sq);

grad_0 = grad(1);
% update basic gradient with regularization term
grad_rest = grad(2:n) + ((lambda / m) .* grad(2:n));
grad = [grad_0; grad_rest];

% =============================================================

end
