function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the X and Theta matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta

% get all terms starting with Theta and X
predictions = X * Theta'; % nm x nu
movie_rating_error = (predictions - Y) .* R;  % nm x nu
error_rating = movie_rating_error .^ 2;  % nm x nu

% complete the cost function
J = (1 / 2) * sum(sum(error_rating));

% regularize cost function
reg_1 = 0;
for j = 1:num_users
  for k = 1:num_features
    reg_1 = reg_1 + (Theta(j,k) ^ 2);
  end
end
reg_1 = (lambda / 2) * reg_1;

reg_2 = 0;
for i = 1:num_movies
  for k = 1:num_features
    reg_2 = reg_2 + (X(i, k) ^ 2);
  end
end
reg_2 = (lambda / 2) * reg_2;

J = J + reg_1 + reg_2;

% now compute gradients 
% Theta_grad should be nu x nf
Theta_grad = movie_rating_error' * X;  % (nu x nm) * (nm x nf) = (nu x nf)

% want X_grad to be nm x nf
X_grad = movie_rating_error * Theta;  % (nm x nu) * (nu x nf) = (nm x nf)

% regularize the gradients
X_grad = X_grad + (lambda .* X);

Theta_grad = Theta_grad + (lambda .* Theta);


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
