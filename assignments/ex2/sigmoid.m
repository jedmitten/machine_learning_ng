function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
[m, n] = size(z);
for i = 1:m
  for j = 1:n
    e_term = e .^ (z(i,j) * -1);
    g(i,j) = 1 / (1 + e_term);
endfor;

% =============================================================

end
