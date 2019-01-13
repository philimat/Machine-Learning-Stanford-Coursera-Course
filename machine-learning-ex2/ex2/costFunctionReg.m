function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Non-vectorized cost function for regularizated logistic regression
% for i = 1:m
%     J = J - y(i)*log(sigmoid(X(i,:)*theta)) - (1 - y(i))*log(1 - sigmoid(X(i,:)*theta));
% end
% J = J/m + lambda*(theta'*t*theta)/(2*m);

% Vectorized regularized logistic regression cost function
J = (-y'*log(sigmoid(X*theta)) - (ones(length(y),1)-y)'*log(1-sigmoid(X*theta)))/m;

% Add regularization to cost function while ignoring theta_0 intercept term
t = eye(length(theta));
t(1,1) = 0;
J = J + lambda*(theta'*t*theta)/(2*m);

% Vectorized regularized logistic regression gradient
grad = (X'*(sigmoid(X*theta)-y)/m) + lambda*t*theta/m ;


% =============================================================

end
