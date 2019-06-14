function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Computes the cost and gradient for logistic regression
%   including regularization terms

%
% Initializations 
%
m = length(y); % calculate the number of training examples m
J = 0; % Rerturn values
grad = zeros(size(theta));

% calculate h of X
% Vectorized
z = X * theta; 
hypothesis = sigmoid (z);

% ********************************************
% Regularization term excluding theta (0)
% Note both L1 and L2 regularization routines
% are provided, comment/uncomment as appropriate
% ********************************************
reg_term = (lambda/(2*m))* sum (theta(2:end).^2); %% L1 regularizaiton
fprintf('L1 Regularization\n');
% reg_term = (lambda/(2*m))* sum (abs(theta(2:end))); %% L2 regularization
% fprintf('L2 Regularization\n');

% ********************************************
% Calculate and return cost in J
% ********************************************

J = (1/m) * sum (( - y.* log(hypothesis)) - (( 1 -y).* log(1-hypothesis))); 


% *********************************************
% Vecotrized gradient descent calculations
% *********************************************

grad(1) = (1/m) * (X(:,1)'*(hypothesis-y));
grad(2:end) = (1/m) * (X(:,2:end)'*(hypothesis-y)) + (lambda/m) * (theta(2:end));

end
