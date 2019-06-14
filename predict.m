function p = predict(theta, X)
%  *************************************************************
%   This routine has not been modified yet to work with Resumes
%  *************************************************************

% Predicts label (1 | 0) using learned logistic regressions parameters passed in as theta
% and the features passed in as X
%
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % m equals the number of training examples

% Return variables initialzied
p = zeros(m, 1);
q = zeros(m, 1);

%  ******************************************
%  Make the predciton on matrix of samples X
%  Result = p is a vector of 0's and 1's
%  ******************************************
for i = 1:m
q(i) = sigmoid (sum ((theta' .* X(i,:))));
if ( q(i) >= 0.5)
	p(i) = 1;
	else
	p(i) = 0;
endif
endfor
% Need code here for matching the predcitons iwth a y vector if provided and outputting
% success rate. Maybe?? this could be handled outside of the predict routine to keep it clean.

end
