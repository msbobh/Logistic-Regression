%
% Routine Definition: 
%  Trains a logistic regression classifier and then saves the parameters file
%  to "parameters.dat".
%  
%  RegLogReg_Train:
%     Trains a logistic regression classifer using the training set "trainingset.mat", 
%     initializes theta to all zeros matching the size of the training set (X)
%     Calls fminunc - to minimize the cost fuction for the trainign set using 
%     the cost function routine in costFunction.m which calls sigmoid.m to calcuate
%     the sigmoind function on the training set. The trained parameters are saved in
%     a file named Paramaters.dat
%
% Uses the following subroutines
%     sigmoid.m
%     costFunction.m – For optimization might not need this one
%     predict.m -  accepts a set of parameters and a matrix of test values and returns a vector of predictions 
%     costFunctionReg.m – Calculates the cost function "J" and the gradient as required
%     by the optimization algorithm. Also implements either L1 or L2 regularization
%
%  
%

%% Initializations
clear ; close all; clc
lambda = .01;

%% Load Data Files
loadfile = 'trainingset.mat';
labelfile = 'labels.mat';
fprintf("Loading the training set (%s), could take a while\n",loadfile);
load(loadfile);
fprintf("Loaded(%s), matrix dimensions:%d x %d\n",loadfile, rows(X), columns(X));
load (labelfile); %This loads the labels from training set and leaves them in y
fprintf("Loaded labels (%s), matrix dimensions: %d x %d\n",labelfile, rows(y), columns(y));


% Initialize fitting parameters all zeros n x 1 vector of zeros
initial_theta = zeros(size(X, 2), 1);

% Set Options for Octave version of fminunc, version of unconstrained linear optimization method.
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Call the optimization routine
fprintf('Calling optimization rountine... this could take a while\n');
[theta, J, exit_flag] = ...
	fminunc(@(t) (costFunctionReg(t,X,y, lambda)), initial_theta, options);


p = predict(theta, X); % X = training set data 

% Save calculated theta vaules in a data file
save "parameters.dat" theta;

fprintf('Training Set Accuracy: %f, with Lambda = %d\n', mean(double(p == y)) * 100, lambda);

fprintf(' Loading testing data\n');
load testset.mat;
load testlabels.mat;
fprintf(' Test data %d x %d, test labels %d x %d\n',rows(X),  columns(X), rows(y), columns(y));
p = predict(theta,X);
fprintf('Test set accuracy: %f\n', mean(double(p == y)) * 100);

