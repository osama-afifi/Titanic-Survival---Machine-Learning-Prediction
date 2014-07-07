%% Osama M. Afifi
%% 7/07/2013
%% Titanic Survival Prediction

%  Procedure
%  ------------
% 
%	1.1  Load and Manipulate the Data.
%	1.2  Data Synthesis (Optional).
%	2    Initializing Parameters
%	3.1  Train Neural Network
%	3.2  Find Suitable Reg. Parameter
%	3.3  Find Suitable Iteration Limit
%	4	 Visualize Hidden Layer Weights
%	5	 Predict Labels and Calculate Accuracy Perc.

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this program
input_layer_size  = 10;  
hidden_layer_size = 5;   
num_labels = 2;          
                         
						  
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('.................... Phase 1 .......................\n')
fprintf('Loading Data File ...\n')
Data = csvread('../Data/train2.csv');
fprintf('Setting up Label Vector ...\n')
y = Data(:,5);
fprintf('Setting up Feature Matrix ...\n')
feature_columns = [1, 2, 3, 4];
Class = Data(:,3);
Sex = Data(:,5);
Age = Data(:,6);

Sex( Sex == "male" )= "1";
Sex( Sex == "female" )= "1";
str2num (Sex);

size(X,1)
size(X,2)
m = size(X, 1);
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Parameters ================
%  A two layer neural network that classifies digits. we will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)
fprintf('.................... Phase 2 .......................\n')
%warning('off', 'Octave:possible-matlab-short-circuit-operator');
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Part 3: Training NN ===================
%  To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.

fprintf('.................... Phase 3 .......................\n')
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 4: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%
fprintf('.................... Phase 7 .......................\n')
lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 5: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%
fprintf('.................... Phase 5 .......................\n')
[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 6: Predict =================
%  After training the neural network, we would like to use it to predict the labels of the training set. This lets
%  you compute the training set accuracy.

fprintf('.................... Phase 6 .......................\n')
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% ================= Part 7: Predict Testing Data =================
%  After training the neural network, we would like to use it to predict the labels of the tesring data

fprintf('.................... Phase 7 .......................\n')
XTest = load('Data/test.csv');
predTest = predict(Theta1, Theta2, XTest);
numLables = ([1:size(predTest,1)])';
predTest = [numLables predTest];
csvwrite ('predTest.csv', predTest);
