                                                                                                                                      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Group ID : 329
% Members : Alexandar Arambašic, Juraj Peršic, Konstantinos Voulgaris,
% Giannis Kapnisakis, Ricard Bordalba
% Date : 14/10/2015 
% Lecture: 6 Linear Discrimination
% Dependencies: 
% Matlab version: 
% Functionality: Use LDA to reduce data of 3 classes to 2 features, find 
% a model for the train data and then do a classification with the test
% data. this can later be compared with PCA algorithm results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%Load data
load './data/mnist_all.mat'

% Put the data together
data = double([train5; train6; train8]);
%label data
gnd = [5*ones(length(train5),1); 6*ones(length(train6),1); 8*ones(length(train8),1)];

% Use the LDA algorithm using fisherface approach: gnd is the vector
% classifying the train data.
options = [];
options.Fisherface = 1; %use fisher face approach
[eigvector, eigvalue] = LDA(gnd, options, data);
%we obtain the eigenvector that gives the maximum distance between the
%classes, as then it will be easier to find different models to classify
%We use the eigenvector to reduce the dimensions of data
%The dimension of reduced data Y will be the number of classes of data -1:
%3classes-1=2 dimensions in this case.

%the eigenvector is perpendicular to the   w vector.
Y = data*eigvector; 

figure;%plot reduced data not classified
scatter(Y(:,1),Y(:,2))

figure;%plot reduced classified data
scatter(Y(1:length(train5),1),Y(1:length(train5),2),'r')
hold on
scatter(Y(length(train5):length(train5)+length(train6),1),Y(length(train5):length(train5)+length(train6),2),'g')
scatter(Y(length(train5)+length(train6)+1:length(data),1),Y(length(train5)+length(train6)+1:length(data),2),'b')

%Find model of this data, but first we reduce train data to 2 dimensions
train5 = double(train5)*eigvector;
train6 = double(train6)*eigvector;
train8 = double(train8)*eigvector;
mean5 = mean(train5);
cov5 = cov(train5);
mean6 = mean(train6);
cov6 = cov(train6);
mean8 = mean(train8);
cov8 = cov(train8);

%COnvert test data with the same eigenvectors
test_data = double([test5; test6; test8]);
test_data = test_data*eigvector;

%check for probabilities (pdf) of class 5, 6 or 8 for the train data z
%created (reduced to 2 dimensions)
pr_5 = mvnpdf(test_data,mean5,cov5);
pr_6 = mvnpdf(test_data,mean6,cov6);
pr_8 = mvnpdf(test_data,mean8,cov8);

% Make a simple classification rule between 3 options: check higher
% Probability. Then compare with the train data
class = 5*(pr_5>pr_6 & pr_5>pr_8)+6*(pr_6>pr_5 & pr_6>pr_8)+8*(pr_8>pr_6 & pr_8>pr_5);
succes5 = length(find(class(1:length(test5))==5))/length(test5)
succes6 = length(find(class(length(test5)+1:length(test5)+length(test6))==6))/length(test6)
succes8 = length(find(class(length(test5)+length(test6)+1:length(test_data))==8))/length(test8)
