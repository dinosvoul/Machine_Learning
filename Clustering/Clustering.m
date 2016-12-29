                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Group ID : 329
% Members : Alexandar Arambašic, Juraj Peršic, Konstantinos Voulgaris,
% Giannis Kapnisakis, Ricard Bordalba
% Date : 7/10/2015 
% Lecture: 5 Clustering
% Dependencies: 
% Matlab version: 
% Functionality: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all
% close all
% clc

%Load data
load './data/data.mat'
z = z';
% This assignment is based on the previously generated 2-dimensional data 
%of the three classes (5, 6 and 8) from the MNIST database of handwritten 
% digits. First, mix the 2-dimensional data (training data only) by 
%removing the labels and then use one Gaussian mixture model to model them.

%plot raw 2 dimensional data
scatter(z(:,1),z(:,2))

% random means of the 3 gaussian models
m1 = [0 1000];
m2 = [1000 1000];
m3 = [0 -1000];
%Random variances
var1 = 500*eye(2);
var2 = 500*eye(2);
var3 = 500*eye(2);

count = 0;
response = 0;
%pause;
figure(2);
while (response == 0)
    %Classify according to probabilities
% Probability that z belongs to each Gaussian model
P1 = mvnpdf(z,m1,var1);
P2 = mvnpdf(z,m2,var2);
P3 = mvnpdf(z,m3,var3);

for i = 1:length(P1)
    if P1(i) >= P2(i) && P1(i) >= P3(i)
        class(i) = 1;
    else if P2(i) > P1(i) && P2(i) > P3(i)
        class(i) = 2;    
    else
        class(i) = 3;    
        end
    end
end
data1 = [z(find(class == 1),1),z(find(class == 1),2)];
data2 = [z(find(class == 2),1),z(find(class == 2),2)];
data3 = [z(find(class == 3),1),z(find(class == 3),2)];

%Recalculate new mean
m1 = mean(data1,1);
m2 = mean(data2,1);
m3 = mean(data3,1);

var1 = cov(data1);
var2 = cov(data2);
var3 = cov(data3);

count = count + 1;

if count == 10
    scatter(data1(:,1),data1(:,2),'r')
    hold on
    scatter(data2(:,1),data2(:,2),'g')
    scatter(data3(:,1),data3(:,2),'b')

    response = input('Write 0 to continue. Else to stop');
    count = 0;
end
end

% Show the final models found (mean and covariance)
m1 = mean(data1,1)
m2 = mean(data2,1)
m3 = mean(data3,1)

var1 = cov(data1)
var2 = cov(data2)
var3 = cov(data3)

%Secondly, compare the Gaussian mixture model with the Gaussian models 
%trained in the previous assignment.

