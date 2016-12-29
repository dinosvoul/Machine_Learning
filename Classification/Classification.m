%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Group ID : 329
% Members : Alexandar Arambašic, Juraj Peršic, Konstantinos Voulgaris,
% Giannis Kapnisakis, Ricard Bordalba
% Date : 16/09/2015
% Lecture: 3 Parametric and nonparametric methods
% Dependencies: file 'dataset1_G_noisy.mat'
% Matlab version: 
% Functionality: This script find a model from train data and then
% classifies handwritten digits. It also test the error obtained compared
% to known data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

%a) classify instances in tst_xy, and use the corresponding label file 
% tst_xy_class to calculate the accuracy;

% load Data
load './data/dataset1_G_noisy.mat'

%plot known data to find model
scatter(trn_x(:,1),trn_x(:,2),'b')
hold on
scatter(trn_y(:,1),trn_y(:,2),'r')

% gaussian distribution in two dimension, so we need mean and covariance in
% order to find the models
mean_x = mean(trn_x)
cov_x = cov(trn_x)

mean_y = mean(trn_y)
cov_y = cov(trn_y)

% calculating probabilities for the test data
%check for probabilities (pdf) of class x and class y
pr_x = mvnpdf(tst_xy,mean_x,cov_x);
pr_y = mvnpdf(tst_xy,mean_y,cov_y);

%Total of test data
total = length(trn_x) + length(trn_y);

%PRobabilities to have which class
P_x = length(trn_x) / total;
P_y = length(trn_y) / total;

% calculate g_i(x)
g_x = pr_x * P_x;
g_y = pr_y * P_y;

%Compute the class and then compare with the know class
class_xy=(g_x < g_y)+ones(size(tst_xy_class));
err_a = sum(abs(class_xy-tst_xy_class))/length(tst_xy)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%b)
disp('press to see part b)')
pause;

% calculating probabilities for the test data
%check for probabilities (pdf) of class x and class y
pr_x_b = mvnpdf(tst_xy_126,mean_x,cov_x);
pr_y_b = mvnpdf(tst_xy_126,mean_y,cov_y);

% calculate g_i(x), P is 0.5 as its uniformely distributed
g_x = pr_x_b * 0.5;
g_y = pr_y_b * 0.5;

%Compare
class_xy_b=(g_x < g_y)+ones(size(tst_xy_126_class));
err_b = sum(abs(class_xy_b-tst_xy_126_class))/length(tst_xy_126_class)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%c)
disp('press to see part c)')
pause;

%check for class x and y
pr_x_c = mvnpdf(tst_xy_126,mean_x,cov_x);
pr_y_c = mvnpdf(tst_xy_126,mean_y,cov_y);

% calculate g_i(x), P is 0.9 for x, and 0.1 for y
g_x = pr_x_c * 0.9;
g_y = pr_y_c * 0.1;

%Compare
class_xy_c=(g_x < g_y)+ones(size(tst_xy_126_class));
err_c = sum(abs(class_xy_c-tst_xy_126_class))/length(tst_xy_126_class)