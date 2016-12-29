
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Group ID : 329
% Members : Alexandar Arambašic, Juraj Peršic, Konstantinos Voulgaris,
% Giannis Kapnisakis, Ricard Bordalba
% Date : 23/09/2015
% Lecture: 4 Dimensionality reduction
% Dependencies: 
% Matlab version: 
% Functionality: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

%Load data
load './data/mnist_all.mat'

% (1) from the 10-class database, choose three classes (5, 6 and 8) and 
% then reduce dimension to 2. --> train5, train6, train8 are put together
% to create the same PCA (Principal Component Analysis)
data = [train5; train6; train8];

%- Find the mean and variance of each pixel from the train data (directly
%find covariance) (slide 19) 
m = mean(data,1);
c = cov(double(data));

%- In order to find w, we know they are the eigenvector of Covariance matrix
%from the data (slide 20)
[eV,eigen] = eig(c);
%alternative: pca(double(data)) gives eigenvalues of Covariance matrix of
%raw data.

%- The variance will be maximum when w1 is = to eigenvector with higher
%eigenvalues. So we Find 2 largest eigenvalues because we wan to reduce the
%dimension to 2. The corresponding eigenvector of these 2 largest eigenvalues
%will be w1.
sorted = sort(diag(eigen));
max_eig = sorted(end-1:end);


%Find the 2 eigenvecotr of the 2 largest eigenvalues
pos = find(diag(eigen)==max_eig(1));
pos2 = find(diag(eigen)==max_eig(2));
W = [eV(:,pos) eV(:,pos2)];

%Alternative: eigenvalue decomposition
%[max eigenvalues, eigenvectors=W] = eigdec(c,2);

%Find the projection of x on the direction of w --> z=W'·(x-mean)'
z = W'*(double(data)-ones(length(data),1)*m)';

scatter(z(1,:),z(2,:))

%(2) perform 3-class classification based on the generated 2-dimensional data.
% -Classify the train data reduced dimension in 3 groups. The mean that we
% substract is the mean of the 3 train data
t5 = W'*(double(train5)-ones(length(train5),1)*m)';
t6 = W'*(double(train6)-ones(length(train6),1)*m)';
t8 = W'*(double(train8)-ones(length(train8),1)*m)';
figure;
scatter(t5(1,:),t5(2,:),'r')
hold on
scatter(t6(1,:),t6(2,:),'g')
scatter(t8(1,:),t8(2,:),'b')
legend('5','6','8')
%Find the 2D gaussian mean and variance of each train data MODEL
m5 = mean(t5,2);
m6 = mean(t6,2);
m8 = mean(t8,2);

cov5 = cov(t5');
cov6 = cov(t6');
cov8 = cov(t8');

% calculating probabilities for the test data
%check for probabilities (pdf) of class 5, 6 or 8 for the train data z
%created (reduced to 2 dimensions)
pr_5 = mvnpdf(z',m5',cov5);
pr_6 = mvnpdf(z',m6',cov6);
pr_8 = mvnpdf(z',m8',cov8);

% Make a simple classification rule between 3 options: check higher
% Probability. Then compare with the train data
class = 5*(pr_5>pr_6 & pr_5>pr_8)+6*(pr_6>pr_5 & pr_6>pr_8)+8*(pr_8>pr_6 & pr_8>pr_5);
succes5 = length(find(class(1:length(t5))==5))/length(t5)
succes6 = length(find(class(length(t5)+1:length(t5)+length(t6))==6))/length(t6)
succes8 = length(find(class(length(t5)+length(t6)+1:length(z))==8))/length(t8)

%Comment: 5 and 9 are correlated in the graph, so it´s normal we have a bad
%success rate, on the other hand, 6 was clearly easier to detect so the
%high success rate is expected.