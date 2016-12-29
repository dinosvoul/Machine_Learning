%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Group ID : 329
% Members : Alexandar Arambašic, Juraj Peršic, Konstantinos Voulgaris,
% Giannis Kapnisakis, Ricard Bordalba
% Date : 14/10/2015 
% Lecture: 7 Multilayer perceptions
% Dependencies: 
% Matlab version: 
% Functionality: PCA 2 dimensional reduced data is classified according to
% MLP.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%Load reduced data from PCA: z, length of 5 l5, length of 6 l6, length of 8 l8
load './data/data.mat'

% 2 dimensional data
X = z;
figure;%plot reduced data not classified
scatter(z(:,1),z(:,2))

%We create the structure of the network MLP
% nin = number of inputs(dimensions), the number of features our data has
% nout = number of outputs, the number of classes we have or we want.
% nhidden = number of hidden units, it's a tunning parameter, 100, 200,...
NIN = 2; NHIDDEN = 10; NOUT = 3;
NET = mlp(NIN, NHIDDEN, NOUT, 'softmax') 

%train an MLP network:
% Target data T. we want 3 classes so ideally we would want 1,2,3 as the
% output(target), but it gives better results if we use [1 0 0][0 1 0][0 0 1]
% X is the input data, so the train data to find the network model.


%label data according to the one given
for i=1:l5
 T(i,:) = [1 0 0];
end
for i=l5+1:l6+l5
 T(i,:) = [0 1 0];
end
for i=l6+l5+1:l5+l6+l8
 T(i,:) = [0 0 1];
end

ITS = 500;
[NET, ERROR] = mlptrain(NET, X, T, ITS);
% trains a network data
% structure NET using the scaled conjugate gradient algorithm  for ITS
% cycles with input data X, target data T.
% ITS are iterations


% Now use this NET model to classify data.
% NET is the network that we defined and then trained in order to gives us
% the desired output (classification).
% Y gives the classification label to the data X
Y = mlpfwd(NET, X); 

%from the output function of 3 dimensions [a b c], we choose the closest
%number to 1. (the highest)
success = 0;
for i = 1: l5+l6+l8
    maxpos = find(Y(i,:) == max(Y(i,:)));
    Y(i,maxpos) = 1;
    if Y(i,maxpos) == T(i,maxpos)
        success = success + 1;
    end
end

rate = success/(l5+l6+l8)