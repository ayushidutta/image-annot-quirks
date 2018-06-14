
clear

mex tagpropCmt.c

%{
load corel5k.mat

[ model ll ] = tagprop_learn(NN,[],Y);

P = tagprop_predict(NN,[],model);

%}

% Load features

% load distance

% load NN

% Tagprop
[ model ll ] = tagprop_learn(train_NN,train_ND,train_Y,'type','dist','sigmoids',true);
P = tagprop_predict(test_NN,test_ND,model);

% Evaluate
