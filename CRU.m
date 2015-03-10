%%MAT Consulting Chems R Us Project

%% DatasetA.csv Creating and dividing the Data

% Read the DatasetA.csv Data in the CSV file 
% and extract the matrix D of observations and vector y of class labels (1 or -1).  
Doriginal = csvread('DatasetA.csv');

Y=Doriginal(:,end);   % Y contains the class labels 1 or -1
D=Doriginal(:,1:(end-1));  % All the rest are the features 

%Transform the Data Matrix D to have mean 0 and standard deviation 1.   
s=std(D);
a = diag(1./s);
[m,n] = size(D);
one_m = ones(m,m);

D = (D - (1/m)*(ones(m,m)*D))*a; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Training and testing matrices for DatasetA

% Classp_train  := Class 1 training data
% Classm_train  := Class -1 training data
% Classp_test   := Class 1 testing data
% Classm_test   := Class -1 testing  data

% Set random number to an initial seed
[r,c]=size(D);
s=RandStream('mt19937ar','Seed',550);
%generate a permutation of the data
p=randperm(s,r);
D=D(p,:);
Y=Y(p);
%Use trainpct percent of the data for training and the rest for testing.
trainpct=.90;
train_size=ceil(r*trainpct);

% Grab training and test data
Train = D(1:train_size,:);
Test = D(train_size+1:end,:);
YTrain = Y(1:train_size,:);
YTest = Y(train_size+1:end,:);

%Break them up into Class 1 and Class -1
Classp_train = Train(YTrain==1,:);
Classm_train = Train(YTrain==-1,:);

Classp_test = Test(YTest==1,:);
Classm_test = Test(YTest==-1,:);

%% Mean Method DatasetA.csv

%Calculate the mean classifier 

% Calculate w as difference of the class means
meanp=mean(Classp_train);
meanm=mean(Classm_train);
w=(meanp-meanm)';
w=w/norm(w);

t= (meanp+meanm)/2*w;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Count the number of training  points missclassified in each class. 

MeanPosErrorTrain = sum(Classp_train*w <= t);
MeanNegErrorTrain = sum(Classm_train*w >= t);

MeanTrainError = ((MeanPosErrorTrain + MeanNegErrorTrain)/(size(Train,1)))

%Calculate the testing error of the Mean Method

MeanPosErrorTest = sum(Classp_test*w <= t);
MeanNegErrorTest = sum(Classm_test*w >= t);

MeanTestError = ((MeanPosErrorTest + MeanNegErrorTest)/(size(Test,1))) 

HistClass(Classp_train,Classm_train,w,t,...
    'Mean Method Training Results',MeanTrainError); %Histogram of Mean Training Results

HistClass(Classp_train,Classm_train,w,t,...
    'Mean Method Testing Results',MeanTestError); %Histogram of Mean Testing Results

%% DatasetA.csv Fisher Method
% Construct the Fisher Linear Discriminant
% Calculate  the Fisher LDA using Classp_train and Classm_train 
% for training data.

psize=size(Classp_train,1);
nsize=size(Classm_train,1);
Bp=Classp_train-ones(psize,1)*meanp;
Bn=Classm_train-ones(nsize,1)*meanm;
Sw=Bp'*Bp+Bn'*Bn;
wfisher = Sw\(meanp-meanm)';
wfisher=wfisher/norm(wfisher)

tfisher=(meanp+meanm)./2*wfisher
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Analyze training data  results of the Fisher Linear Discriminant

FisherPosErrorTrain = sum(Classp_train*wfisher <= tfisher)
FisherNegErrorTrain = sum(Classm_train*wfisher >= tfisher)

FisherTrainError= ((FisherPosErrorTrain + FisherNegErrorTrain)/(size(Train,1)))  

HistClass(Classp_train,Classm_train,wfisher,tfisher,...
    'Fisher Method Training Results',FisherTrainError); % Histogram of Fisher Training Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FisherPosErrorTest = sum(Classp_test*wfisher <= tfisher);
FisherNegErrorTest = sum(Classm_test*wfisher >= tfisher);

FisherTestError= ((FisherPosErrorTest + FisherNegErrorTest)/(size(Test,1)))   

HistClass(Classp_test,Classm_test,wfisher,tfisher,...
    'Fisher Method Testing Results',FisherTestError); % Histogram of Fisher Testing Results