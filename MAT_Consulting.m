%% MAT Cousulting for Chem R US by Thomas Wagner, Alexander Allen, MingYi Wang
% March 19 2015

%% DatasetA processing
%read in D
Doriginal = csvread('DatasetA.csv');

%Break D into id's, class, and features
IDA=Doriginal(:,1); %id column
Class=Doriginal(:,end);   % Y contains the class labels 1 or -1
DA=Doriginal(:,2:(end-1));  % All the rest are the featuresUnIDA=Doriginal(:,2:end);

%% define positive class and calculate mean and covariance
DAp = DA(Class==1,:); %Class 1 of DSA;
[mp,np]=size(DAp);
DAp_mean = (1/mp)*ones(1,mp)*DAp;

figure
imagesc(DAp_mean)
title('Mean Vector of Class 1 DatasetA')
colormap(gray)
colorbar

DAp_centered= DAp - (1/mp)*(ones(mp,mp)*DAp);

CovAp = (1/(mp-1))*DAp_centered'*DAp_centered; %Covariance of class 1

figure 
imagesc(CovAp)
title('Covariance Matrix of Class 1 DatasetA')
colormap(gray)
colorbar

%% define negative class and caluclate mean and covariance

DAn = DA(Class==-1,:); 
[mn,nn]=size(DAn);
DAn_mean=(1/mn)*ones(1,mn)*DAn;

figure
imagesc(DAn_mean)
title('Mean Vector of Class -1 DatasetA')
colormap(gray)
colorbar

DAn_centered=DAn - (1/mn)*(ones(mn,mn)*DAn);

CovAn = (1/(mn-1))*DAn_centered'*DAn_centered;

figure 
imagesc(CovAn)
title('Covariance Matrix of Class -1 DatasetA')
colormap(gray)
colorbar

%% Define testing and trianing sets

% Training and testing matrices for DatasetA

% Classp_train  := Class 1 training data
% Classm_train  := Class -1 training data
% Classp_test   := Class 1 testing data
% Classm_test   := Class -1 testing  data


% Set random number to an initial seed
[r,c]=size(DA);
s=RandStream('mt19937ar','Seed',550);
%generate a permutation of the data
p=randperm(s,r);
DA=DA(p,:);
Y=Class(p);
%Use trainpct percent of the data for training and the rest for testing.
trainpct=.90;
train_size=ceil(r*trainpct);

% Grab training and test data
Train = DA(1:train_size,:);
Test = DA(train_size+1:end,:);
YTrain = Y(1:train_size,:);
YTest = Y(train_size+1:end,:);

%Break them up into Class 1 and Class -1
Classp_train = Train(YTrain==1,:);
Classm_train = Train(YTrain==-1,:);

Classp_test = Test(YTest==1,:);
Classm_test = Test(YTest==-1,:);

%% Mean Method on DatasetA


%Calculate the mean classifier 

% Calculate w as difference of the class means
meanp=mean(Classp_train);
meanm=mean(Classm_train);
w=(meanp-meanm)';
w=w/norm(w);

t= (meanp+meanm)./2*w;
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


%% Fisher method on DatasetA

meanp=mean(Classp_train);
meanm=mean(Classm_train);

psize=size(Classp_train,1)
nsize=size(Classm_train,1)
Bp=Classp_train-ones(psize,1)*meanp
Bn=Classm_train-ones(nsize,1)*meanm;

Sw=Bp'*Bp+Bn'*Bn;
wfisher = Sw\(meanp-meanm)';
wfisher=wfisher/norm(wfisher)

tfisher=(meanp+meanm)./2*wfisher%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Analyze training data  results of the Fisher Linear Discriminant

FisherPosErrorTrain = sum(Classp_train*wfisher <= tfisher)%
FisherNegErrorTrain = sum(Classm_train*wfisher >= tfisher)%

FisherTrainError= ((FisherPosErrorTrain + FisherNegErrorTrain)/(size(Train,1)))  

HistClass(Classp_train,Classm_train,wfisher,tfisher,...
    'Fisher Method Training Results',FisherTrainError); % Histogram of Fisher Training Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


FisherPosErrorTest = sum(Classp_test*wfisher <= tfisher);
FisherNegErrorTest = sum(Classm_test*wfisher >= tfisher);

FisherTestError= ((FisherPosErrorTest + FisherNegErrorTest)/(size(Test,1)))   

HistClass(Classp_test,Classm_test,wfisher,tfisher,...
    'Fisher Method Testing Results',FisherTestError); % Histogram of Fisher Testing Results

%% KNN classifier on Test set

classifier=knnsearch(Train,Test);
total_error=0;
[s,z]=size(Test)
for i=1:s,
    if(YTest(i)~=YTrain(classifier(i)))
        total_error=total_error+1;
    end
end
error_percent = total_error/s

%% FisherMedian DatasetA

medianp=median(Classp_train);
medianm=median(Classm_train);


BMp=Classp_train-ones(psize,1)*medianp;
BMn=Classm_train-ones(nsize,1)*medianm;

Sw=BMp'*BMp+BMn'*BMn;
wFishMed = Sw\(medianp-medianm)';
wFishMed=wFishMed/norm(wFishMed)

tFishMed=(medianp+medianm)./2*wFishMed

MedFishPosErrorTrain = sum(Classp_train*wFishMed <= tFishMed);
MedFishNegErrorTrain = sum(Classm_train*wFishMed >= tFishMed);
MedFishTrainError = ((MedFishPosErrorTrain + MedFishNegErrorTrain)/(size(Train,1)));

MedFishPosErrorTest = sum(Classp_test*wFishMed <= tFishMed);
MedFishNegErrorTest = sum(Classm_test*wFishMed >= tFishMed);
MedFishTestError= ((MedFishPosErrorTest + MedFishNegErrorTest)/(size(Test,1)));

HistClass(Classp_test,Classm_test,wFishMed,tFishMed,...
    'MedianFisher Method Testing Results',MedFishTestError);

HistClass(Classp_train,Classm_train,wFishMed,tFishMed,...
    'MedianFisher Training Results',MedFishTrainError);

%% DatasetV Analysis


DV = csvread('DatasetV.csv');

IDV=DV(:,1); %id column
DV=DV(:,2:end);

[m,n] = size(DV);

DVmean = (1/m)*ones(1,m)*DV %mean for DatasetV

% image for mean
figure                      
imagesc(DVmean)
title('Mean Vector DatasetV')
colormap(gray)
colorbar
 
%Prediction by fisher LDA method
PClassCount = sum(DV*wfisher > tfisher)
NClassCount = sum(DV*wfisher < tfisher)

classes=ones(m,1);
for i=1:m,
    if(DV(i,:)*wfisher <= tfisher)
        classes(i,1)=-1;
    end 
end

DVLabels = cat(2,IDV,classes);
csvwrite('MAT_Consulting_DSV_prediction.csv',DVLabels);

%Covariance of DV
DV_centered = (DV - (1/m)*(ones(m,m)*DV));
CovDV = (1/(m-1))*DV_centered'*DV_centered


%image of covariance of DatasetV
figure              
imagesc(CovDV)
title('Covariance Matrix of DatasetV')
colormap(gray)
colorbar


DV_labeled = csvread('MAT_Consulting_DSV_prediction.csv');

DVPos = sum(DV_labeled(:,2)==1) %Number of points in DV estimated to be class 1
DVNeg = sum(DV_labeled(:,2)==-1) %Number of points in DB estimated to be class -1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DA ErrorsFisher
DApmis = sum(DAp*wfisher <= tfisher); %Number of erroneous class 1 points using fisher method
DAnmis = sum(DAn*wfisher >= tfisher); %Number of erroneous class -1 points
DAPosError = DApmis/size(DAp,1)
DANegError = DAnmis/size(DAn,1)

TotalDAError = (DApmis+DAnmis)/size(DA,1)%Total error of Fisher on DatasetA
