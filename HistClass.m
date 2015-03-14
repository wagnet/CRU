
function [ h ] = HistClass( Pdata,Ndata,w,t,Ctitle,Toterror)
%% Plot class histograms and threshold line
% Inputs 
%  Pdata =  data matrix for Class 1
%  Ndata  = data matrix for Class -1 
%  w = normal 
%  t= threhold
%  TotError = total error
%  Title is title of the plot
% Output h is the handle to the figure

%% Start the Figure
h=figure;
hold on;

%%  Calculate Scalar Projections
Cp=Pdata*w;
Cn=Ndata*w;

%%Calculate the binsizes
min_val = min([Cp;Cn]);
max_val = max([Cp;Cn]);
binsize=(max_val-min_val)/20;

%% Plot the histogram  

[n1, xout1] = hist(Cp,min_val + (1:20)*binsize );
[n2, xout2] = hist(Cn,xout1);
bar(xout2,[n1',n2']);

errf = sprintf('- Error %0.2f%% ',Toterror*100);
title(strcat(Ctitle,errf));
xlabel('scalar projection');
ylabel('count');
ylim_val = get(gca,'ylim');
line([t,t],ylim_val,'LineStyle','--','Color','k');
legend('Class 1','Class -1','threshold');
hold off


end

