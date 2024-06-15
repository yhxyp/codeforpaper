function AUC = ROCextent(s_real,s,Cortex,seedvox,varargin)
% Discription: this script is to calculate a unbiased AUC, especially for
% patch source.

% Reference: C. Grova.2006 
% "Evaluation of EEG localization methods using realistic simulations
% of interictal spikes"

% Author: Liu Ke
% Date: 2013/10/30
%       2014/11/15 (Change roc calculate using matlab default)
% Low_limit = 1e-8;
flag = 1; % for most cases, flag = 1; for very sparse solutions (such as SBL and SSI), flag = 2;

% [sSurface, iSurface] = bst_get('SurfaceFileByType',[],'Cortex');
% bst_call(@export_matlab, {char(sSurface.FileName)},'Cortex');

Power_real = sqrt(mean(s_real.^2,2));%行向量的均值
Power = sqrt(mean(s.^2,2));%重构源的能量
ActiveVox = find(Power_real~=0);
nSource = size(s,1);
Low_limit = min(Power(Power~=0))/max(Power(Power~=0));

if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'low_limit'   
                Low_limit = varargin{i+1}; 
            case 'flag'
                flag = varargin{i+1};
        end
    end
end


degree = 10;
Nei = seedvox;
for i =1:degree
     Nei = union(Nei,tess_scout_swell(Nei, Cortex.VertConn));
end

disp('-----Generate near field-----')
% Nei = [];
% [~, VertArea] = tess_area(Cortex);
%     for k = 1:numel(seedvox)
%         N = PatchGenerate(seedvox(k),Cortex.VertConn,VertArea,200*1e-4);
%         Nei = union(N,Nei);
%     end


disp('-----Near field finished-----')


N_close = setdiff(Nei,ActiveVox);
N_far = setdiff((1:nSource)',Nei);

iter = 50;
AUC_far = zeros(iter,1); 
AUC_close = zeros(iter,1); 
FPR_close = []; TPR_close = [];
FPR_far = []; TPR_far = [];
for i = 1: iter
    ind_close = randperm(numel(N_close));
    index_close = union(N_close(ind_close(1:numel(ActiveVox))),ActiveVox);
    ind_far = randperm(numel(N_far));
    index_far = union(N_far(ind_far(1:numel(ActiveVox))),ActiveVox);
    if isempty(Low_limit)
        AUC_close(i) = 0;
        AUC_far(i) = 0;
    else
        [AUC_close(i),temp1,temp2] = ROC_Liu(Power_real(index_close),Power(index_close),'low_limit',Low_limit,'flag',flag);
        % FPR_close = [FPR_close;temp1]; TPR_close = [TPR_close;temp2];
        
        [AUC_far(i),temp1,temp2] = ROC_Liu(Power_real(index_far),Power(index_far),'low_limit',Low_limit,'flag',flag);
        % FPR_far = [FPR_far;temp1]; TPR_far = [TPR_far;temp2];
    end
end
%  figure
% hold on
% plot(mean(FPR_close,1),mean(TPR_close,1))
% % plot(mean([FPR_close;FPR_far],1),mean([TPR_close;TPR_far],1))
AUC.mean = (AUC_close+AUC_far)/2;
AUC.far = AUC_far;
AUC.close = AUC_close;


function [AUC,FPR,TPR] = ROC_Liu(P_real,P,varargin)

if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'low_limit'   
                Low_limit = varargin{i+1}; 
            case 'flag'
                flag = varargin{i+1}; 
        end
    end
end
Nm = numel(P);

if flag == 2 % Especially for very sparse solutions, such as SBL and SSI.
    alfa = logspace(log10(Low_limit),0,100);
    % alfa = linspace(Low_limit,1,100);
    alfa = fliplr(alfa);
    TP = zeros(numel(alfa),1); FN = TP;
    FP = TP; TN = TP;
    for i =1:numel(alfa)
        for j = 1 : Nm
            if P(j)>=alfa(i)*max(P) && P_real(j)~=0
                TP(i) = TP(i)+1;
            elseif P(j)>=alfa(i)*max(P) && P_real(j)==0
                FP(i) = FP(i)+1;
            elseif P(j)<alfa(i)*max(P) && P_real(j)~=0
                FN(i) = FN(i)+1;
            elseif P(j)<alfa(i)*max(P) && P_real(j)==0
                TN(i) = TN(i)+1;
            end
        end
    end
    TPR = TP./(TP+FN);
    FPR = FP./(FP+TN);
    TPR = [0 ; TPR ; 1];
    FPR = [0 ; FPR ; 1];
    AUC = auroc(TPR, FPR);
    AUC = (1-max(FPR))*max(TPR)+AUC;
    
elseif flag == 1
    ActiveVox = find(P_real~=0);
    t = zeros(1,Nm); t(ActiveVox) = 1;
    y = P'./max(P);
%     if ~isempty(find(y(ActiveVox)~=0, 1))
%         y = y + 0.1*min(y( ActiveVox( y(ActiveVox)~=0) ) );
%     end
    [TPR,FPR,thresholds] = roc(t,y);
     AUC = trapz(FPR,TPR);  
    AUC = trapz(FPR,TPR) + 1 - max(FPR); % 矩形
%     AUC = trapz(FPR,TPR) + (1 + max(TPR))*(1 - max(FPR))/2; % 梯形
%     figure
%     plotroc(t,y);

%     AUC = (1-max(FPR))*max(TPR)+AUC;
end




function A = auroc(tp, fp)
%
% AUROC - area under ROC curve
%
%    An ROC (receiver operator characteristic) curve is a plot of the true
%    positive rate as a function of the false positive rate of a classifier
%    system.  The area under the ROC curve is a reasonable performance
%    statistic for classifier systems assuming no knowledge of the true ratio
%    of misclassification costs.
%
%    A = AUROC(TP, FP) computes the area under the ROC curve, where TP and FP
%    are column vectors defining the ROC or ROCCH curve of a classifier
%    system.
%
%    [1] Fawcett, T., "ROC graphs : Notes and practical
%        considerations for researchers", Technical report, HP
%        Laboratories, MS 1143, 1501 Page Mill Road, Palo Alto
%        CA 94304, USA, April 2004.
%
%    See also : ROC, ROCCH

%
% File        : auroc.m
%
% Date        : Wednesdaay 11th November 2004 
%
% Author      : Dr Gavin C. Cawley
%
% Description : Calculate the area under the ROC curve for a two-class
%               probabilistic classifier.
%
% References  : [1] Fawcett, T., "ROC graphs : Notes and practical
%                   considerations for researchers", Technical report, HP
%                   Laboratories, MS 1143, 1501 Page Mill Road, Palo Alto
%                   CA 94304, USA, April 2004.
%
% History     : 22/03/2001 - v1.00
%               10/11/2004 - v1.01 minor improvements to comments etc.
%
% Copyright   : (c) G. C. Cawley, November 2004.
%
%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program; if not, write to the Free Software
%    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
%

n = size(tp, 1);
A = sum((fp(2:n) - fp(1:n-1)).*(tp(2:n)+tp(1:n-1)))/2;