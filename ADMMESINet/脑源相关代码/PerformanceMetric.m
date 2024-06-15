% Discription: This script is to calculate the perfomance metric for a
% specified localization method
% Two metrics:
% (1)spatial dispersion (SD), which measures how source locations are spatially
%   blurred;
% (2) distance of localization error (DLE), which measures the 
%   averaged error distance between the true source
%   locations and the maximum of source estimates.
% (3) RMSE: To assess the ability to recover the theoretical distribution of current s_real with an accurate amplitude

% Input : GridLoc : 3D locacation of each dipole within the brain
%         s: reconstructed currents
%         s_real: simulated currents(Ground Truth)
% Output: SD and DLE

% Reference : (1)Chang W, Nummenmaa A, Hsieh J, Lin F. 
% Spatially sparse source cluster modeling by compressive neuromagnetic tomography. 
% Neuroimage 53(1): 146--160, 2010.
% (2) Grova C, Daunizeau J, Lina JM, Benar CG, Benali H, Gotman J. 
% Evaluation of EEG localization methods using realistic simulations of interictal  spikes. 
% Neuroimage 29(3): 734-53, 2006.

% Author : Ke Liu
% Date : 2013/11/15


function [SD,DLE,RMSE,nRMSE,SE] = PerformanceMetric(GridLoc,s,s_real,ActiveVoxSeed,varargin)

% SD = 0; DLE = 0;
TimeInterval = 1:size(s,2);
for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'interval'   
                TimeInterval = varargin{i+1}; %TimeInterval = MetricsInterval=[]
            otherwise
                error(['Unrecognized parameter: ''' varargin{i} '''']);
        end
end
if isempty(TimeInterval)
    TimeInterval = 1:size(s,2);%1,2,3...n�����еĴ�С
end
          keeplist = (1:size(s,1))';  %1,2,3...ds �����еĴ�С       
          P = sqrt(sum(s(:,TimeInterval).^2,2));%sum(x,2)��x�������
%           index = (P~=0);
          index = (P>=0.10*max(P));%index��0,1�߼�ֵ��ż���ӵ������������������10%
          keeplist = keeplist(index);%keeplistֻ����indexΪ1���У�indexΪ0���в�Ҫ��keeplist:�ع�Դs�Ļż��������
          ActiveVox = find(mean(s_real(:,TimeInterval).^2,2)~=0);%mean(x,2)��x�������ֵ,��sƽ�������еľ�ֵ�������ƽ�������������ҵ���Ϊ0�����꣬activevox��ģ��Դs_real�Ļż��������
          I = SpatialNeigh(ActiveVox,GridLoc,keeplist);
          SD = SpatialDispersion(s(:,TimeInterval),I,ActiveVox,GridLoc);
          DLE = DisLE(s(:,TimeInterval),I,ActiveVox,GridLoc);

          
%           s = s.*norm(s_real,'fro')/norm(s,'fro'); % Normalize

          s = s./norm(s,'fro');
          s_real = s_real./norm(s_real,'fro');
          %% MSE within the whole cortex
          RMSE = norm(s-s_real,'fro')^2/norm(s_real,'fro')^2;
          %% MSE within the active areas
          nRMSE = norm(s(ActiveVox,:)-s_real(ActiveVox,:),'fro')^2/norm(s_real(ActiveVox,:),'fro')^2; 
          %% Correlation coefficient between mean simulated and estimated time courses within the active areas
          c = zeros(numel(ActiveVoxSeed),1);
          for i = 1:numel(ActiveVoxSeed)
              c(i) = corr2(mean(s(ActiveVoxSeed{i},:),1),s_real(ActiveVoxSeed{i}(1),:));
          end
          SE = mean(c);
%           smean  = mean(s(ActiveVox,:))./max(mean(s(ActiveVox,:)));
%           srealmean = mean(s_real(ActiveVox,:))./max(mean(s_real(ActiveVox,:)));
%           SE = corr2(smean,srealmean);%norm(smean - srealmean,'fro')^2;
end


        
function I = SpatialNeigh(ActiveVox,GridLoc,keeplist)  
% only the dipoles whose power are larger than 1% of the maximum power is
% taken into consideration;
% keeplist: the index number of dipoles whose power is larger than 10% of
% the maximum power
    I = cell(numel(ActiveVox),1);  %ActiveVox�����������10%��ż���� ,  ģ��Դs_real�Ļż�������� 
    for j =1:numel(keeplist)%�ع�Դs�Ļż��������
%          distance = zeros(numel(ActiveVox),1);
%        for k =1 :numel(ActiveVox)  %k�����ż����
%            distance(k) = norm(GridLoc(keeplist(j),:)-GridLoc(ActiveVox(k),:));%ģ��Դk���ع�Դj֮��ľ���
%        end
       distance = sqrt(sum((GridLoc(ActiveVox,:)-repmat(GridLoc(keeplist(j),:),numel(ActiveVox),1)).^2,2));
       
       [~,IX] = sort(distance,'ascend');%���������Ȼ��IX���δ����ԭ��������
       I{IX(1)} = union(I{IX(1)},keeplist(j));%IX(1)��������ż���ӵ�����
%           end
    end
end   
    
    
    
function SD = SpatialDispersion(s,I,ActiveVox,GridLoc) 
     SD = 0;
    for i = 1:numel(ActiveVox)
        if numel(I{i})~=0
            for j = 1:numel(I{i})
                SD = SD + norm(GridLoc(I{i}(j),:)-GridLoc(ActiveVox(i),:))^2*sum(s(I{i}(j),:).^2);
            end
        end
    end
    SD = SD/sum(sum(s.^2));
    SD = sqrt(SD);
end
  
    
    
 function DLE = DisLE(s,I,ActiveVox,GridLoc)    
    J = [];
    for i =1 :numel(ActiveVox)
        if numel(I{i})~=0
            J = union(J,i);
        end
    end
    distance = zeros(numel(J),1);
    for i = 1:numel(J)
            power = sum(s(I{J(i)},:).^2,2);
            [~,IX] = sort(power,'descend');
            distance(i) = norm(GridLoc(ActiveVox(J(i)),:)-GridLoc(I{J(i)}(IX(1)),:));
    end
    DLE = mean(distance);
 end
