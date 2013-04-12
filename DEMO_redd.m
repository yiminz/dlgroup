%==========================================================================
% Compare results for ILS-DLA (MOD), K-SVD and RLS-DLA. 
%                 RLS-DLA is here used without forgetting factor lambda.
% An AR(1) signal is used here. Sparseness is s=4. 
% Dictionary size is N=16, K=32
%
% demo copied (and modified) from http://www.ux.uis.no/~karlsk/dle
% many thanks to Karl Skretting.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.uis.no/~karlsk/
% 
% [ompbox10] is used.
%
% Author : liubenyuan@gmail.com
% Date   : 2013-04-06
%==========================================================================

%% 1. load data and optimum dictionary
load('refrigerator.mat');

N = 60;        % dimension of the test signal
L = 4000;      % snapshots
K = 2*N;       % dictionary
S = 4;         % sparsify
maxit = 80;    % maximum iterations

len = length(s);
X = zeros(N,L);
for i = 1 : L
    j = round(rand(1)*(len-N))+1;
    
    sig1 = s(j:j+N-1);
    X(:,i) = sig1;
end
clear s;

%% play with mod_dla
[Dksvd,~,err] = mod_nn_dla(X,K,S,maxit);

figure; plot(10*log10(1./err)); title('K-SVD error convergence');
xlabel('Iteration'); ylabel('RMSE');

fprintf('  Dictionary size: %d x %d', N, K);
fprintf('  Number of examples: %d', L);

% [d,~] = dictdiff(Dksvd,D);
% fprintf('  Dictionary Distances = %.4f\n', d);
% 
% [~,ratio] = dictdist(Dksvd,D);
% fprintf('  Ratio of recovered atoms: %.2f%%\n', ratio*100);

figure
imagesc(Dksvd); colorbar;

save('D_redd.mat','Dksvd');

% plot some basis
load('D_redd.mat');
bas = [1 7 11 17 19 31 37 46 51];
figure
for i = 1:length(bas)
    j = bas(i);
    subplot(3,3,i); plot(Dksvd(:,j),'LineWidth',2); title(['basis=' num2str(j)]); axis tight;
end