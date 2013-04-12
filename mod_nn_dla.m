%==========================================================================
% The Method of Optimized Directions (MOD)
%
% liubenyuan@gmail.com
% 2013-04-07
%==========================================================================
function [Dict,W,err] = mod_nn_dla(X,K,S,maxiter)

% initialize column-normalized dictionary
% data_ids = find(colnorms_squared(X) > 1e-6); % ensure no zero data elements are chosen
% perm = randperm(length(data_ids));
% D = X(:,data_ids(perm(1:K)));
N = size(X,1);
D = randn(N,K);
D(D<0) = 0;
D = normcols(D);

sumXX = sum(colnorms_squared(X)); % sumXX = sum(sum(X.*X));
err = zeros(maxiter,1);

L = size(X,2);
Sv = S.*ones(1,L); % vectors of S

% main loop
for i = 1 : maxiter
    % 1. calculate W
    [~,W] = NN_OMP(X, D, Sv);
    R = X - D*W;
    % 2. update D
    B = X*W';
    A = W*W';
    D = B/A;
    D(D<0) = 0;
    D = normcols(D);
    % SNR = 10*log10(sumXX/sum(sum(R.*R)));
    err(i) = sum(colnorms_squared(R.*R))/sumXX;
    
    fprintf('iteration %3d / %3d maxiter (ERR=%f) \n',i,maxiter,err(i));
end

Dict = D;

%==========================================================================
% codes borrowed from ksvd by Ron Rubinstein
function Y = colnorms_squared(X)

% compute in blocks to conserve memory
Y = zeros(1,size(X,2));
blocksize = 2000;
for i = 1:blocksize:size(X,2)
    blockids = i : min(i+blocksize-1,size(X,2));
    Y(blockids) = sum(X(:,blockids).^2);
end

function y = normcols(x)
%NORMCOLS Normalize matrix columns.
%  Y = NORMCOLS(X) normalizes the columns of X to unit length, returning
%  the result as Y.
%
%  See also ADDTOCOLS.

%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  April 2009

y = x*spdiag(1./sqrt(sum(x.*x)));

function Y = spdiag(V,K)
%SPDIAG Sparse diagonal matrices.
%   SPDIAG(V,K) when V is a vector with N components is a sparse square
%   matrix of order N+ABS(K) with the elements of V on the K-th diagonal. 
%   K = 0 is the main diagonal, K > 0 is above the main diagonal and K < 0
%   is below the main diagonal. 
%
%   SPDIAG(V) is the same as SPDIAG(V,0) and puts V on the main diagonal.
%
%   See also DIAG, SPDIAGS.

%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  June 2008

if (nargin<2)
  K = 0;
end

n = length(V) + abs(K);

if (K>0)
  i = 1:length(V);
  j = K+1:n;
elseif (K<0)
  i = -K+1:n;
  j = 1:length(V);
else
  i = 1:n;
  j = 1:n;
end

Y = sparse(i,j,V(:),n,n);