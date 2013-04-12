function [x,w] = fnnls2(XtX,Xty,tol)

%% same as fnnls but doesn't use the gradient step

%FNNLS	Non-negative least-squares.
%
% 	Adapted from NNLS of Mathworks, Inc.
%
%	x = fnnls(XtX,Xty) returns the vector X that solves x = pinv(XtX)*Xty
%	in a least squares sense, subject to x >= 0.
%	Differently stated it solves the problem min ||y - Xx|| if
%	XtX = X'*X and Xty = X'*y.
%
%	A default tolerance of TOL = MAX(SIZE(XtX)) * NORM(XtX,1) * EPS
%	is used for deciding when elements of x are less than zero.
%	This can be overridden with x = fnnls(XtX,Xty,TOL).
%
%	[x,w] = fnnls(XtX,Xty) also returns dual vector w where
%	w(i) < 0 where x(i) = 0 and w(i) = 0 where x(i) > 0.
%
%	See also NNLS and FNNLSb

%	L. Shure 5-8-87
%	Revised, 12-15-88,8-31-89 LS.
%	(Partly) Copyright (c) 1984-94 by The MathWorks, Inc.

%	Modified by R. Bro 5-7-96 according to
%       Bro R., de Jong S., Journal of Chemometrics, 1997, 11, 393-401
% 	Corresponds to the FNNLSa algorithm in the paper
%
%	
%	Rasmus bro
%	Chemometrics Group, Food Technology
%	Dept. Dairy and Food Science
%	Royal Vet. & Agricultural
%	DK-1958 Frederiksberg C
%	Denmark
%	rb@kvl.dk
%	http://newton.foodsci.kvl.dk/rasmus.html


%  Reference:
%  Lawson and Hanson, "Solving Least Squares Problems", Prentice-Hall, 1974.



% initialize variables
if nargin < 3
    tol = 10*eps*norm(XtX,1)*max(size(XtX));
end

[m,n] = size(XtX);

[m, n] = size(Xty);
if n > m
    Xty = Xty';
    m = n;
end

P = zeros(1,m);
Z = 1:m;
x = P';
z = x;
ZZ=Z;
%w = Xty-XtX*x;
w = Xty;

% set up iteration criterion
iter = 0;
itmax = 3*m;

% outer loop to put variables into set to hold positive coefficients
while any(Z) & any(w(ZZ) > tol) & iter < itmax
    
    iter = iter + 1;
    
    [wt,t] = max(w(ZZ));
    t = ZZ(t);
    P(1,t) = t;
    Z(t) = 0;
    PP = find(P);
    ZZ = find(Z);
    nzz = size(ZZ);

    z(PP)=(Xty(PP)'/XtX(PP,PP));
    
    z(ZZ) = 0;
    z=z(:);
% inner loop to remove elements from the positive set which no longer belong

    while any((z(PP) <= tol)) 

        iter = iter + 1;
        %if iter == itmax
        %    iter;
        %end
        
        QQ = find((z <= tol) & P');
        
        Z(QQ) = QQ;
        P(QQ) = 0;
        
        
        %{
        alpha = min(x(QQ)./(x(QQ) - z(QQ)));
        x = x + alpha*(z - x);
        ij = find(abs(x) < tol & P' ~= 0);
        Z(ij)=ij';
        P(ij)=zeros(1,max(size(ij)));
        %}
        PP = find(P);
        ZZ = find(Z);
        nzz = size(ZZ);
        
        z(PP)=(Xty(PP)'/XtX(PP,PP));
        z(ZZ) = 0;
        %z(ZZ) = zeros(nzz(2),nzz(1));
        %z=z(:);
    end
    x = z;
    w = Xty-XtX(:,PP)*x(PP);
end


 %{
    tic
    gg = inv(XtX(PP,PP));
    z(PP') = Xty(PP)' * gg;
    toc
    %tic
    %%z(PP') = gg' * Xty(PP);
    %toc
    %tic
    %z(PP') = Xty(PP)' * gg;
    %toc
    %}
