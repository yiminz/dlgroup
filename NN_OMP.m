%% assumed one atom per note

function [S,H] = NN_OMP(Data, Dictionary, k_sparse)

G = Dictionary' * Dictionary;

[~,time_bins] = size(Data);
[~,num_atoms] = size(Dictionary);

H = zeros(num_atoms, time_bins);
S = zeros(num_atoms, time_bins);

Residual = Data;

XtY = Dictionary' * Data;

k = zeros(1,size(Data,2));
converged = zeros(1,size(Data,2));
% atoms_selected = converged;
iter = 0;

while any(k_sparse > k) && iter < max(k_sparse)
    
    iter = iter + 1;
    
    t_ind = find(k < k_sparse);
    
    %tic
    XtR = Dictionary' * Residual(:,t_ind);

    for ta = 1:length(t_ind)
        
        t = t_ind(ta);
        
        [value,ind] = max(XtR(:,ta));
        
        if value < 0
           converged(t) = 1; 
        else
            
            S(ind, t) = 1;
        
            current_support = find(S(:,t) == 1);

            % get the group that this belongs
            
            H(current_support, t) = fnnls2(G(current_support, current_support), XtY(current_support, t));
            
        end
    end
    
    Residual = Data - (Dictionary * H);
    
    k = sum(S);
    cind = find(converged == 1);
    k(cind) = k_sparse(cind);
        
end
    
end
    
    