function [clAssign,labelCounts_final] = fsa_assign(rho, nneigh, icl, labels)
% fsa_assign This applies the assignment scheme proposed in the FSA/DPC algorithm
%   In descending order of rho, assign to same cluster as its nearest
%   higher density point (indicated in nneigh). This is what allows the
%   algorithm to classify abnormally shaped clusters like DBSCAN.
    ND = length(rho);
    NCLUST = length(icl);
    
    clAssign = zeros(1,ND);
    labelCounts = ones(1,NCLUST);
    [~,ordrho]=sort(rho,'descend');

    for i=1:NCLUST
        clAssign(icl(i)) = labels(i);
    end

    %Cluster Center Assignation for rest of points
    for i=1:ND
        if (clAssign(ordrho(i))==0)
            clAssign(ordrho(i))=clAssign(nneigh(ordrho(i)));
            
            classLabel = clAssign(ordrho(i));
            labelCounts(classLabel) = labelCounts(classLabel)+1;
        end
    end

    % Need to reorder the labelCounts according to center order
    labelCounts_final = zeros(1,NCLUST);
    for i=1:NCLUST
        labelCounts_final(i) = labelCounts(labels(i));
    end
end