function [fScore,incorrectIdx] = evalFScore_v3(spike_class,assignedClass,setIdx)
% The main different v3 provides from v1 and v2 is that the labeling phase
% is separated into another function (labelCenters). 
% Class assignation should also be performed elsewhere as well
% (can be density-based or distance-based). 

    % Evaluate accuracy
    fp = 0; % false positve (misclassified)
    tp = 0; % true positive (correct classification)

    incorrectIdx = zeros(length(setIdx),1);
    for i=1:length(setIdx)
        answer = spike_class(setIdx(i));
        if assignedClass(i) ~= answer
            fp = fp+1;
            incorrectIdx(fp) = i;
        else
            tp = tp+1;
        end
    end

    recall = tp/(tp+fp);
    fScore = 2*recall/(1+recall); 
    incorrectIdx = incorrectIdx(1:fp);

    % fScore = 2*(precision*recall)/(precision+recall) -- harmonic mean
    % if precision is 1, then this becomes 2*recall/(1+recall),

end