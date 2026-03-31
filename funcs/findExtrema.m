function [extremaVal,extremaIdx] = findExtrema(mode,data)
    extremaIdx = 1;
    extremaVal = data(extremaIdx);
    % Find min
    if mode == 0
        for i=2:length(data)
            if data(i) < extremaVal
                extremaIdx = i;
                extremaVal = data(extremaIdx);
            end
        end
    % Find max
    elseif mode == 1
        for i=2:length(data)
            if data(i) > extremaVal
                extremaIdx = i;
                extremaVal = data(extremaIdx);
            end
        end
    end
end