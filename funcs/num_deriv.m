function [dx_data] = num_deriv(order,samplingInterval,spikeVecs)
%Compute numerical derivative to first or second order on data.
    
    if order == 1 %Calculate first derivative
        dx_data = zeros([size(spikeVecs,1),size(spikeVecs,2)]);
        for spikeIdx=1:size(spikeVecs,1)
            for pointIdx=1:size(spikeVecs,2)

                if pointIdx==1
                    dx_data(spikeIdx,pointIdx) = (spikeVecs(spikeIdx,pointIdx+1)-spikeVecs(spikeIdx,pointIdx))/samplingInterval; %fwd approx
                elseif pointIdx==size(spikeVecs,2)
                    dx_data(spikeIdx,pointIdx) = (spikeVecs(spikeIdx,pointIdx)-spikeVecs(spikeIdx,pointIdx-1))/samplingInterval; %bwd approx
                else
                    dx_data(spikeIdx,pointIdx) = (spikeVecs(spikeIdx,pointIdx+1)-spikeVecs(spikeIdx,pointIdx-1))/(2*samplingInterval); %centered approx
                end

            end
        end
    elseif order == 2 %Calculate second derivative
        dx_data = zeros([size(spikeVecs,1),size(spikeVecs,2)]);
        for spikeIdx=1:size(spikeVecs,1)
            
            for pointIdx=1:size(spikeVecs,2)

                if pointIdx==1
                    dx_data(spikeIdx,pointIdx) = (spikeVecs(spikeIdx,pointIdx+2)-2*spikeVecs(spikeIdx,pointIdx+1)+spikeVecs(spikeIdx,pointIdx))/(samplingInterval^2); %fwd approx
                elseif pointIdx==size(spikeVecs,2)
                    dx_data(spikeIdx,pointIdx) = (spikeVecs(spikeIdx,pointIdx)-2*spikeVecs(spikeIdx,pointIdx-1)+spikeVecs(spikeIdx,pointIdx-2))/(samplingInterval^2); %bwd approx
                else                                
                    dx_data(spikeIdx,pointIdx) = (spikeVecs(spikeIdx,pointIdx+1)-2*spikeVecs(spikeIdx,pointIdx)+spikeVecs(spikeIdx,pointIdx-1))/(samplingInterval^2); %centered approx
                end

            end
        end  
    end

end