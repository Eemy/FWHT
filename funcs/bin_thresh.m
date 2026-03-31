function [centers,NCLUST,delta_thresh,icl,J] = bin_thresh(trainingFeatures,epsilon,normScale,rho_cnt_bins,rho_idx_bins,rho,delta,verbose)
    % Useful Constants:
    maxClusters = 10;
    numComponents = size(trainingFeatures,2);
    numBins = length(rho_cnt_bins);

    % Outputs:
    NCLUST = 0;
    centers = zeros(maxClusters,numComponents); %10 is just maximum length
    icl = zeros(1,maxClusters);
    delta_thresh = zeros(numBins,1);
    J = zeros(numBins,1);

    % Set parameters for adaptive bin thresholding    
    jump_thresh = epsilon*normScale;
    max_delta = max(delta);
    margin = jump_thresh/5;
    rollavg_thresh = 0.25*max_delta; % initialize (arbitrary)
    if verbose
        disp("Jump_threshold")
        disp(jump_thresh);
    end

    for idx1=1:numBins
        %Obtain delta values for current rho bin
        delta_list = delta( rho_idx_bins(idx1,1:rho_cnt_bins(idx1)) );
        binSize = rho_cnt_bins(idx1);
        
        %Determine threshold for current bin
        if binSize > 0
            sorted_delta_list = sort(delta_list,"ascend");
            
            if binSize > 1
                J(idx1) = (sorted_delta_list(binSize)-sorted_delta_list(binSize-1))/normScale;
            end

            delta_thresh(idx1) = sorted_delta_list(1)+margin;
            cursor = 1;
            while((cursor+1 <= binSize) && (sorted_delta_list(cursor+1)-sorted_delta_list(cursor) < jump_thresh))
                cursor = cursor+1;
                delta_thresh(idx1) = sorted_delta_list(cursor)+margin;
            end

            if((idx1 ~= 1) && (delta_thresh(idx1)-rollavg_thresh > jump_thresh))
                delta_thresh(idx1) = rollavg_thresh;
                cursor = 1;
            end
        else
            continue; %next bin
        end
    
        %Save centers that are found
        % for idx2=cursor+1:binSize
        %     NCLUST = NCLUST + 1;
        %     centers(NCLUST,:) = trainingFeatures(rho_idx_bins(idx1,idx2),:);
        %     icl(NCLUST) = rho_idx_bins(idx1,idx2);
        % end
        for idx2=1:binSize
            if delta_list(idx2) > delta_thresh(idx1)
                NCLUST = NCLUST + 1;
                centers(NCLUST,:) = trainingFeatures(rho_idx_bins(idx1,idx2),:);
                %conf(NCLUST) = delta_list(idx2) - delta_thresh(idx1);
                icl(NCLUST) = rho_idx_bins(idx1,idx2);
            end
        end

        %Update rolling average of threshold
        if idx1 == 1
            rollavg_thresh = delta_thresh(idx1);
        else
            rollavg_thresh = (rollavg_thresh*(idx1-1)+delta_thresh(idx1))/idx1;
        end
    end %bin number

    if verbose
        %figure;
        rho_bins = linspace(0,1,length(rho_cnt_bins)+1);
        rho_bins = rho_bins(2:end);

        plot(rho,delta,'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
        hold on
        plot(rho_bins,delta_thresh,'o',"MarkerFaceColor",'r');
        hold off

        % Write the number of centers found and jump_thresh used
        text(0,max_delta/2,"Centers: "+NCLUST,'Color','b','FontSize',15.0)
        text(0,max_delta/2-max_delta*0.1,"Jump: "+jump_thresh,'Color','b','FontSize',15.0)
    end

end %end func