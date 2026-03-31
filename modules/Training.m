classdef Training
    methods(Static)
        %% Visualizations (Decision Graph, Gamma)
        function showDecisionGraph(rho,delta,centers,NCLUST,rho_min,delta_min,graphTitle)
            %figure;
            labelFontSize = 10; %50 in paper figure
            axesFontSize = 10; %50 in paper figure
            titleFontSize = 10; %55 in paper figure
            thresholdFontSize = 10; %50 in paper figure
            centerSize = 14; %32 in paper figure

            rhoMax = max(rho);
            deltaMax = double(max(delta));

            plot(rho(:),double(delta(:)),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
            title(graphTitle,'FontSize',titleFontSize)
            xlim([0 rhoMax+0.1*rhoMax])
            ylim([0 deltaMax+0.1*deltaMax])
            xlabel ('\rho','FontSize',labelFontSize);
            ylabel ('\delta','FontSize',labelFontSize);
            ax = gca(gcf);
            ax.XAxis.FontSize = axesFontSize;
            ax.YAxis.FontSize = axesFontSize;

            %Mark centers
            colors = [[0 .75 .75];[0 1 0];[.75 .75 0]];
            for i=1:NCLUST
               %ic=int8((i*64.)/(NCLUST*1.));
               hold on
               plot(rho(centers(i)),double(delta(centers(i))),'o','MarkerSize',centerSize,'MarkerFaceColor',colors(mod(i,3)+1,:),'MarkerEdgeColor',colors(mod(i,3)+1,:));
               hold off
            end

            %Mark rho and delta thresholds
            xline(double(rho_min*rhoMax),'-','\rho min',"FontSize",thresholdFontSize);
            yline(double(delta_min*deltaMax),"-",'\delta min',"FontSize",thresholdFontSize);
        end %end showDecisionGraph
       
        function showGammaGraph(rho,delta,graphTitle)
            gamma = delta.*rho;
            plot(gamma,'o');
            title(graphTitle);
        end

        %% Thresholding attempts for finding centers
        function [delta_thresh,rho_thresh] = trainThresholds(rhoGridSize,deltaGridSize,trainingPortion,spikeVecs,feMethod,numComponents,distanceMethod,percentCutOff)
            %Attempt to find hor and vert line cutoffs for rho and delta by
            %sweeping a "grid" of combinations and finding the mode...

            %It seems like this would require a lot of iterations to nail
            %down, which would make it pretty inefficient...
            rho_min = 0.1;
            rho_max = 0.7;
            delta_min = 0.1;
            delta_max = 0.7;
            numSpikes = size(spikeVecs,1);

            rho_vals = linspace(rho_min,rho_max,rhoGridSize);
            delta_vals = linspace(delta_min,delta_max,deltaGridSize);
            [allFeats,~] = FeatureExtract.getFeatureVecs(feMethod,spikeVecs,numComponents,1:numSpikes,[]);
            nClust = zeros(rhoGridSize,deltaGridSize);

            dgFig = figure('Name',strcat("Decision_Graphs_Training"));

            for i=1:length(rho_vals)
                for j=1:length(delta_vals)
                    
                    [trainingIdx,~] = FeatureExtract.splitData(trainingPortion,numSpikes);
                    trainingFeatures = zeros(length(trainingIdx),numComponents);
                    for k=1:length(trainingIdx)
                        trainingFeatures(k,:) = allFeats(trainingIdx(k),:);
                    end
                    
                    [dMat,idx_k,pairs,~] = ComputeDistance.computeAllPairs(distanceMethod,trainingFeatures,percentCutOff);
                    [~,centers,rho,delta] = Training.fsa_deltaOpt(dMat,idx_k,pairs,rho_vals(i),delta_vals(j),false,false);
                    nClust(i,j) = length(centers);

                    set(0,'CurrentFigure',dgFig);
                    graphIndex = (i-1)*length(delta_vals)+(j-1)+1;
                    subplot(length(rho_vals),length(delta_vals),graphIndex);
                    Training.showDecisionGraph(rho,delta,centers,length(centers),rho_vals(i),delta_vals(j),"Rho/Delta" + rho_vals(i) + "/" + delta_vals(j));

                end
            end
            %numClusters = mode(nClust,'all');
            numClusters = mode(mode(nClust));
            for i=1:length(rho_vals)
                for j=1:length(delta_vals)
                    if nClust(i,j) == numClusters
                        delta_thresh = delta_vals(j);
                        rho_thresh = rho_vals(i);
                    end
                end
            end %for i
        end %end function

        function [centers,NCLUST] = findCenters_gamma(delta,rho,N)
            %Attempting to find off where the sorted gamma function has an
            %elbow -- the cluster gammas should be much higher than the
            %rest of the point (also very highly dataset variable...)
            gamma = delta.*rho;
            [gamma_sorted,ord_gamma] = sort(gamma,'descend');
            gamma_sorted = gamma_sorted(1:N);
            x=1:N;
            xn = (x-1) / (N-1);
            yn = (gamma_sorted - min(gamma_sorted)) / (max(gamma_sorted)-min(gamma_sorted));

            d = (1-xn)-yn;
            [~,elbow_idx] = max(d);
            centers = ord_gamma(1:elbow_idx);
            NCLUST = length(centers); 
        end

        function [centers,NCLUST,conf,delta_thresh,icl] = nonlinear_thresh(trainingFeatures,epsilon,UB_p,dij,d_LB,rho_cnt_bins,rho_idx_bins,rho,delta,verbose)
            % Useful Constants:
            maxClusters = 10;
            numComponents = size(trainingFeatures,2);
            numBins = length(rho_cnt_bins);

            % Outputs:
            NCLUST = 0;
            centers = zeros(maxClusters,numComponents); %10 is just maximum length
            conf = zeros(1,maxClusters);
            icl = zeros(1,maxClusters);
            delta_thresh = zeros(numBins,1);

            % Find clusters via adaptive thresholding
            d_UB = prctile(dij,UB_p);
            
            max_delta = max(delta);
            margin = 0.1*max_delta; % initialize (arbitrary)
            rollavg_thresh = 0.25*max_delta; % initialize (arbitrary)
        
            % Delta "Big Jump" Definition
            jump_thresh = epsilon*(d_UB-d_LB);
            og_jump_thresh = jump_thresh;
            if verbose
                disp("Jump_threshold")
                disp(jump_thresh);
            end
    
            minBin = 1;
            for idx1=minBin:numBins
                %This was an attempt to make cluster finding more
                %restrictive for lower rho-value bins
                % if idx1 == 1 || idx1 == 2
                %     jump_thresh = og_jump_thresh*2;
                % else
                %     jump_thresh = og_jump_thresh;
                % end

                %Obtain delta values for current rho bin
                delta_list = delta( rho_idx_bins(idx1,1:rho_cnt_bins(idx1)) );
                binSize = rho_cnt_bins(idx1);
                
                %Determine threshold for current bin
                if binSize > 2
                    sorted_delta_list = sort(delta_list,"ascend");
                    cursor = 2;
                    while((cursor-1 < binSize) && (sorted_delta_list(cursor-1)+jump_thresh) > sorted_delta_list(cursor))
                        cursor = cursor+1;
                    end
                    delta_thresh(idx1) = sorted_delta_list(cursor-1)+margin;
                    %disp(cursor)
                elseif binSize == 2
                    sorted_delta_list = sort(delta_list,'ascend');
                    if sorted_delta_list(2) > (jump_thresh+sorted_delta_list(1))
                        delta_thresh(idx1) = sorted_delta_list(1)+margin;
                        cursor = 2;
                    else
                        delta_thresh(idx1) = sorted_delta_list(2)+margin;
                        cursor = 3;
                    end

                    if (rollavg_thresh+jump_thresh < delta_thresh(idx1))
                        delta_thresh(idx1) = rollavg_thresh;
                        cursor = 1;
                    end
                elseif binSize == 1
                    cursor = 2;
                    delta_thresh(idx1) = delta_list(1)+margin;
                    if (rollavg_thresh+jump_thresh < delta_thresh(idx1))
                        delta_thresh(idx1) = rollavg_thresh;
                        cursor = 1;
                    end
                elseif binSize == 0 %I should just remove empty bins altogether...
                    delta_thresh(idx1) = rollavg_thresh;
                end
            
                %Save centers that are found
                % %This piece of code was being so temperamental..
                % if (cursor-1 < binSize) && (binSize > 0)
                %     for idx2=cursor:binSize
                %         NCLUST = NCLUST + 1;
                %         centers(NCLUST,:) = trainingFeatures(rho_idx_bins(idx1,idx2),:);
                %     end
                % end
                for idx2=1:binSize
                    if delta_list(idx2) > delta_thresh(idx1)
                        NCLUST = NCLUST + 1;
                        centers(NCLUST,:) = trainingFeatures(rho_idx_bins(idx1,idx2),:);
                        conf(NCLUST) = delta_list(idx2) - delta_thresh(idx1);
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
                text(0,max_delta/2,"Centers: "+NCLUST,'Color','b','FontSize',10.0)
                text(0,max_delta/2-max_delta*0.1,"Thresh: "+jump_thresh,'Color','b','FontSize',10.0)
            end

        end %end func

        %% Auxiliary Function
        function [rho,delta,rho_idx_bins,rho_cnt_bins,nneigh] = computeRhoDelta(dist,idx_k,pairs,graph)
            maxClusters = 10; %number of spike classes hopefully never exceeds this
            ND = size(dist,1);
            
            %Compute rho for each datapoint
            rho=zeros(1,ND);
            for idx=1:length(idx_k)
                pairIdx = idx_k(idx);
                i = pairs(pairIdx,1);
                j = pairs(pairIdx,2);
                rho(i) = rho(i)+1;
                rho(j) = rho(j)+1;
            end
            rhoNormFactor = max(rho);
            rho_idx_bins = zeros(rhoNormFactor,100);
            rho_cnt_bins = zeros(1,rhoNormFactor);

            %MAIN MODIFICATION: sort delta values into rho bins with counts
            for idx=1:length(rho)
                if rho(idx) > 0
                    binNumber = rho(idx);
                    rho_cnt_bins(binNumber) = rho_cnt_bins(binNumber)+1;
                    rho_idx_bins(binNumber,rho_cnt_bins(binNumber)) = idx;
                end
            end
            rho = rho/rhoNormFactor;
            
            %Sort density (rho) values in descending order
            delta = zeros(1,ND);
            nneigh = zeros(1,ND);
            [~,ordrho]=sort(rho,'descend');
            delta(ordrho(1))=-1.;
            nneigh(ordrho(1))=0;
            
            %Compute delta values: distance of the closest higher density point
                %nneigh: of higher density points, the nearest one
            maxd=max(max(dist));
            for ii=2:ND
                delta(ordrho(ii))=maxd;
                for jj=1:ii-1
                    if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
                        nneigh(ordrho(ii))=ordrho(jj);
                    end
                end
            end
            delta(ordrho(1))=max(delta(:));

            if graph
                plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
                rhoMax = max(rho);
                deltaMax = double(max(delta));
                xlim([0 rhoMax+0.1*rhoMax])
                ylim([0 deltaMax+0.1*deltaMax])
                xlabel ('\rho');
                ylabel ('\delta');
            end
        end

        %% Various versions of the fsa algorithm
        function [clAssign,icl,rho,delta,nneigh] = fsa(dist,idx_k,pairs,rho_min,delta_min,graph,verbose)
            %Rho computation is reduced to O(k) complexity instead of a
            %O(n^2) complexity
            maxClusters = 10; %number of spike classes hopefully never exceeds this
            ND = size(dist,1);
            
            %Compute rho for each datapoint
            rho=zeros(1,ND);
            for idx=1:length(idx_k)
                pairIdx = idx_k(idx);
                i = pairs(pairIdx,1);
                j = pairs(pairIdx,2);
                rho(i) = rho(i)+1;
                rho(j) = rho(j)+1;
            end
            rhoNormFactor = max(rho);
            rho = rho/rhoNormFactor;
            
            %Sort density (rho) values in descending order
            delta = zeros(1,ND);
            nneigh = zeros(1,ND);
            [~,ordrho]=sort(rho,'descend');
            delta(ordrho(1))=-1.;
            nneigh(ordrho(1))=0;
            
            %Compute delta values: distance of the closest higher density point
                %nneigh: of higher density points, the nearest one
            maxd=max(max(dist));
            for ii=2:ND
                delta(ordrho(ii))=maxd;
                for jj=1:ii-1
                    if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
                        nneigh(ordrho(ii))=ordrho(jj);
                    end
                end
            end
            delta(ordrho(1))=max(delta(:));
            
            %Draw "rectangle" to capture centers in decision graph
            rhomin = max(rho)*rho_min; 
            deltamin = max(delta)*delta_min;
        
            %Get cluster centers
            clAssign = zeros(1,ND);
            icl = zeros(1,maxClusters);
            NCLUST=0;
            for i=1:ND
                if ((rho(i)>rhomin) && (delta(i)>deltamin))
                    NCLUST=NCLUST+1;
                    clAssign(i)=NCLUST;
                    icl(NCLUST)=i;
                end
            end
                    
            %Cluster Center Assignation for rest of points
            for i=1:ND
                if (clAssign(ordrho(i))==0)
                    clAssign(ordrho(i))=clAssign(nneigh(ordrho(i)));
                end
            end
            icl = icl(1:NCLUST);

            %Additional output for debugging if specified by user
            %Print output if specified
            if verbose
                fprintf('NUMBER OF CLUSTERS: %i \n', int8(NCLUST));
                eltCount = zeros(1,maxClusters);
                for i=1:ND
                    eltCount(clAssign(i)) = eltCount(clAssign(i))+1;
                end
                for i=1:NCLUST
                   fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i\n', int8(i),int16(icl(i)),int16(eltCount(i)));
                end
            end

            if graph
                Training.showDecisionGraph(rho,delta,icl,NCLUST,rho_min,delta_min,'FSA Decision Graph');
            end
        end %end fsa

        function [clAssign,icl,rho,delta] = fsa_deltaOpt(dist,idx_k,pairs,rho_min,delta_min,graph,verbose)
            maxClusters = 10; %number of spike classes hopefully never exceeds this
            ND = size(dist,1);
            
            %Compute rho for each datapoint
            rho=zeros(1,ND);
            for idx=1:length(idx_k)
                pairIdx = idx_k(idx);
                i = pairs(pairIdx,1);
                j = pairs(pairIdx,2);
                rho(i) = rho(i)+1;
                rho(j) = rho(j)+1;
            end
            rhoNormFactor = max(rho);
            rho = rho/rhoNormFactor;
            
            %Sort density (rho) values in descending order
            delta = zeros(1,ND);
            nneigh = zeros(1,ND);
            [~,ordrho]=sort(rho,'descend');
            delta(ordrho(1))=-1.;
            nneigh(ordrho(1))=0;
            
            %Compute delta values: distance of the closest higher density point
                %nneigh: of higher density points, the nearest one
            maxd=max(max(dist));
            numSkipped = 0;
            for ii=2:ND
                if rho(ordrho(ii)) > rho_min
                    delta(ordrho(ii))=maxd;
                    for jj=1:ii-1
                        if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                            delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
                            nneigh(ordrho(ii))=ordrho(jj);
                        end
                    end
                else
                    numSkipped = numSkipped+1;
                end
            end
            delta(ordrho(1))=max(delta(:));
            
            %Draw "rectangle" to capture centers in decision graph
            rhomin = max(rho)*rho_min; 
            deltamin = max(delta)*delta_min;
        
            %Get cluster centers
            clAssign = zeros(1,ND);
            icl = zeros(1,maxClusters);
            NCLUST=0;
            for i=1:ND
                if ((rho(i)>rhomin) && (delta(i)>deltamin))
                    NCLUST=NCLUST+1;
                    clAssign(i)=NCLUST;
                    icl(NCLUST)=i;
                end
            end
            %[icl,NCLUST] = Training.findCenters_gamma(delta,rho,ND-numSkipped);
                    
            % %Cluster Center Assignation for rest of points
            % for i=1:ND
            %     if (clAssign(ordrho(i))==0)
            %         clAssign(ordrho(i))=clAssign(nneigh(ordrho(i)));
            %     end
            % end
            icl = icl(1:NCLUST);

            %Additional output for debugging if specified by user
            %Print output if specified
            if verbose
                fprintf('NUMBER OF CLUSTERS: %i \n', int8(NCLUST));
                eltCount = zeros(1,maxClusters);
                for i=1:ND
                    eltCount(clAssign(i)) = eltCount(clAssign(i))+1;
                end
                for i=1:NCLUST
                   fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i\n', int8(i),int16(icl(i)),int16(eltCount(i)));
                end
            end

            if graph
                Training.showDecisionGraph(rho,delta,icl,NCLUST,rho_min,delta_min,'FSA_deltaopt');
            end
        end %end fsa_deltaopt

        function [clAssign,icl,rho,delta] = fsa_original(dist,pairs,dij,cutoffPercent,rho_min,delta_min,graph,verbose)
            maxClusters = 10; %number of spike classes hopefully never exceeds this
        
            ND = max(pairs(:,1));
            NL = max(pairs(:,2));
            if(NL>ND)
                ND=NL;
            end
            N = size(pairs,1);
            
            %Populate dist Matrix (NDxND)
            % dist = zeros(ND);
            % for i=1:N
            %     dist(pairs(i,1),pairs(i,2))=dij(i);
            %     dist(pairs(i,2),pairs(i,1))=dij(i);
            % end
            
            %Set cut-off
            position=round(N*cutoffPercent/100);
            sda=sort(dij);
            dc=sda(position);
            
            %Compute rho for each datapoint
            rho=zeros(1,ND);
            for i=1:ND-1
                for j=i+1:ND
                    if (dist(i,j)<dc)
                      rho(i)=rho(i)+1.;
                      rho(j)=rho(j)+1.;
                    end
                end % for j
            end % for i
            
            %Sort density (rho) values in descending order
            delta = zeros(1,ND);
            nneigh = zeros(1,ND);
            [~,ordrho]=sort(rho,'descend');
            delta(ordrho(1))=-1.;
            nneigh(ordrho(1))=0;
            
            %Compute delta values: distance of the closest higher density point
                %nneigh: of higher density points, the nearest one
            maxd=max(max(dist));
            for ii=2:ND
                delta(ordrho(ii))=maxd;
                for jj=1:ii-1
                    if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
                        nneigh(ordrho(ii))=ordrho(jj);
                    end
                end
            end
            delta(ordrho(1))=max(delta(:));
            
            %Draw "rectangle" to capture centers in decision graph
            rhomin = max(rho)*rho_min;
            deltamin = max(delta)*delta_min;
        
            %Get cluster centers and classify them
            clAssign = zeros(1,ND);
            icl = zeros(1,maxClusters);
            NCLUST=0;
            for i=1:ND
                if ((rho(i)>rhomin) && (delta(i)>deltamin))
                    NCLUST=NCLUST+1;
                    clAssign(i)=NCLUST;
                    icl(NCLUST)=i;
                end
            end

            %Assignation
            % for i=1:ND
            %     if (clAssign(ordrho(i))==0)
            %         clAssign(ordrho(i))=clAssign(nneigh(ordrho(i)));
            %     end
            % end
            icl = icl(1:NCLUST);

            if verbose
                fprintf('NUMBER OF CLUSTERS: %i \n', int8(NCLUST));
                %Bookkeeping
                eltCount = zeros(1,maxClusters);
                for i=1:ND
                    eltCount(clAssign(i)) = eltCount(clAssign(i))+1;
                end
                for i=1:NCLUST
                    fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i\n', int8(i),int16(icl(i)),int16(eltCount(i)));
                end
            end
            
            if graph
                Training.showDecisionGraph(rho,delta,icl,NCLUST,rho_min,delta_min,'FSA_og');
            end

        end %fsa_original

        %% FIXED-POINT VERSION OF FSA
        function [clAssign,icl,rho,delta] = fsa_fi(dist,idx_k,pairs,rho_min,delta_min,graph,verbose,wordLength,fractionLength,F)
            maxClusters = 10; %number of spike classes hopefully never exceeds this
            ND = size(dist,1);
            
            %Compute rho for each datapoint
            rho=zeros(1,ND);
            for idx=1:length(idx_k)
                pairIdx = idx_k(idx);
                i = pairs(pairIdx,1);
                j = pairs(pairIdx,2);
                rho(i) = rho(i)+1;
                rho(j) = rho(j)+1;
            end
            rhoNormFactor = max(rho);
            rho = rho/rhoNormFactor;
            rho = fi(rho,0,wordLength,fractionLength,F);
            
            %Sort density (rho) values in descending order
            delta = fi(zeros(1,ND),0,wordLength,fractionLength,F);
            nneigh = zeros(1,ND);
            [~,ordrho]=sort(rho,'descend');
            delta(ordrho(1))=0;
            nneigh(ordrho(1))=0;
            
            %Compute delta values: distance of the closest higher density point
                %nneigh: of higher density points, the nearest one
            maxd=max(max(dist));
            for ii=2:ND
                delta(ordrho(ii))=maxd;
                for jj=1:ii-1
                    if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
                        nneigh(ordrho(ii))=ordrho(jj);
                    end
                end
            end
            delta(ordrho(1))=max(delta(:));
            
            %Draw "rectangle" to capture centers in decision graph
            rhomin = max(rho)*rho_min; 
            deltamin = max(delta)*delta_min;
        
            %Get cluster centers
            clAssign = zeros(1,ND);
            icl = zeros(1,maxClusters);
            NCLUST=0;
            for i=1:ND
                if ((rho(i)>rhomin) && (delta(i)>deltamin))
                    NCLUST=NCLUST+1;
                    clAssign(i)=NCLUST;
                    icl(NCLUST)=i;
                end
            end
                    
            %Cluster Center Assignation for rest of points
            for i=1:ND
                if (clAssign(ordrho(i))==0)
                    clAssign(ordrho(i))=clAssign(nneigh(ordrho(i)));
                end
            end
            icl = icl(1:NCLUST);

            %% Additional output for debugging if specified by user
            %Print output if specified
            if verbose
                fprintf('NUMBER OF CLUSTERS: %i \n', int8(NCLUST));
                eltCount = zeros(1,maxClusters);
                for i=1:ND
                    eltCount(clAssign(i)) = eltCount(clAssign(i))+1;
                end
               for i=1:NCLUST
                   fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i\n', int8(i),int16(icl(i)),int16(eltCount(i)));
               end
            end

            if graph
                Training.showDecisionGraph(rho,delta,icl,NCLUST,rho_min,delta_min,'FSA_fi');
            end
        end %end fsa_fi

    end %methods
end %classdef