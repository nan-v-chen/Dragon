function [mag, curScore, iter, gs] = mmmhcSearchMag(covMat, indMap, discreteVars, predefined, forbidden, nVars, nSamples, tol, verbose, mmpc_final)
% Author: nan.v.chen@gmail.com
% this function is developed based on the original M3HC
% =======================================================================
% Inputs
% =======================================================================
%   covMat: covariance matrix
%   indMap: struct, originalIndex and newIndex after doing one-hot embedding to discrete data
%   discreteVars: vector<INT>, the index of discrete variables in data
%   predefined: predefined directed edges {[i, j]}, i->j, please don't add cycles
%   forbidden: forbidden directed edges {[i, j]}, i->j
%   nVars: INT, number of vars
%   nSamples: INT, sample size
%   tol: DOUBLE, tolerance
%   verbose: BOOL, debug information
%   mmpc_final: skeleton_final obtrained by MMPC
% =======================================================================
% Outputs
% =======================================================================
%   mag: mag
%   curScore: overall score
%   iter: iteration used
%   gs: records of iterations in greedy search
%   time: time taken
% =======================================================================    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% initialization
%     isAncestor = false(nVars);      % if i -> ... -> j
%     stepAncestor = zeros(nVars);    % always >= 0, number of steps i -> ... -> j: if 0, not ancestor, no path can find
    isBidir = false(nVars);         % if i <-> j
    mag = zeros(nVars);
    isParent= false(nVars);         % if i -> j
    if ~isempty(predefined)
        for i = 1:numel(predefined)
            from = predefined{i}(1);
            to = predefined{i}(2);
            mag(from, to) = 2;
            mag(to, from) = 3;
            isParent(from, to) = true;
            isParent(to, from) = false;
        end
    end
    [isAncestor, stepAncestor] = warshall2(isParent);
    isAncestor(eye(size(isAncestor)) == 1) = false;

    
    % inComponent(i) is the index of component that node i belongs to
    % all index in comps, inComponent are the original index before embedding
    [nComps, sizes, comps, inComponent] = concomp(mag);
    nsf = -nSamples / 2;
    scores = zeros(1, nComps);
    for iComp = 1:nComps
        % get the compMag (sub_mag, including all members of this component (detected accoding to bi-edges) and the parents of those members)
        % district is all members of compMag
        % all index in district are the original index before embedding
        [compMag, district] = componentMag(comps{iComp}, nVars, mag, isParent);
        % compute its score
        scores(iComp) = score_contrib(compMag, comps{iComp}, district, sizes(iComp), covMat, indMap, nSamples, tol);
    end
    nEdges = 0;
    tmpSll = nsf * sum(scores);

    % initial scores
    % curScore = -2 * tmpSll + log(nSamples) * (nVars+nEdges);
    curScore = -2 * tmpSll + bicPenalty(nSamples, nVars, nEdges);
    
    % all node pairs
    % pairs = nchoosek(1:nVars, 2);
    % pairs are obtained through mmpc_final
    pairs = [];
    for j = 1:length(mmpc_final)
        for i = 1:length(mmpc_final{j})
            if j<mmpc_final{j}(i)
                pairs = [pairs; [j, mmpc_final{j}(i)]];
            else
            end
        end
    end
    nPairs = size(pairs,1);
    
    % start greedy search
    whichPair = nan;	% current edge
    gs = struct();
    bool = true;
    iter = 0;
    while bool
        iter = iter + 1;        
        if verbose
            fprintf("Entering iteration %d, curScore: [%.4f], nComponents: %d, nEdges: %d\n", iter, curScore, nComps, nEdges);
        end
        
%         % for debug
%         if iter == 13
%             fprintf("")
%         end
        
        % initialize allowed actions 
        allowed = true(nPairs, 4);  % allowed actions: from->to; to->from; from<->to; delete
        stepScores = nan(nPairs, 4);    % score changes for each action
        scoreContribs = cell(nPairs, 4);    % scores
        
        % action on current pair
        for iPair = 1:nPairs
            if iPair == whichPair % if you just changed this edge, continue
                continue;
            end
            
            % nodes of current pair
            from = pairs(iPair, 1);
            to = pairs(iPair, 2);
            % fprintf('\n-----pair %d - %d ------\n', from , to);
            
%             % for debug
%             if from == 8 && to == 24
%                 fprintf("")
%             end
            
            % do you create a cycle?
            if stepAncestor(from, to) > 1
                % fprintf('%d--->%d in the graph\n', from, to)
                allowed(iPair, 1:3) = [true false false];
            elseif stepAncestor(to, from) > 1
                % fprintf('%d--->%d in the graph\n', to, from)
                allowed(iPair, 1:3) = [false true false];
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% edges or bi-edges to discrete nodes are not allowed
            if ismember(from, discreteVars)
                allowed(iPair, 2) = false;  % disallow add edge to -> from
                allowed(iPair, 3) = false;  % disallow add biedge from <-> to
            end
            if ismember(to, discreteVars)
                allowed(iPair, 1) = false;  % disallow add edge from -> to
                allowed(iPair, 3) = false;  % disallow add biedge from <-> to
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% deleting predefined edges is not allowed
            if any(cellfun(@(x) isequal(x, [from, to]), predefined)) || any(cellfun(@(x) isequal(x, [to, from]), predefined))
                allowed(iPair, 4) = false;
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% forbidden edges
            if any(cellfun(@(x) isequal(x, [from, to]), forbidden))
                allowed(iPair, 1) = false;
                allowed(iPair, 3) = false;
            end
            if any(cellfun(@(x) isequal(x, [to, from]), forbidden))
                allowed(iPair, 2) = false;
                allowed(iPair, 3) = false;
            end

            % check if bidirectional edges would create a cycle to->bdFrom<->bdTo->from
            [bdFrom, bdTo] = find(isBidir);
            for iBDedge = 1:length(bdFrom)
                if isAncestor(from, bdFrom(iBDedge)) && isAncestor(bdTo(iBDedge), to)
                    allowed(iPair, 1) = false;
                elseif isAncestor(to, bdFrom(iBDedge)) && isAncestor(bdTo(iBDedge), to)
                    allowed(iPair, 2) = false;
                end
            end
            
            % check if edge already exists
            if mag(from, to) == 0
                allowed(iPair, 4) = false;  % disallow removal action
            else
                if isParent(from, to)
                % fprintf('%d->%d already in the graph\n', from, to)
                    allowed(iPair, 1) = false;  % disallow adding direction from->to
                elseif isParent(to, from)
                % fprintf('%d->%d already in the graph\n', to, from)
                    allowed(iPair, 2) = false;  % disallow adding direction to->from
                elseif mag(from, to)==2 && mag(to, from)==2
                % fprintf('%d<->%d already in the graph\n', from, to)
                    allowed(iPair, 3) = false;  % disallow adding bidirectional edge from<->to
                end
            end
            
            % add edge from->to
            % fprintf(' \t 1 \t')
            if allowed(iPair, 1)
                 [stepScores(iPair, 1), scoreContribs{iPair, 1}] = addDirectedEdge(from, to, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol);
            % else fprintf('\t %d->%d not allowed\n',  from, to);
            end
            
            % add edge to->from
            % fprintf(' \t 2 \t')
            if allowed(iPair, 2)
                [stepScores(iPair, 2), scoreContribs{iPair, 2}] = addDirectedEdge(to, from, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol);
            % else fprintf('\t %d->%d not allowed\n', to, from);
            end
            
            % add edge from<->to
            % fprintf(' \t 3 \t')
            if allowed(iPair, 3)
                % update components
              [stepScores(iPair, 3), scoreContribs{iPair, 3}] = addBidirectedEdge(from, to, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol);
            end
            % else fprintf('\t %d<->%d not allowed\n', from, to);
            
            % delete edge between from and to
            % fprintf(' \t 4 \n')
            if allowed(iPair, 4)
                % if the edge is bidirected
                if mag(from,to)==2 && mag(to, from)==2
                    [stepScores(iPair, 4), scoreContribs{iPair, 4}] = removeBidirectedEdge(from, to, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol);
                else
                    if mag(from, to) == 2
                         [stepScores(iPair, 4), scoreContribs{iPair, 4}] = removeDirectedEdge(from, to, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol);
                    elseif mag(to, from) == 2
                        [stepScores(iPair, 4), scoreContribs{iPair, 4}] = removeDirectedEdge(to, from, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol);
                    end
                end
                % else fprintf('\t %d %d not allowed\n', from, to);
            end
        end % end for iPair
        
        % choose the best action, update descendants, parents
        [minScores, actions] = nanmin(stepScores, [],  2);
        [minScore, whichPair] = nanmin(minScores);
        if minScore < curScore
            from = pairs(whichPair, 1);
            to = pairs(whichPair, 2);
            % flag = true you have removed a directed edge and you have to update ancestor matrix
            flag = false;
            switch actions(whichPair)
            case 1
                if verbose
                    fprintf("\t \t Adding edge %d->%d\n", from, to);
                end
                if mag(from, to)==2 && mag(to,from)==2
                    [nComps, sizes, comps, inComponent] = updateConcompRem(from, to, nComps, sizes, comps, inComponent, mag);
                end
                if mag(from, to) == 3
                    flag = true;
                    nEdges = nEdges - 1;
                end
                mag(from, to) = 2;
                mag(to, from) = 3;
                nEdges = nEdges + 1;
                [isParent, isAncestor, stepAncestor] = updateAncestors(from, to, isParent, isAncestor, stepAncestor, flag);
        	case 2
                if verbose 
                    fprintf("\t \t Adding edge %d->%d\n", to, from);
                end
                if mag(from, to)==2 && mag(to,from)==2
                    [nComps, sizes, comps, inComponent] = updateConcompRem(to, from, nComps, sizes, comps, inComponent, mag);
                end
                if mag(to, from) == 3
                    flag = true;
                    nEdges = nEdges - 1;
                end
                mag(from, to) = 3;
                mag(to, from) = 2;
                nEdges = nEdges + 1;
                [isParent, isAncestor, stepAncestor] = updateAncestors(to, from, isParent, isAncestor, stepAncestor, flag);
            case 3
                if verbose 
                    fprintf("\t \t Adding edge %d<->%d\n", to, from);
                end
                [nComps, sizes, comps, inComponent] = updateConcomp(from, to, nComps, sizes, comps, inComponent);
                isBidir(from, to) = true;
                isBidir(to, from) = true;
                if mag(to, from) == 3
                    isParent(from, to) = false;
                    mag(to, from) = 2;
                    nEdges = nEdges - 1;
                    isAncestor = findancestors(mag);
                elseif mag(from, to) == 3
                    isParent(to, from)=false;
                    mag(from, to) = 2;
                    nEdges= nEdges - 1;
                    isAncestor = findancestors(mag);
                else
                    mag(from, to) = 2;
                    mag(to, from) = 2;
                end
                nEdges = nEdges + 1;
            case 4
                if verbose
                    fprintf("\t \t Removing edge %d*-*%d\n", to, from);
                end
                [nComps, sizes, comps, inComponent] = updateConcompRem(from, to, nComps, sizes, comps, inComponent, mag);
                 if mag(to, from) == 3
                    mag(to, from) = 0;
                    mag(from, to) = 0;
                    isParent(from,to) = false;
                    [isAncestor, stepAncestor] = warshall2(double(isParent));
                elseif mag(from, to)==3
                    isParent(to, from)=false;
                    mag(to, from)=0;mag(from, to)=0; 
                    [isAncestor, stepAncestor] = warshall2(double(isParent));
                 else
                    mag(from, to) = 0;
                    mag(to, from) = 0;  
                    isBidir(from, to) = false;
                    isBidir(to, from) = false;
                 end
                nEdges = nEdges - 1;
                isParent(from, to) = false;
                isParent(to, from) = false;
            end
            
            % update score.
            curScore = minScore;
            scores = scoreContribs{whichPair, actions(whichPair)};

            if iter == 1 || mod(iter, 100) == 0
                gs(iter).stepAncestor = stepAncestor;
                gs(iter).isAncestor = isAncestor;
                gs(iter).score=curScore;
                gs(iter).mag=mag;
            end
    
        else 
            bool = false;
            if verbose
                fprintf("Iteration %d: No score improvements, exiting greedy search\n", iter);
            end
        end
        
    end % end while
    
end


function [newScore, scoreContribs, newNComps, newSizes, newComps, newInComponent, newMag] = addDirectedEdge(from, to, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol)  
    % if the edge is bidirected you first have to update the components.
    if mag(to, from)==2 && mag(from, to)==2
       [~, scores, newNComps, newSizes, newComps, newInComponent, newMag] = removeBidirectedEdge(from, to, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol);
    else
        [newNComps, newSizes, newComps, newInComponent, newMag] = deal(nComps, sizes, comps, inComponent, mag);
    end
    
    % add one edge
    newMag(from, to) = 2;
    newMag(to, from) = 3;
    tmpIsParent = isParent;
    tmpIsParent(from, to) = true;
    
    % if the edge was directed in reverse, you must also update the score of inComponent(from);
    scoreContribs = scores;
    if mag(from, to)==3
        tmpIsParent(to, from) = false;
        component = comps{inComponent(from)};
        [compMag, district] = componentMag(component, nVars, newMag, tmpIsParent);
        scoreContribs(inComponent(from)) = score_contrib(compMag, component, district, sizes(inComponent(from)), covMat, indMap, nSamples, tol);
    end
    % update district and 
    component = comps{inComponent(to)};
    [compMag, district] = componentMag(component, nVars, newMag, tmpIsParent);
    % get new score of inComponent(to)
    scoreContribs(inComponent(to)) = score_contrib(compMag, component, district, sizes(inComponent(to)), covMat, indMap, nSamples, tol);
    
    tmpSll = (-nSamples/2) * sum(scoreContribs);
    if mag(from, to) == 0
        nEdges = nEdges + 1;
    end
    % newScore = -2 * tmpSll + log(nSamples) * (nVars+nEdges);
    newScore = -2 * tmpSll + bicPenalty(nSamples, nVars, nEdges);
end

function [newScore, scoreContribs, newNComps, newSizes, newComps, newInComponent, newMag] = addBidirectedEdge(from, to, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol)
    % if edge was directed, update isParent
    if mag(to, from) == 3 
        isParent(from, to) = false;
    elseif mag(from, to) == 3
        isParent(to, from) = false;
    end
    [newNComps, newSizes, newComps, newInComponent, k, m] = updateConcomp(from, to, nComps, sizes, comps, inComponent);
    % add edge
    newMag = mag;
    newMag(to, from) = 2;
    newMag(from, to) = 2;
    component = newComps{k};
    [compMag, district] = componentMag(component, nVars, newMag, isParent);
    % keep old scores
    keepScores = [1:m-1, m+1:nComps];
    if k < m
        newScores = scores(keepScores);
    else
        newScores = scores;
    end
    newScores(k) = score_contrib(compMag, component, district, newSizes(k), covMat, indMap, nSamples, tol);
    scoreContribs = newScores;
    
    tmpSll = (-nSamples/2) * sum(newScores);
    tmp_nEdges = nEdges + 1;
    % newScore =-2 * tmpSll + log(nSamples) * (nVars+tmp_nEdges);
    newScore = -2 * tmpSll + bicPenalty(nSamples, nVars, tmp_nEdges);
end


function [newScore, scoreContribs, nComps, sizes, comps, inComponent, newMag] = removeDirectedEdge(from, to, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol)
    newMag = mag;
    newMag(from, to) = 0;
    newMag(to, from) = 0;
    % update district
    tmpIsParent = isParent;
    tmpIsParent(from, to) = false;
    component = comps{inComponent(to)}; 
    [compMag, district] = componentMag(component, nVars, newMag, tmpIsParent);
    % get new score
    scoreContribs = scores;
    scoreContribs(inComponent(to)) = score_contrib(compMag, component, district, sizes(inComponent(to)), covMat, indMap, nSamples, tol);
    
    tmpSll = (-nSamples/2) * sum(scoreContribs);
    tmp_nEdges = nEdges + 1;
    %newScore =-2 * tmpSll + log(nSamples) * (nVars + tmp_nEdges);
     newScore = -2 * tmpSll + bicPenalty(nSamples, nVars, tmp_nEdges);
end


function [newScore, scoreContribs, newNComps, newNsizes, newComps, newInComponent, newMag] = removeBidirectedEdge(from, to, nEdges, nComps, sizes, comps, inComponent, mag, covMat, indMap, isParent, scores, nSamples, nVars, tol)
    % remove edge
    newMag = mag;
    newMag(to, from) = 0;
    newMag(from, to) = 0;
    % update components 
    [newNComps, newNsizes, newComps, newInComponent, k, m] = updateConcompRem(from, to, nComps, sizes, comps, inComponent, mag);
    % keep old scores
    if isnan(m) % if you have not split the component.
        tmp_scores = scores;
    else
        keepScores = [1:k-1, k+1:m-1, m+1:newNComps];
        tmp_scores = nan(1, newNComps);
        tmp_scores(keepScores) = scores([1:k-1,k+1:nComps]);
        % update m-score
        component= newComps{m};
        [compMag, district] = componentMag(component, nVars, newMag, isParent);
        tmp_scores(m)  = score_contrib(compMag, component, district, newNsizes(m), covMat, indMap, nSamples,  tol);
    end
    %update k-score
    component = newComps{k};
    [compMag, district] = componentMag(component, nVars, newMag, isParent);
    tmp_scores(k) = score_contrib(compMag, component, district, newNsizes(k), covMat, indMap, nSamples, tol);

    scoreContribs = tmp_scores;
    
    % calculate new score
    tmpSll = (-nSamples/2) * sum(tmp_scores);
    tmp_nEdges = nEdges - 1;
    newScore = -2 * tmpSll + bicPenalty(nSamples, nVars, tmp_nEdges);
end

function [isParent, isAncestor, stepAncestor] = updateAncestors(from, to, isParent, isAncestor, stepAncestor, flag)
    % flag=true : You have removed a directed edge and you have to recompute all ancestors
    if flag
        isParent(to, from) = false;
        isParent(from, to) = true;
        [isAncestor, stepAncestor] = warshall2(isParent);
    else
        isParent(from, to)= true;
        Anc_from = isAncestor(:, from);
        Desc_to = isAncestor(to, :);

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Anc_from = logical(Anc_from);
        Desc_to = logical(Desc_to);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        stepAncestor(Anc_from, to) = 2;
        stepAncestor(Anc_from, Desc_to) = 2;
        stepAncestor(from, Desc_to) = 2;

        Anc_from(from) = true;
        Desc_to(to) = true;
        isAncestor(Anc_from, Desc_to) = true;
    end
end

function bp = bicPenalty(nSamples, nVars, nEdges)
    bp = log(nSamples) * (2*nVars+nEdges);
end