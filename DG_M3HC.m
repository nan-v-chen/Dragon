function [mmmhcMag, mmmhc_bs, mmmhcIters, mmmhc, skeleton_final] = DG_M3HC(data, discreteVars, predefined, forbidden, maxCondSet, threshold, tol, verbose, cor, skeleton)
% Author: nan.v.chen@gmail.com
% this function is developed based on the original M3HC
% =======================================================================
% Inputs
% =======================================================================
%   data: the original data with discrete variables
%   discreteVars: vector<INT>, the index of discrete variables in data
%   predefined: predefined directed edges {[i, j, dir]}, i < j, dir=2: i->j; dir=3: j->i; please don't add cycles
%   forbidden: forbidden directed edges {[i, j, dir]}, i < j, dir=2: i->j; dir=3: j->i
%   maxCondSet: INT, maximum conditioning set size for MMPC
%   threshold: DOUBLE, significance level alpha for MMPC
%   tol: DOUBLE, tolerance
%   cor: BOOL, true for correlation matrix / false for covariance matrix
%   skeleton: String, method to compute the skeleton before greedy search
% =======================================================================
% Outputs
% =======================================================================
%   mmmhcMag:
%   mmmhc_bs:
%   mmmhcIters:
%   mmmhc:
%   skeleton_final:
%   time:
% =======================================================================

    nSamples = size(data, 1);
    nVars = size(data, 2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% data pre-processing: convert data to newData; store originalIndex and newIndex in indMap
    newData = [];
    indMap = struct();
    for i = 1:nVars
        % if it is discrete, do one-hot embedding, then remove the last dimention
        if ismember(i, discreteVars)
            onehotEncoded = dummyvar(categorical(data(:, i)));
            onehotEncoded = onehotEncoded(:, 1:end-1);
            startIdx = size(newData, 2) + 1;
            endIdx = startIdx + size(onehotEncoded, 2) - 1;
            indMap(i).originalIndex = i;
            indMap(i).newIndex = startIdx:endIdx;
            newData = [newData, onehotEncoded];
        else
            indMap(i).originalIndex = i;
            indMap(i).newIndex = size(newData, 2) + 1;
            newData = [newData, data(:, i)];
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% original covMat and new covMat
    if cor == logical(true)
        covMat = corr(data);
        newCovMat = corr(newData);
    else
        covMat = cov(data);
        newCovMat = cov(newData);
    end

    if strcmp(skeleton, "MMPC")
        testParams = struct('N', nSamples);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% just run MMPC for continuous vars,then add edges for every discrete val to others (data as the last param is just used for debug)
        isDiscrete = false(1, nVars);
        isDiscrete(discreteVars) = true;
        % get graph only for continuous vars
        graph3 = MMPC_skeleton(covMat(~isDiscrete, ~isDiscrete), maxCondSet, threshold, @FisherTestFast, testParams, data);
        % update the graph
        nonDiscreteVars = setdiff(1:nVars, discreteVars);
        newGraph3 = ones(nVars) - eye(nVars);
        newGraph3(nonDiscreteVars, nonDiscreteVars) = graph3;
        for i = 1:nVars
            skeleton_final{1,i} = find(newGraph3(i,:)==1);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% run MMPC for all val
%         graph3 = MMPC_skeleton(covMat, maxCondSet, threshold, @FisherTestFast, testParams, data);
%         for i = 1:nVars
%             skeleton_final{1,i} = find(graph3(i,:)==1);
%         end
    end

    [mmmhcMag, mmmhc_bs, mmmhcIters, mmmhc] = mmmhcSearchMag(newCovMat, indMap, discreteVars, predefined, forbidden, nVars, nSamples, tol, verbose, skeleton_final);
end