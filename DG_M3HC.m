function [mag, curScore, iter, gs, skeleton_final] = DG_M3HC(data, discreteVars, predefined, forbidden, maxCondSet, threshold, tol, verbose, cor, skeleton)
% Author: nan.v.chen@gmail.com
% this function is developed based on the original M3HC
% =======================================================================
% Inputs
% =======================================================================
%   data: the original data with discrete variables
%   discreteVars: vector<INT>, the index of discrete variables in data
%   predefined: predefined directed edges {[i, j]}, i->j, please don't add cycles
%   forbidden: forbidden directed edges {[i, j]}, i->j
%   maxCondSet: INT, maximum conditioning set size for MMPC
%   threshold: DOUBLE, significance level alpha for MMPC
%   tol: DOUBLE, tolerance
%   verbose: BOOL, debug information
%   cor: BOOL, true for correlation matrix / false for covariance matrix
%   skeleton: String, method to compute the skeleton before greedy search
% =======================================================================
% Outputs
% =======================================================================
%   mag: mag
%   curScore: overall score
%   iter: iteration used
%   gs: records of iterations in greedy search
%   skeleton_final: skeleton_final obtrained by MMPC
% =======================================================================
    if nargin < 10 || isempty(skeleton)
        skeleton = 'MMPC';
    end
    if nargin < 9 || isempty(cor)
        cor = false;
    end
    if nargin < 8 || isempty(verbose)
        verbose = false;
    end
    if nargin < 7 || isempty(tol)
        tol = 1e-6;
    end
    if nargin < 6 || isempty(threshold)
        threshold = 0.05;
    end
    if nargin < 5 || isempty(maxCondSet)
        maxCondSet = 10;
    end
    if nargin < 4 || isempty(forbidden)
        forbidden = {};
    end
    if nargin < 3 || isempty(predefined)
        predefined = {};
    end
    if nargin < 2 || isempty(discreteVars)
        discreteVars = [];
    end
    if nargin < 1 || isempty(data)
        error('DG_M3HC:MissingInput', ...
            'The first input argument "data" is required.');
    end

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

    [mag, curScore, iter, gs] = mmmhcSearchMag(newCovMat, indMap, discreteVars, predefined, forbidden, nVars, nSamples, tol, verbose, skeleton_final);
end