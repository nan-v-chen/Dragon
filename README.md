# Dragon
Degenerate Gaussian (DG) score-based causal discovery

## Usage
### 1. Synthetic data
```matlab
clc; clear;
addpath("M3HC");
addpath("utilities");
addpath("FCI");

% parameters
nnVars = [20];
maxParents = 3;
nnLatent = 0.1;
nnDiscrete = 0.1;
categoryProbs = [0.51, 0.42, 0.07];
nnSamples = [100, 500, 1000, 10000];
nIters = 100;
tol = 10^-6;
methods = ["FCI", "cFCI", "GSMAG", "M3HC"];

% parameters for M3HC
% COR = true for correlation matrix / false for covariance matrix
COR = false;
maxCondSetM3HC = 10;
skeleton = "MMPC";

% for different number of variables
for inVars = 1:length(nnVars)
    nVars = nnVars(inVars);
    nLatent = ceil(nnLatent * nVars);
    nDiscrete = ceil(nnDiscrete * nVars);
    fprintf('----------------------nVars: %d----------------\n', nVars);

    % for different sample size
    for inSamples = 1:length(nnSamples)
        nSamples = nnSamples(inSamples);
        if nSamples <= 1000
            threshold = 5e-2;
        elseif nSamples <= 10000
            threshold = 5e-4;
        elseif nSamples <= 100000
            threshold = 5e-6;
        end
        fprintf('----------------------nSamples: %d----------------\n', nSamples);
        
        % different iterations
        for iter = 1:nIters
            fprintf("Iter %d:\n", iter);
            
            rng(iter);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% choose latent and discrete variables
            selectedVals = randsample(1:nVars, nLatent+nDiscrete);  % all selected variables: random, disordered, can be separated then
            latentVals = selectedVals(1:nLatent);
            discreteVars = selectedVals(nLatent+1:end);
            isLatent = false(1, nVars);
            isLatent(latentVals) = true;
            
            % generate random DAG without edge to discrete vals
            dag = randomdag(nVars, maxParents);
            dag(:, discreteVars) = 0;
            % generate mag and pag with dag
            magL = dag2mag(dag, isLatent);
            magT = magL(~isLatent, ~isLatent);
            pagT = mag2pag(magT);
            % simulate data with gaussian distribution from dag
            bn = dag2randBN(dag, 'gaussian');
            ds = simulatedata(bn, nSamples, 'gaussian', 'isLatent', isLatent);
            data = ds.data(:, ~isLatent);   % delete latent variables
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% get new indexes for discrete variables after deleting latent variables
            newIndMap = find(~isLatent);
            [~, newDiscreteVars] = ismember(discreteVars, newIndMap);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% discrete data
            for i = 1:length(newDiscreteVars)
                varIdx = newDiscreteVars(i);
                probs = categoryProbs;
                edges = norminv(cumsum([0, probs]), mean(data(:, varIdx)), std(data(:, varIdx)));
                categorys = discretize(data(:, varIdx), edges);
                data(:, varIdx) = categorys;
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% run for methods
            for imethod = 1:length(methods)
                fprintf("Method %s:\n", methods(imethod));
                if  methods(imethod) == "FCI"
                    tFCI = tic;
                    fciPag = FCI(ds, "test", "fisher", "heuristic", 3, "verbose", false, "pdsep", false, "cons", false);
                    fciPag.graph = fciPag.graph(~isLatent, ~isLatent);
                    timeFCI(inSamples, iter) = toc(tFCI);
                    shdsFCI(inSamples, iter) = structuralHammingDistancePAG(fciPag.graph, pagT);
                    [precisionsFCI(inSamples, iter), recallsFCI(inSamples, iter)] = precisionRecall(fciPag.graph, pagT);
                    [diffedgesFCI(inSamples, iter), diffendpointsFCI(inSamples, iter)] = diffEdgeEndpoints(fciPag.graph, pagT);
                    fprintf("time: %f\n", timeFCI(inSamples, iter));
                    fprintf("precision: %f\n", precisionsFCI(inSamples, iter));
                    fprintf("recall: %f\n", recallsFCI(inSamples, iter));
                
                elseif  methods(imethod) == "cFCI"
                    tCFCI=tic;
                    cfciPag = FCI(ds, "test", "fisher", "heuristic", 3, "verbose", false, "pdsep", false, "cons", true);
                    cfciPag.graph = cfciPag.graph(~isLatent, ~isLatent);
                    timeCFCI(inSamples, iter) = toc(tCFCI);
                    shdsCFCI(inSamples, iter) = structuralHammingDistancePAG(cfciPag.graph, pagT);
                    [precisionsCFCI(inSamples, iter), recallsCFCI(inSamples, iter)] = precisionRecall(cfciPag.graph, pagT);
                    [diffedgesCFCI(inSamples, iter), diffendpointsCFCI(inSamples, iter)] = diffEdgeEndpoints(cfciPag.graph, pagT);
                    fprintf("time: %f\n", timeCFCI(inSamples, iter));
                    fprintf("precision: %f\n", precisionsCFCI(inSamples, iter));
                    fprintf("recall: %f\n", recallsCFCI(inSamples, iter));
                    
                elseif methods(imethod) == "GSMAG"
                    tGS = tic;
                    % DG_GSMAG: 2(i->j); 3(j->i);
                    [gsMag, bs, gsIters, gs] = DG_GSMAG(data, newDiscreteVars, {}, {}, tol, false);
                    gsPag = mag2pag(gsMag);
                    
                    timeGS(inSamples, iter) = toc(tGS);
                    shdsGS(inSamples, iter) = structuralHammingDistancePAG(gsPag, pagT);
                    [precisionsGS(inSamples, iter), recallsGS(inSamples, iter)] = precisionRecall(gsPag, pagT);
                    [diffedgesGS(inSamples, iter), diffendpointsGS(inSamples, iter)] = diffEdgeEndpoints(gsPag, pagT);
                    fprintf("time: %f\n", timeGS(inSamples, iter));
                    fprintf("precision: %f\n", precisionsGS(inSamples, iter));
                    fprintf("recall: %f\n", recallsGS(inSamples, iter));
                    
                elseif methods(imethod) == "M3HC"
                    tM3HC = tic;
                    % DG_M3HC
                    [m3hcMag, m3hc_bs, m3hcIters, m3hc, mmpc_final] = DG_M3HC(data, newDiscreteVars, {}, {}, maxCondSetM3HC, threshold, tol, false, COR, skeleton);
                    m3hcMag = ag2mag(m3hcMag);
                    m3hcPag = mag2pag(m3hcMag);
                    
                    timeM3HC(inSamples, iter) = toc(tM3HC);
                    shdsM3HC(inSamples, iter) = structuralHammingDistancePAG(m3hcPag, pagT);
                    [precisionsM3HC(inSamples, iter), recallsM3HC(inSamples, iter)] = precisionRecall(m3hcPag, pagT);
                    [diffedgesM3HC(inSamples, iter), diffendpointsM3HC(inSamples, iter)] = diffEdgeEndpoints(m3hcPag, pagT);
                    fprintf("time: %f\n", timeM3HC(inSamples, iter));
                    fprintf("precision: %f\n", precisionsM3HC(inSamples, iter));
                    fprintf("recall: %f\n", recallsM3HC(inSamples, iter));
                end
            end
        end
    end
    
    nameALL = num2str(nVars) + "_" + num2str(maxParents) + "_" + num2str(nnLatent*10) + "_" + num2str(nnDiscrete*10);
    resPath = "res/" + nameALL + "/";
    if ~exist(resPath, "dir")
        mkdir(resPath);
    end
    save(resPath+"timeFCI.mat", "timeFCI"); save(resPath+"timeCFCI.mat", "timeCFCI");
    save(resPath+"precisionsFCI.mat", "precisionsFCI"); save(resPath+"precisionsCFCI.mat", "precisionsCFCI");
    save(resPath+"recallsFCI.mat", "recallsFCI"); save(resPath+"recallsCFCI.mat", "recallsCFCI");
    save(resPath+"shdsFCI.mat", "shdsFCI"); save(resPath+"shdsCFCI.mat", "shdsCFCI");
    save(resPath+"diffedgesFCI.mat", "diffedgesFCI"); save(resPath+"diffedgesCFCI.mat", "diffedgesCFCI");
    save(resPath+"diffendpointsFCI.mat", "diffendpointsFCI"); save(resPath+"diffendpointsCFCI.mat", "diffendpointsCFCI");
    
    csvwrite(resPath+"timeFCI.csv"', timeFCI); csvwrite(resPath+"timeCFCI.csv"', timeCFCI);
    csvwrite(resPath+"precisionsFCI.csv"', precisionsFCI); csvwrite(resPath+"precisionsCFCI.csv"', precisionsCFCI);
    csvwrite(resPath+"recallsFCI.csv"', recallsFCI); csvwrite(resPath+"recallsCFCI.csv"', recallsCFCI);
    csvwrite(resPath+"shdsFCI.csv"', shdsFCI); csvwrite(resPath+"shdsCFCI.csv"', shdsCFCI);
    csvwrite(resPath+"diffedgesFCI.csv"', diffedgesFCI); csvwrite(resPath+"diffedgesCFCI.csv"', diffedgesCFCI);
    csvwrite(resPath+"diffendpointsFCI.csv"', diffendpointsFCI); csvwrite(resPath+"diffendpointsCFCI.csv"', diffendpointsCFCI);
    
    save(resPath+"timeGS.mat", "timeGS"); save(resPath+"timeM3HC.mat", "timeM3HC");
    save(resPath+"precisionsGS.mat", "precisionsGS"); save(resPath+"precisionsM3HC.mat", "precisionsM3HC");
    save(resPath+"recallsGS.mat", "recallsGS"); save(resPath+"recallsM3HC.mat", "recallsM3HC");
    save(resPath+"shdsGS.mat", "shdsGS"); save(resPath+"shdsM3HC.mat", "shdsM3HC");
    save(resPath+"diffedgesGS.mat", "diffedgesGS"); save(resPath+"diffedgesM3HC.mat", "diffedgesM3HC");
    save(resPath+"diffendpointsGS.mat", "diffendpointsGS"); save(resPath+"diffendpointsM3HC.mat", "diffendpointsM3HC");
    
    csvwrite(resPath+"timeGS.csv"', timeGS); csvwrite(resPath+"timeM3HC.csv"', timeM3HC);
    csvwrite(resPath+"precisionsGS.csv"', precisionsGS); csvwrite(resPath+"precisionsM3HC.csv"', precisionsM3HC);
    csvwrite(resPath+"recallsGS.csv"', recallsGS); csvwrite(resPath+"recallsM3HC.csv"', recallsM3HC);
    csvwrite(resPath+"shdsGS.csv"', shdsGS); csvwrite(resPath+"shdsM3HC.csv"', shdsM3HC);
    csvwrite(resPath+"diffedgesGS.csv"', diffedgesGS); csvwrite(resPath+"diffedgesM3HC.csv"', diffedgesM3HC);
    csvwrite(resPath+"diffendpointsGS.csv"', diffendpointsGS); csvwrite(resPath+"diffendpointsM3HC.csv"', diffendpointsM3HC);
end
```

### 2. Real-world data
```matlab
clc; clear;
addpath("M3HC");
addpath("utilities");

COR = false;
maxCondSetM3HC = 10;
skeleton = "MMPC";
threshold = 5e-2;
tol = 10^-6;

kingdom = "BacteriaClay";
kingdom = "BacteriaSand";
% kingdom = "combi";

data_ori = readtable("data/"+kingdom+"/variables_all.csv", "Delimiter", ",", "ReadVariableNames", true, "ReadRowNames", true);
data = table2array(data_ori);
nVars = size(data, 2);

% bootstrap
numBoot = 100;
n = length(data);

rng(123)

for id = 0:numBoot
    fprintf("\n==========================================================\n");
    fprintf("Bootstrap: %d\n", id);
    
    if exist("res/real/" + kingdom + "/m3hcMag" + string(id) + ".mat", "file")
        continue;
    end

    if id == 0
        sample = data;
    else
        idx = randsample(n, n, true);
        disp(idx)
        sample = data(idx, :);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% zscore
    if kingdom == "combi"
        discreteVars = [11, 20, 21, 25];
        isDiscrete = false(1, nVars);
        isDiscrete(discreteVars) = true;
        sample(:, ~isDiscrete) = zscore(sample(:, ~isDiscrete));
    else
        discreteVars = [19, 20, 24];
        isDiscrete = false(1, nVars);
        isDiscrete(discreteVars) = true;
        sample(:, ~isDiscrete) = zscore(sample(:, ~isDiscrete));
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% define forbiddenEdges and predefinedEdges
    if kingdom == "combi"
        managementVars = [20, 21, 22, 23, 24, 25];
        forbiddenEdges = {};
        for i = 1:nVars
            for j = managementVars
                if i~= j
                    forbiddenEdges = [forbiddenEdges, [i, j]];
                end
            end
        end
    else
        managementVars = [19, 20, 21, 22, 23, 24];
        forbiddenEdges = {};
        for i = 1:nVars
            for j = managementVars
                if i~= j
                    forbiddenEdges = [forbiddenEdges, [i, j]];
                end
            end
        end
    end


    fprintf("running Dragon_M3HC...\n")
    [m3hcMag, m3hc_bs, m3hcIters, m3hc, mmpc_final] = DG_M3HC(sample, discreteVars, {}, forbiddenEdges, maxCondSetM3HC, threshold, tol, true, COR, skeleton);
    m3hcMag = ag2mag(m3hcMag);
    save("res/real/"+kingdom+"/m3hcMag"+string(id)+".mat", "m3hcMag");
    csvwrite("res/real/"+kingdom+"/m3hcMag"+string(id)+".csv"', m3hcMag);
end
```
