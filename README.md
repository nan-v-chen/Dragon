# Dragon
Degenerate Gaussian (DG) score-based causal discovery

## Installation
1. Download the latest `Dragon.mltbx` file from the `release/` folder.
2. Double-click `Dragon.mltbx`.
3. MATLAB will automatically install the toolbox and add the required paths.

## Usage
```matlab
kingdom = "BacteriaClay";
% kingdom = "BacteriaSand";
% kingdom = "combi";

toolboxRoot = fileparts(which("DG_M3HC"));
dataFile = fullfile(toolboxRoot, "data", kingdom, "variables_all.csv");

dataOri = readtable(dataFile, ...
    "Delimiter", ",", ...
    "ReadVariableNames", true, ...
    "ReadRowNames", true);

data = table2array(dataOri);
nVars = size(data, 2);

numBoot = 100;
n = size(data, 1);

rng(123)

for id = 0:numBoot
    fprintf("\n==========================================================\n");
    fprintf("Bootstrap: %d\n", id);

    if id == 0
        sample = data;
    else
        idx = randi(n, n, 1);
        sample = data(idx, :);
    end

    if kingdom == "combi"
        discreteVars = [11, 20, 21, 25];
        managementVars = [20, 21, 22, 23, 24, 25];
    else
        discreteVars = [19, 20, 24];
        managementVars = [19, 20, 21, 22, 23, 24];
    end

    isDiscrete = false(1, nVars);
    isDiscrete(discreteVars) = true;
    sample(:, ~isDiscrete) = zscore(sample(:, ~isDiscrete));

    forbiddenEdges = {};
    for i = 1:nVars
        for j = managementVars
            if i ~= j
                forbiddenEdges = [forbiddenEdges, [i, j]];
            end
        end
    end

    fprintf("Running DG_M3HC...\n")
    [m3hcMag, m3hc_bs, m3hcIters, m3hc, mmpc_final] = ...
        DG_M3HC(sample, discreteVars, {}, forbiddenEdges);

    m3hcMag = ag2mag(m3hcMag);
end
```
