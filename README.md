# Dragon
Degenerate Gaussian (DG) score-based causal discovery

## Installation
1. Download the latest `Dragon.mltbx` file from the `release/` folder.
2. Double-click `Dragon.mltbx`.
3. MATLAB will automatically install the toolbox and add the required paths.

## Function and Parameters
### `DG_M3HC(data, discreteVars, predefined, forbidden, maxCondSet, threshold, tol, verbose, cor, skeleton)`
Run Dragon_M3HC.
#### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | matrix | Input data matrix of shape `(n_samples, n_variables)`. |
| `discreteVars` | vector | Indices of discrete variables. Default: `[]`. |
| `predefined` | cell array | Predefined directed edges `{[i,j]}` representing `i -> j`. Default: `{}`. |
| `forbidden` | cell array | Forbidden directed edges `{[i,j]}` representing `i -> j`. Default: `{}`. |
| `maxCondSet` | integer | Maximum conditioning set size used by MMPC. Default: `10`. |
| `threshold` | double | Significance level used by MMPC. Default: `0.05`. |
| `tol` | double | Numerical tolerance used in score computation. Default: `1e-6`. |
| `verbose` | logical | Whether to print debugging information. Default: `false`. |
| `cor` | logical | If `true`, use the correlation matrix; otherwise use the covariance matrix. Default: `false`. |
| `skeleton` | string | Method used to construct the initial skeleton. Default: `"MMPC"`. |
#### Returns
| Return | Type | Description |
|---------|------|-------------|
| `mag` | matrix | Estimated maximal ancestral graph. |
| `curScore` | double | Final overall DG-BIC score. |
| `iter` | integer | Number of greedy search iterations used. |
| `gs` | struct | Records of selected iterations during greedy search. |
| `skeleton_final` | cell array | Final skeleton obtained before greedy search. |
#### MAG Encoding
The returned MAG is represented as an adjacency matrix:
| Pattern | Interpretation |
|----------|---------------|
| `mag(i,j)=2`, `mag(j,i)=3` | `i → j` |
| `mag(i,j)=3`, `mag(j,i)=2` | `j → i` |
| `mag(i,j)=2`, `mag(j,i)=2` | `i ↔ j` |
| `mag(i,j)=0`, `mag(j,i)=0` | No edge |

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
