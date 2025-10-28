function dataset=simulatedata(nodes, numCases,  type, varargin)
numNodes = length(nodes);
[domainCounts, isLatent, isManipulated, verbose] = process_options(varargin, 'domainCounts', 2*ones(1, numNodes), 'isLatent',  false(1, numNodes),'isManipulated', false(1, numNodes),  'verbose', false);
dataset.isLatent = isLatent;
dataset.isManipulated = isManipulated;
headers = cell(1, numNodes);
if isequal(type, 'discrete')
    edges=0;
    for i = 1:numNodes
        headers{i} = nodes{i}.name;
        edges = edges + length(nodes{i}.parents);
    end
    graph = spalloc(numNodes, numNodes, edges);
    for i = 1:numNodes
        graph(nodes{i}.parents, i) = 1;
    end
    ord = graphtopoorder(sparse(graph));

    data = zeros(numCases, length(nodes));
    node_values = zeros(1, numNodes);
    for case_cnt = 1:numCases
        % Loop over all nodes to be simulated
        for node = ord
            node_values(node) = randomSampleDiscrete(nodes{node}.cpt, node_values(nodes{node}.parents));                            
        end 
        data(case_cnt, :) = node_values;    
    end 
    data = data-1;
    dataset.data = data;
    dataset.domainCounts = domainCounts;
    dataset.type = 'discrete';

elseif isequal(type, 'gaussian')
%first step: creating the dataset
    edges=0;
     for i=1:numNodes
        headers{i} = nodes{i}.name;
        edges = edges + length(nodes{i}.parents);
    end
    graph = spalloc(numNodes, numNodes, edges);
    for i = 1:numNodes
        graph(nodes{i}.parents, i) = 1;
    end
    ord = graphtopoorder(sparse(graph));

    data = zeros(numCases, length(nodes));
    for node = ord
      % Sample
      data(:, node) = randomSampleGaussian(nodes{node}, data(:, nodes{node}.parents), numCases);                            
    end
    dataset.data = data;
    dataset.domainCounts = [];
    dataset.type = 'continuous';

elseif isequal(type, 'gaussian-nonlinear')
    edges = 0;
    for i = 1:numNodes
        headers{i} = nodes{i}.name;
        edges = edges + length(nodes{i}.parents);
    end
    graph = spalloc(numNodes, numNodes, edges);
    for i = 1:numNodes
        graph(nodes{i}.parents, i) = 1;
    end
    ord = graphtopoorder(sparse(graph));

    data = zeros(numCases, length(nodes));
    for node = ord
        data(:, node) = randomSampleGaussianNonlinear(nodes{node}, data(:, nodes{node}.parents), numCases);
    end
    dataset.data = data;
    dataset.domainCounts = [];
    dataset.type = 'continuous';

elseif isequal(type, 'non-gaussian')
    edges = 0;
    for i = 1:numNodes
        headers{i} = nodes{i}.name;
        edges = edges + length(nodes{i}.parents);
    end
    graph = spalloc(numNodes, numNodes, edges);
    for i = 1:numNodes
        graph(nodes{i}.parents, i) = 1;
    end
    ord = graphtopoorder(sparse(graph));

    data = zeros(numCases, length(nodes));
    for node = ord
        data(:, node) = randomSampleLinearNonGaussian(nodes{node}, data(:, nodes{node}.parents), numCases);
    end
    dataset.data = data;
    dataset.domainCounts = [];
    dataset.type = 'continuous';

else % type is unknown
    errprintf('Unknown data type:%s\n', type);
    dataset = nan;
end
dataset.headers = headers;
end


function value = randomSampleDiscrete(cpt, instance)
% value = RANDOMSAMPLEDISCRETE(CPT, INSTANCE)
% Returns a value for a discrete variable using the conditional probability table
% cpt, for parent instanciation instance.

if(isempty(instance))
    x = 1;
else
    s = size(cpt);
    x = mdSub2Ind_mex(s(2:end), instance);
end

cumprobs = cumsum(cpt(:,x));
value = [];
while isempty(value)
      value = find(cumprobs - rand > 0, 1 );
end
end


function value = randomSampleGaussian(node, instance, numCases)
% value = RANDOMSAMPLEGAUSSIAN(NODE, INSTANCE) Returns a value y = beta
% *instance+e for a conditional variable with parent instanciation instance

%calculate the normal distribution mean conditioned by parents value
if ~isempty(instance)
    distrMean = [node.mi + (node.beta * instance')]';
else
    distrMean = node.mi;
end

%normal distribution standard deviation
distrS = node.s * ones(numCases, 1);

%calculate the node value
value = normrnd(distrMean, distrS, numCases, 1);
end


function value = randomSampleGaussianNonlinear(node, instance, numCases)
% y = f( mi + beta * instance' ) + e,   e ~ N(0, s^2)

    % linear part
    if ~isempty(instance)
        zMean = [node.mi + (node.beta * instance')]';
    else
        zMean = node.mi * ones(numCases,1);
    end

    % choose linear
    f = local_get_activation(node);

    % apply nonlear part
    mu = f(zMean);
    sigma = node.s * ones(numCases,1);
    value = normrnd(mu, sigma, numCases, 1);
end

function f = local_get_activation(node)
    if isa(node.nonlin, 'function_handle')
        f = node.nonlin;
        return;
    end
    if ~ischar(node.nonlin) && ~isstring(node.nonlin)
        f = @(z) z;
        return;
    end
    switch lower(string(node.nonlin))
        case "tanh"
            f = @(z) tanh(z);
        case "relu"
            f = @(z) max(z,0);
        case "sigmoid"
            f = @(z) 1./(1 + exp(-z));
        case "softplus"
            f = @(z) log1p(exp(z));
        case "identity"
            f = @(z) z;
        otherwise
            f = @(z) z;
    end
end


function value = randomSampleLinearNonGaussian(node, instance, numCases)
    if ~isempty(instance)
        mu = [node.mi + (node.beta * instance')]';
    else
        mu = node.mi * ones(numCases,1);
    end

    noise = sampleNoise(node, numCases);

    value = mu + noise;
end

function e = sampleNoise(node, numCases)
    % Ensure noiseType and noiseParams fields exist
    if ~isfield(node, 'noiseType') || isempty(node.noiseType)
        node.noiseType = 'gaussian';
    end
    if ~isfield(node, 'noiseParams') || isempty(node.noiseParams)
        node.noiseParams = struct();
    end

    % Read parameters
    s = node.s;                         % target standard deviation
    typ = lower(string(node.noiseType));
    p = node.noiseParams;

    switch typ
        case "gaussian"
            % Standard Gaussian noise: e ~ N(0, s^2)
            e = s .* randn(numCases,1);

        case "laplace"
            % Laplace(0, b), variance = 2*b^2 => b = s/sqrt(2)
            b = s / sqrt(2);
            U = rand(numCases,1) - 0.5;  % Uniform(-0.5, 0.5)
            % Inverse CDF method for Laplace distribution
            e = -b .* sign(U) .* log(1 - 2*abs(U));
        
        case "uniform"
            % Uniform distribution on (-a, a), variance = a^2/3
            % => a = s*sqrt(3)
            a = s * sqrt(3);
            e = (2*rand(numCases,1) - 1) * a;

        case "mog"
            % Mixture of two Gaussians: w*N(0,1) + (1-w)*N(0, scale^2)
            % Then normalized to target variance s^2
            if ~isfield(p,'w'),     p.w = 0.9; end
            if ~isfield(p,'scale'), p.scale = 3; end
            w = p.w; c = p.scale;
            comp1 = randn(numCases,1);    % N(0,1)
            comp2 = c * randn(numCases,1);% N(0,c^2)
            choose = rand(numCases,1) < w;
            mix = choose .* comp1 + (~choose) .* comp2;
            % Variance of mixture: Var = w*1^2 + (1-w)*c^2 (mean = 0)
            varMix = w*1 + (1-w)*c^2;
            % Normalize to have variance s^2
            e = s ./ sqrt(varMix) .* mix;

        otherwise
            warning('Unknown noiseType "%s". Using Gaussian noise instead.', typ);
            e = s .* randn(numCases,1);
    end
end