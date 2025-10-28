function [nodes, domainCounts] = dag2randBN(graph, type,  varargin)
numNodes = size(graph, 1);
nodes = cell(numNodes, 1);
domainCounts = nan(1, numNodes);

%topologically ordering the graph
% levels = toposort(graph);
% [~, order] = sortrows([(1:numNodes)' levels'], 2);

order = graphtopoorder(sparse(graph));
if isequal(type, 'discrete')
    [minNumStates, maxNumStates, headers] = process_options(varargin, ...
        'minNumStates', 2, 'maxNumStates', 5, ...
        'headers', arrayfun(@(x)sprintf('x%i',x), 1:numNodes, 'uniformOutput', false));

elseif isequal(type, 'gaussian')
    [miMinValue, miMaxValue, sMinValue, sMaxValue, betaMinValue, betaMaxValue, headers] = process_options(varargin,...
        'miMinValue', 0, 'miMaxValue', 0, 'sMinValue', 1, 'sMaxValue', 1, ...
        'betaMinValue', 0.1, 'betaMaxValue', 0.9, ...
        'headers', arrayfun(@(x)sprintf('x%i',x), 1:numNodes, 'uniformOutput', false));

elseif isequal(type, 'gaussian-nonlinear')
    [miMinValue, miMaxValue, sMinValue, sMaxValue, betaMinValue, betaMaxValue, headers, nonlin] = process_options(varargin,...
        'miMinValue', 0, 'miMaxValue', 0, 'sMinValue', 1, 'sMaxValue', 1, ...
        'betaMinValue', 0.1, 'betaMaxValue', 0.9, ...
        'headers', arrayfun(@(x)sprintf('x%i',x), 1:numNodes, 'uniformOutput', false), ...
        'nonlin', 'tanh');

elseif isequal(type, 'non-gaussian')
    [miMinValue, miMaxValue, sMinValue, sMaxValue, betaMinValue, betaMaxValue, headers, noiseType, noiseParams] = process_options(varargin,...
        'miMinValue', 0, 'miMaxValue', 0, 'sMinValue', 1, 'sMaxValue', 1, ...
        'betaMinValue', 0.1, 'betaMaxValue', 0.9, ...
        'headers', arrayfun(@(x)sprintf('x%i',x), 1:numNodes, 'uniformOutput', false), ...
        'noiseType', 'laplace', ...
        'noiseParams', struct());

else
    fprintf('Unknown data type: %s\n', type);
    return;
end

if isequal(type, 'discrete')
    %let's follow the topographical order ;-)
    for i = order
        %info
        %fprintf('Node %d of %d\n', i, length(order))

        %name, parents, number of states and domain counts of the node
        node.name = headers{i};
        node.parents = find(graph(:,i));
        numStates = round(rand()*(maxNumStates - minNumStates) + minNumStates);
        domainCounts(i) = numStates;

        %let's create the cpt...
        if isempty(node.parents)

            %firstly, let's sample...
            node.parents = [];
            node.cpt = dirichletsample(0.5*ones(1,numStates),1);

        else

            %agin, sampling
            node.cpt = dirichletsample(0.5*ones(1,numStates), domainCounts(node.parents));

        end

        nodes{i} = node;

    end
    
elseif isequal(type, 'gaussian') % gaussian
    for i = 1:numNodes

        clear tempNode options cOptions
        tempNode.name = headers{i};
        tempNode.parents = find(graph(:, i))';
        numParents = length(tempNode.parents);
        signs = (-1).^floor(rand(1, numParents).*2);
        tempNode.beta = [betaMinValue + (betaMaxValue-betaMinValue).*rand(1,numParents)].*signs;

        tempNode.mi = (miMaxValue - miMinValue) * rand() + miMinValue;
        tempNode.s = (sMaxValue - sMinValue) * rand() + sMinValue;

        nodes{i} = tempNode;
    
    end
    domainCounts =[];

elseif isequal(type, 'gaussian-nonlinear')
    for i = 1:numNodes
        clear tempNode
        tempNode.name = headers{i};
        tempNode.parents = find(graph(:, i))';
        numParents = length(tempNode.parents);
        signs = (-1).^floor(rand(1, numParents).*2);
        tempNode.beta = [betaMinValue + (betaMaxValue-betaMinValue).*rand(1,numParents)].*signs;
        tempNode.mi = (miMaxValue - miMinValue) * rand() + miMinValue;
        tempNode.s  = (sMaxValue - sMinValue) * rand() + sMinValue;

        %%%%%%%%%%%%%%%%%%%%%%%nonlinear settings
        tempNode.nonlin = nonlin;

        nodes{i} = tempNode;
    end
    domainCounts = [];

elseif isequal(type, 'non-gaussian')
    for i = 1:numNodes
        tempNode.name = headers{i};
        tempNode.parents = find(graph(:,i))';
        numParents = length(tempNode.parents);
        signs = (-1).^floor(rand(1, numParents).*2);
        tempNode.beta = [betaMinValue + (betaMaxValue-betaMinValue).*rand(1,numParents)].*signs;

        tempNode.mi = (miMaxValue - miMinValue) * rand() + miMinValue;
        tempNode.s  = (sMaxValue - sMinValue) * rand() + sMinValue;

        tempNode.noiseType = noiseType;
        tempNode.noiseParams = noiseParams;

        nodes{i} = tempNode;
    end
    domainCounts = [];
end
end
function theta = dirichletsample(alpha, dims)
% SAMPLE_DIRICHLET Sample N vectors from Dir(alpha(1), ..., alpha(k))
% theta = sample_dirichlet(alpha, N)
% theta(i,j) = i'th sample of theta_j, where theta ~ Dir

% We use the method from p. 482 of "Bayesian Data Analysis", Gelman et al.

k = length(alpha);
N = prod(dims);
theta = zeros(N, k);
scale = 1; % arbitrary
for i = 1:k
    theta(:,i) = gamrnd(alpha(i), scale, N, 1);
end
S = sum(theta,2);
theta = theta ./ repmat(S, 1, k);

%let's reshape theta following dims...
theta = reshape(theta', numel(theta), 1);
theta = reshape(theta, [k dims]);
end