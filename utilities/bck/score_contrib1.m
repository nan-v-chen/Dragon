function sc = score_contrib(compMag, component, district, compSize, covMat, indMap, nSamples, tol)
% Author: nan.v.chen@gmail.com
% this function is developed based on the original GSMAG
% =======================================================================
% Inputs
% =======================================================================
%   compMag: sub_mag, including all nodes but only edges about members of this component (detected accoding to bi-edges) and the parents of those members
%   component: vector<INT>, members of this component
%   district: BOOL, members of compMag
%   compSize: INT, size of this component = length(component)
%   covMat: covariance matrix
%   indMap: struct, originalIndex and newIndex after doing one-hot embedding to discrete data
%   nSamples: INT, number of samples
%   tol: DOUBLE, tolerance
% =======================================================================
% Outputs
% =======================================================================
%   sc: score of compMag
% =======================================================================

% % test
% district = [1, 0, 1, 0, 0, 0, 0, 0, 0];
% dag = randomdag(10, 3);
% bn = dag2randBN(dag, 'gaussian');
% nLatent = ceil(1);
% isLatent = false(1, 10);
% isLatent(randsample(1:10, nLatent)) = true;
% magL = dag2mag(dag, isLatent);
% compMag = magL(~isLatent, ~isLatent);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% size after doing one-hot embedding to discrete data
sizeExp = max(cellfun(@max, {indMap(:).newIndex}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update district according to indMap
districtExp = false(1, sizeExp);
for i = 1:length(district)
    newInd = indMap(i).newIndex;
    districtExp(newInd) = district(i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update compMag according to indMap
compMagExp = zeros(sizeExp, sizeExp);
for i = 1:size(compMag, 1)
    newIndsI = indMap(i).newIndex;
    for j = 1:size(compMag, 2)
        newIndsJ = indMap(j).newIndex;
        value = compMag(i, j);
        if value ~= 0
            for x = newIndsI
                for y = newIndsJ
                    compMagExp(x, y) = value;
                end
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update component according to indMap
newComponent = [];
for i = 1:length(component)
    oriInd = component(i);
    newInd = indMap(oriInd).newIndex;
    newComponent = [newComponent, newInd];
end

if sum(district) == 1
    if sum(districtExp) == 1
        sc = logdet(2*pi*covMat(districtExp, districtExp)) + (nSamples-1)/nSamples;
    else
        sc = 0;
        for i = 1:length(newComponent)
            sc = sc + logdet(2*pi*covMat(newComponent(i), newComponent(i))) + (nSamples-1)/nSamples;
        end
        sc = sc / length(newComponent);
    end
else
    curCovMat = covMat(districtExp, districtExp);   %
    curCompMag = compMagExp(districtExp, districtExp);	% real sub_mag
    [~, ~, curHatCovMat, ~] = RICF_fit(curCompMag, curCovMat, tol);
    remParents = districtExp;
    remParents(newComponent) = false;  % the rest parents other than the nodes in the component
    parInds = remParents(districtExp);

    if any(remParents)
        l1 = compSize * log(2*pi);
        l2 = logdet(curHatCovMat) - log(prod(diag(curHatCovMat(parInds, parInds))));
        l3 = (nSamples-1)/nSamples * (trace(curHatCovMat\curCovMat)-sum(remParents));
        sc = l1+l2+l3;
    else
        l1 = compSize * log(2*pi);
        l2 = logdet(curHatCovMat);
        l3 = (nSamples-1) / nSamples * trace(curHatCovMat\curCovMat);
        sc = l1 + l2 + l3;
    end
end
