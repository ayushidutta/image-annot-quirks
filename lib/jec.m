function [ Prob_w_given_I2 ] = jec( distance_vI_vJ,trainAnnotations,assignLabels )
%JEC @Yashaswi

[numOfTrainImages,numOfLabels] = size(trainAnnotations);
numOfTestImages = size(distance_vI_vJ,2);
Nw = zeros(1,numOfLabels);
for i = 1:numOfLabels
	Nw(1,i) = sum(trainAnnotations(:,i));
end;
Prob_w = Nw/numOfTrainImages;

cooccurFreq = zeros(numOfLabels,numOfLabels);
for i = 1:numOfTrainImages
	currLabels = find(trainAnnotations(i,:)==1);
	for j = 1:length(currLabels)
		ind1 = currLabels(j);
		for k = j+1:length(currLabels)
			ind2 = currLabels(k);
			cooccurFreq(ind1,ind2) = cooccurFreq(ind1,ind2) + 1;
			cooccurFreq(ind2,ind1) = cooccurFreq(ind2,ind1) + 1;
		end;
	end;
end;



Prob_w_given_I2 = zeros(numOfLabels,numOfTestImages);


for i = 1:numOfTestImages
    fiveNnIndxDist = zeros(assignLabels,2);
    trainDist = distance_vI_vJ(:,i);
    for j = 1:assignLabels
    	[val,indx] = min(trainDist);    
	fiveNnIndxDist(j,1) = indx;
	fiveNnIndxDist(j,2) = trainDist(indx);
	trainDist(indx) = inf; %10^15; 
    end;
    fiveNnIndxDist = sortrows(fiveNnIndxDist,2);
    
    firstNbrLabels = find(trainAnnotations(fiveNnIndxDist(1,1),:)==1);
    
    firstNbrLabelsFreq = zeros(length(firstNbrLabels),2);
    for j = 1:length(firstNbrLabels)
	firstNbrLabelsFreq(j,1) = firstNbrLabels(j);
	firstNbrLabelsFreq(j,2) = Prob_w(firstNbrLabels(j));
    end;
    firstNbrLabelsFreq = sortrows(firstNbrLabelsFreq,2);

    counter = 0;
    addedLabels = [];    
    for j = length(firstNbrLabels):-1:1
	labelIndx = firstNbrLabelsFreq(j,1);
	Prob_w_given_I2(labelIndx,i) = 1;
	counter = counter + 1;
	addedLabels = [addedLabels labelIndx];
	if( counter == assignLabels )
		break;
	end;
    end;

    if( counter < assignLabels )
	fourNbrsIndx = fiveNnIndxDist(2:assignLabels,1);
	%fourNbrsIndx = fiveNnIndxDist(1:5,1);
	
	fourNbrsLabels = []; 
	for j = 1:assignLabels-1
		%for j = 1:5
		currLabels = find(trainAnnotations(fourNbrsIndx(j),:)==1);
		fourNbrsLabels = [fourNbrsLabels currLabels];
	end;						
	temp = setdiff(fourNbrsLabels,addedLabels);
	
	fourNbrsLabelsFreq = zeros(length(temp),2);
	for j = 1:length(temp);
		fourNbrsLabelsFreq(j,1) = temp(j);
		tempcount = 0;
		for k = 1:length(fourNbrsLabels)	
			if( fourNbrsLabels(k)==temp(j) )
				tempcount = tempcount+1;
			end;
		end;
		fourNbrsLabelsFreq(j,2) = tempcount;
	end;

	fourNbrsLabels = temp;
	for j = 1:length(fourNbrsLabels)
		currVal = fourNbrsLabelsFreq(j,2);
		val1 = 0;
		for k = 1:length(addedLabels)
			val1 = val1 + (cooccurFreq(fourNbrsLabelsFreq(j,1),addedLabels(k))+eps);
		end;
		fourNbrsLabelsFreq(j,2) = currVal*val1;
	end;
	fourNbrsLabelsFreq = sortrows(fourNbrsLabelsFreq,2);
	
	for j = length(fourNbrsLabels):-1:1
		labelIndx = fourNbrsLabelsFreq(j,1);
		%Prob_w_given_I2(labelIndx,i) = fourNbrsLabelsFreq(j,2); 
		Prob_w_given_I2(labelIndx,i) = 1;
		addedLabels = [addedLabels labelIndx];
		counter = counter + 1;
		if( counter == assignLabels )
			break;
		end;
	end;
    end;
end; 



end

