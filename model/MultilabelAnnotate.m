classdef MultilabelAnnotate
    %MULTILABELANNOTATE Annotates based on label scores, per row i.e.
    %N(images)*L(labels)
    
    properties
    end
        
    methods (Static)
        
        % Assigns top K labels, K<<N
        function [predictedAnnot] = annotateTopK(labelScores,K)
            n_rows = size(labelScores,1);
            n_labels = size(labelScores,2);
            predictedAnnot = zeros([n_rows n_labels]);
            for i=1:n_rows
                for j=1:K
                    [~,idx] = max(labelScores(i,:));
                    labelScores(i,idx) = -inf; 
                    predictedAnnot(i,idx)=1;
                end;    
            end;   
        end
        
        function [predictedAnnot] = annotateAintersectB(annotA,annotB)
            n_rows = size(annotA,1);
            n_labels = size(annotA,2);
            predictedAnnot = zeros([n_rows n_labels]);
            for i=1:n_rows
                predictA = find(annotA(i,:)==1);
                for j=predictA
                    if (annotB(i,j)==1)
                        predictedAnnot(i,j)=1;
                    end;    
                end;    
            end;    
        end
        
        function [predictedAnnot] = annotateAminusB(annotA,annotB)
            n_rows = size(annotA,1);
            predictedAnnot = annotA;
            for i=1:n_rows
                ground = find(annotA(i,:)==1);
                for j=ground
                    if (annotB(i,j)==1)
                        predictedAnnot(i,j)=0;
                    end;    
                end;    
            end;    
        end
        
        function [predictedAnnot] = annotateAunionB(annotA,annotB)
            n_rows = size(annotA,1);
            predictedAnnot = annotA;
            for i=1:n_rows
                predictB = find(annotB(i,:)==1);
                for j=predictB
                    predictedAnnot(i,j)=1;
                end;    
            end;    
        end
        
        %fillKMode = 1-3 [Rare,Frequent,Random]
        function [predictedAnnot] = fillK(annot,trainAnnot,topK,fillKMode)
            n_rows = size(annot,1);
            n_labels = size(annot,2);
            predictedAnnot = annot;
            labelFreq = sum(trainAnnot)';
            labelRank = [(1:n_labels)' labelFreq];    
            if (fillKMode ==1)
                labelRank = sortrows(labelRank,2);
            elseif (fillKMode == 2)    
                labelRank = sortrows(labelRank,-2);
            end
            labelRank = labelRank(:,1);
            for i=1:n_rows
                predict = find(annot(i,:)==1);
                n_predict=length(predict);  
                ch=1;
                for j=n_predict+1:topK
                    if(fillKMode==3)
                        selectLabel = randi(n_labels);
                        while predictedAnnot(i,selectLabel)==1
                            selectLabel = randi(n_labels);
                        end;
                    else
                        selectLabel = labelRank(ch);
                        while predictedAnnot(i,selectLabel)==1
                            ch = ch+1;
                            selectLabel = labelRank(ch);
                        end;
                    end;
                    predictedAnnot(i,selectLabel)=1;
                end;    
            end;   
        end
        
    end
    
end

