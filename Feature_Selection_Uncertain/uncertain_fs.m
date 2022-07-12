function   [selected_features,time]=uncertain_fs(data1,class_index)


start=tic;

[n,p]=size(data1);

selected_features=[];
selected_features1=[];
sf_dep = [];
b=[];
depArray=zeros(1,p-1); 
mode = zeros(1,p-1);
Y = data1(:,p);

 for i=1:p-1%the last feature is the class attribute, i.e., the target)
      
      %梯形
      x = unifrnd(0,1);
      if x < 0.25
          alpha = (4/25)*x + 0.01;
      
      elseif 0.25 <= x && x<0.75
          alpha = 0.05;
      elseif x >= 0.75
          alpha = 0.2*x-0.1;
      end
                 
       
     %for very sparse data 
     n1=sum(data1(:,i));
      if n1==0
        continue;
      end
     
               
        stop=0;
        CI=1;
     
        [CI,dep,pro]=my_cond_indep_fisher_z(data1,i,class_index,[],n,alpha);
        
        if pro < 0.5*alpha
             if ~isempty(selected_features)
               [CI,dep]=compter_dep_2(selected_features,i,class_index,3, 0, alpha, 'z',data1);
           end
           
           if CI==0 && ~isnan(dep)
                    
               selected_features=[selected_features,i]; %adding i to the set of selected_features
               p2=length(selected_features);
               selected_features1=selected_features;           
                  
               for j=1:p2
 
                  b=setdiff(selected_features1,selected_features(j), 'stable');
             
                  if ~isempty(b)
                     
                    [CI,dep]=optimal_compter_dep_2(b,selected_features(j),class_index,3, 0, alpha, 'z',data1);
                   
                    if CI==1 || isnan(dep)
                        selected_features1=b;  
                    end
                 end

            end
           end
        elseif pro<0.1
            depArray(1,i) = dep_an(data1(:,i),Y);
%             sf_dep = [sf_dep,i];
        end
selected_features=selected_features1;
 end
 len1 = length(selected_features);

     
  [max_dep,max_part] = sort(depArray,'descend');
  k = ceil(0.5*len1);
 add_features = max_part(1:k);
 selected_features = [selected_features,add_features];
 time=toc(start);
 end
 
   
function [ dep ] = dep_an(data,Y)

[n,~]=size(data);
card_U=length(Y);
card_ND=0;
D = pdist(data,'seuclidean');
DArray=squareform(D,'tomatrix');
for i=1:n   %遍历每个实例
     d=DArray(:,i);
     class=Y(i,1);
     card_ND=card_ND+card(d,Y,class,n);
%      card_ND=card_ND+card_K(d,Y,class,9);
end
dep=card_ND/card_U;
end

function [c]=card_K(distArray,Y,label,K)
% K个最近邻居标签信息
%
        [~,I]=sort(distArray);        
        cNum=0;
        for i=1:K
            ind=I(i);
            if Y(ind)==label
               cNum=cNum+1;
            end
        end
        c=cNum/K;
end

function [c]=card(sets,Y,label,N)
        [D,I]=sort(sets);        
        
        min_d=D(2,1);
        max_d=D(N,1);
        mean_d=1.5*(max_d-min_d)/(N-2);
        
        cNum=0;
        cTotal=1;
        ind2=I(2,1);
        if Y(ind2,1)==label
               cNum=cNum+1;
        end
        
         for j=3:N
             if D(j,1)-D(j-1,1)<mean_d
                 ind=I(j,1);
                 cTotal=cTotal+1;
                 if Y(ind,1)==label
                     cNum=cNum+1;
                 end
             else
                  break;
             end
         end  
        
         c=cNum/cTotal;

end
  
    
      
