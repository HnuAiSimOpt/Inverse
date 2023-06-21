function act = ActiveEleSelec(temp_phase)
[nele,~] = size(temp_phase);
act = [];
for i = 1:nele
    if(length(find(temp_phase(i,:)>0.99))==1 || length(find(temp_phase(i,:)==0))==2)
        continue;
    else
        act = [act;i];
    end
end
