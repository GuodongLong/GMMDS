label1_index = find(VarName6==1); % index of class 1
label0_index = find(VarName6==0);

scatter_type = {'r*','bo'};

figure(1),plot(VarName2(label1_index),VarName3(label1_index),scatter_type{1}),hold on
plot(VarName2(label0_index),VarName3(label0_index),scatter_type{2})



label11_index = find(VarName18==1); % index of class 1
label00_index = find(VarName18==0);

scatter_type = {'r*','bo'};

figure(2),plot(VarName14(label11_index),VarName15(label11_index),scatter_type{1}),hold on
plot(VarName14(label00_index),VarName15(label00_index),scatter_type{2})