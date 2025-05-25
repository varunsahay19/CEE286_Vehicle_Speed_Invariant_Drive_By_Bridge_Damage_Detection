% This code is owned by Jatin Aggarwal

function data = read_bridge(folder_name,acc_file_name)
% This function reads the bridge acceleration for the data collected
% Inputs are the folder name and filename which has bin files
% output is the data variable which contains acc_b and time
    acc_filename = [folder_name,acc_file_name];
    acc_b = {'v-x','v-y','v-z','Noise','b-x','b-y','b-z','Noise'}; % change the number of channels in based on the whether it was field or lab experiment
    number_of_channels = 12; % change the number to 8 based on the lab experiment
    fid_acc = fopen(acc_filename,'r');
    [accb,count] = fread(fid_acc,[number_of_channels+1,inf],'double');
    accb = accb';
    t = accb(:,1);
    acc = accb(:,2:13); % change the number 16 to 9 for lab experiment
    fclose(fid_acc);
    time = datetime(t,'ConvertFrom','datenum','TimeZone','America/Los_Angeles');
    data.acc = acc;
    data.time = time;
    data.labels = acc_b;
end