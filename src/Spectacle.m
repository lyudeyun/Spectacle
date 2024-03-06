classdef Spectacle < handle
    properties
        %% model info
        bm              % benchmark
        mdl             % the model with problematic NN controller
        mdl_m           % the model for fault injection by mutating weights
        D_run           % dataset for simulation, including ps_x and ps_y
        is_nor          % is normalized or not
        D               % dataset with test suite, for fault localization
        D_size          % the size of dataset
        T
        Ts
        %% nn parameters
        net             % neural network controller
        nn_stru         % nn structure (includes the output layer)
        layer_num       % the number of hidden layers (does not include output layer)
        traj_sz
        weight          % weight of hidden layers and output layer stored as a cell array according to the layer index
        bias            % bias of hidden layers and output layer stored as a cell array according to the layer index
        max_weight      % max weight value
        min_weight      % min weight value
        %% config of input signal of model
        in_name         % input signal of the model
        in_range        % the range of input signal
        in_span         % the time span of input signal
        %% config of input signal of controller
        icc_name        % input constant of controller
        ics_name        % input signal of controller
        ic_const        % input constant value of controller
        %% config of output signal of controller
        oc_name         % output signal of controller
        oc_span         % the time span of output signal
        %% specification
        phi_str
        phi
        sig_str
        %% parameters of fault localization
        sel_flag
        % sel_flag == 0, [only consider the weights in the middle layers];
        % sel_flag == 1, [input weights + the weights in the middle layers]
        % sel_flag == 2, [all the weights]
        roil_l                  % the beginning layer of roi
        roil_r                  % the ending layer of roi
        mut_posNum              % record the number of the weights with a non-zero FI of current mutant
        mut_posNum_list         % record the number of the weights with a non-zero FI of all mutants
        %% parallel computing
        core_num
    end
    methods
        function this = Spectacle(bm, mdl, mdl_m, D_run, is_nor, D, D_size, net, nn_stru, T, Ts, in_name, in_range, in_span, icc_name, ic_const, ics_name, oc_name, oc_span, phi_str, sig_str, sel_flag, core_num)

            this.bm = bm;
            this.mdl = mdl;
            this.mdl_m = mdl_m;
            this.D_run = D_run;
            this.is_nor = is_nor;
            this.D = D;
            this.D_size = D_size;

            this.T = T;
            this.Ts = Ts;

            this.net = net;
            this.nn_stru = nn_stru;
            if numel(nn_stru) ~= net.numLayers
                error('the layer num described in nn_stru and the layer num stored in nn file are inconsistent!');
            end
            this.layer_num = net.numLayers - 1;
            this.traj_sz = T/Ts + 1;
            this.weight = cell(1, net.numLayers);
            this.bias = cell(1, net.numLayers);

            for li = 1:this.layer_num + 1
                if li == 1
                    this.weight{1,li} = this.net.IW{1,1};
                    [li_row, ~] = size(this.net.IW{1,1});
                    if nn_stru(1,li) ~= li_row
                        error('the structure described in nn_stru and the structure stored in nn file are inconsistent!');
                    end
                else
                    this.weight{1,li} = this.net.LW{li,li-1};
                    [li_row, ~] = size(this.net.LW{li,li-1});
                    if nn_stru(1,li) ~= li_row
                        error('the structure described in nn_stru and the structure stored in nn file are inconsistent!');
                    end
                end
                this.bias{1,li} = this.net.b{li};
            end

            this.in_name = in_name;
            this.in_range = in_range;
            this.in_span = cell2mat(in_span);

            this.icc_name = icc_name;
            this.ic_const = ic_const;
            this.ics_name = ics_name;

            this.oc_name = oc_name;
            this.oc_span = cell2mat(oc_span);

            this.phi_str = phi_str;
            this.phi = STL_Formula('phi', this.phi_str);
            this.sig_str = sig_str;

            this.sel_flag = sel_flag;
            % choose the beginning layer and the ending layer for fault localization
            if sel_flag == 0
                this.roil_l = 2;
                this.roil_r = this.layer_num;
            elseif sel_flag == 1
                this.roil_l = 1;
                this.roil_r = this.layer_num;
            elseif sel_flag == 2
                this.roil_l = 1;
                this.roil_r = this.layer_num + 1;
            else
                error('Check the sel_flag!');
            end
            
            % obtain the range of NN controller' weights
            min_w = inf;
            max_w = -inf;
            weight_cp = this.weight;
            for li = this.roil_l: this.roil_r
                if strcmp(this.bm, 'AFC') && isequal(this.nn_stru, [15,15,15,15,1]) && li == 3
                    weight_cp{1,li}(15, 8) = 0;
                end
                if min_w > min(weight_cp{1,li}(:))
                    min_w = min(weight_cp{1,li}(:));
                end
                if max_w < max(weight_cp{1,li}(:))
                    max_w = max(weight_cp{1,li}(:));
                end
            end
            this.min_weight = min_w;
            this.max_weight = max_w;
            this.core_num = core_num;
        end
        %% functions
        function [rob, Br, tau_s, ic_sig_val, oc_sig_val] = signalDiagnosis(this, mdl, in_sig, spec_i)
            % signalDiagnosis function returns the robustness, Br, tau_s, controller's input signal and controller's output signal of a system execution
            %
            % Inputs:
            %   mdl: simulink model
            %   in_sig: the input signals of the simulink model
            %   spec_i: specification index
            % Outputs:
            %   rob: robustness
            %   Br: BreachSimulinkSystem of the Simulink model
            %   tau_s: the first timestamp at which the robustness value turns negative
            %   ic_sig_val: input signal of controller
            %   oc_sig_val: output signal of controller

            Br = BreachSimulinkSystem(mdl);
            Br.Sys.tspan = 0:this.Ts:this.T;

            if length(unique(this.in_span)) == 1
                con_type = 'UniStep';
                input_gen.cp = this.T/this.in_span(1,1);
            else
                con_type = 'VarStep';
                input_gen.cp = this.T./this.in_span;
            end

            input_gen.type = con_type;
            Br.SetInputGen(input_gen);

            if strcmp(con_type, 'UniStep')
                for i = 1:numel(this.in_name)
                    for cpi = 0:input_gen.cp - 1
                        eval(['Br.SetParam({''',this.in_name{1,i},'_u',num2str(cpi),'''}, in_sig{1,i}(1,cpi+1));']);
                    end
                end
            elseif strcmp(con_type, 'VarStep')
                for i = 1:numel(this.in_name)
                    for cpi = 0: input_gen.cp(1,i) - 1
                        eval(['Br.SetParam({''',this.in_name{1,i},'_u',num2str(cpi),'''}, in_sig{1,i}(1,cpi+1));']);
                        if cpi ~= input_gen.cp(1,i) - 1
                            eval(['Br.SetParam({''',this.in_name{1,i},'_dt',num2str(cpi),'''},this.in_span(1,i));']);
                        end
                    end
                end
            end

            Br.Sim(0:this.Ts:this.T);
            rob = Br.CheckSpec(this.phi);

            % extract ic signal and oc signal for forward impact analysis
            % (analysis the impact of a weight to the final result)
            ic_sig_val = sigMatch(Br, this.ics_name);
            oc_sig_val = sigMatch(Br, this.oc_name);

            if this.is_nor == 1
                x_gain = this.D_run.ps_x.gain;
                x_gain = diag(x_gain);
                x_offset = this.D_run.ps_x.xoffset;
                ic_sig_val = ic_sig_val - x_offset;
                ic_sig_val = x_gain * ic_sig_val;

                y_gain = this.D_run.ps_y.gain;
                y_offset = this.D_run.ps_y.xoffset;
                oc_sig_val = oc_sig_val/y_gain;
                oc_sig_val = oc_sig_val + y_offset;
            end

            % calculate the tau_s according to the given specification
            if rob > 0
                tau_s = this.T;
                return;
            end

            if strcmp(this.bm, 'ACC')
                interval_LR = {[0,50]};
            elseif strcmp(this.bm, 'AFC') && spec_i == 1
                interval_LR = {[0,30]};
            elseif strcmp(this.bm, 'AFC') && spec_i == 2
                interval_LR = {[10,30]};
            elseif strcmp(this.bm, 'WT')
                interval_LR = {[4,4.9], [9,9.9], [14,14.9]};
            elseif strcmp(this.bm, 'SC')
                interval_LR = {[30,35]};
            end

            scan_interval = zeros(1, this.traj_sz);
            neg_interval = zeros(1, this.traj_sz);
            for scan_i = 1: numel(interval_LR)
                LR = interval_LR{1, scan_i};
                LR = LR./this.Ts;
                scan_interval(1, LR(1,1)+1:LR(1,2)+1) = 1;
            end

            if strcmp(this.bm, 'ACC') && spec_i == 1
                left_neg_interval = sigMatch(Br, "d_rel") - 1.4 * sigMatch(Br, "v_ego") < 10;
                right_neg_interval = sigMatch(Br, "v_ego") > 30.1;
                neg_idx = (left_neg_interval + right_neg_interval > 0);
                neg_interval(1, neg_idx) = 1;
            elseif strcmp(this.bm, 'ACC') && spec_i == 2
                delta = sigMatch(Br, "d_rel") - 1.4 * sigMatch(Br, "v_ego");
                % i == 500, i = 501, true
                for i = 1: 451
                    if delta(1, i) < 12
                        i_behind = i + 50;
                        delta_behind = delta(1, i:i_behind);
                        if ~any(delta_behind >= 12)
                            neg_interval(1, i_behind) = 1;
                        end
                    end
                end
            elseif strcmp(this.bm, 'AFC') && spec_i == 1
                af = sigMatch(Br, "AF");
                mu = abs(af - 14.7)/14.7;
                neg_interval(mu >= 0.2) = 1;
            elseif strcmp(this.bm, 'AFC') && spec_i == 2
                af = sigMatch(Br, "AF");
                % i == 300, i = 301, true
                for i = 101: 286
                    if af(1, i) >= 1.1*14.7 || af(1, i) <= 0.9*14.7
                        i_behind = i + 15;
                        af_behind = af(1, i:i_behind);
                        if ~any(af_behind <= 1.1*14.7 & af_behind >= 0.9 *14.7)
                            neg_interval(1, i_behind) = 1;
                        end
                    end
                end
                % localize tau_s using online monitoring. Thank you, Zhenya
                % tau = 0;
                % time_ = Br.P.traj{1}.time;
                % time = round(time_, 2);
                % sig_list = cellstr(split(this.sig_str, ","));
                % X = sigMatch(Br, sig_list);
                % trace = [time; X];
                % [rob_up, rob_low] = stl_eval_mex_pw(this.sig_str, this.phi_str, trace, tau);
                % rob_neg = find(rob_up < 0);
                % tau_s = (rob_neg(1, 1) - 1) * this.Ts;
                % % extract the input and output signals of nn controller in the interval [0, tau_s]
                % ic_sig_val = ic_sig_val(:, 1:rob_neg(1, 1));
                % oc_sig_val = oc_sig_val(:, 1:rob_neg(1, 1));
                % return;
            elseif strcmp(this.bm, 'WT')
                h_error = abs(sigMatch(Br, "h_error"));
                neg_interval(h_error > 0.86) = 1;
            elseif strcmp(this.bm, 'SC')
                pressure = sigMatch(Br, "pressure");
                neg_interval(pressure < 87 | pressure > 87.5) = 1;
            end
            int_interval = neg_interval .* scan_interval;
            if ~ismember(1,int_interval)
                sys_time = datevec(datestr(now));
                % record current bug
                buglog_filename = 'buglog';
                for i = 1:numel(sys_time)
                    buglog_filename = strcat(buglog_filename, '_', num2str(sys_time(1,i)));
                end
                buglog_filename = [buglog_filename, '.mat'];
                save(buglog_filename, 'Br', 'mdl', 'in_sig');
                disp('robustness > 0');
                rob = rob + 100000;
                tau_s = this.T;
                return;
            end
            negl_idx = find(int_interval == 1);
            %
            tau_s = (negl_idx(1,1) - 1) * this.Ts;
            % extract the input and output signals of nn controller in the interval [0, tau_s]
            ic_sig_val = ic_sig_val(:,1:negl_idx(1,1));
            oc_sig_val = oc_sig_val(:,1:negl_idx(1,1));

            % the following codes have bugs on ACC2_spec2_dataset1_index_8, WT
            % % localize tau_s using online monitoring. Thank you, Zhenya
            % tau = 0;
            % time_ = Br.P.traj{1}.time;
            % time = round(time_, 2);
            % sig_list = cellstr(split(this.sig_str, ","));
            % X = sigMatch(Br, sig_list);
            % trace = [time; X];
            % [rob_up, rob_low] = stl_eval_mex_pw(this.sig_str, this.phi_str, trace, tau);
            % rob_neg = find(rob_up < 0);
            % tau_s = (rob_neg(1, 1) - 1) * this.Ts;
            %
            % % extract the input and output signals of nn controller in the interval [0, tau_s]
            % ic_sig_val = ic_sig_val(:, 1:rob_neg(1, 1));
            % oc_sig_val = oc_sig_val(:, 1:rob_neg(1, 1));
        end

        function [start_idx] = monitorRob(this, diagInfo)
            % monitorRob function returns the position from which we
            % construct execution spectrum.
            %
            % Inputs:
            %   diagInfo: 
            % Outputs:
            %   start_idx: 

            Br = diagInfo.Br;
            
            if strcmp(this.bm, 'ACC')
                interval_LR = {[0,50]};
            elseif strcmp(this.bm, 'AFC') && spec_i == 1
                interval_LR = {[0,30]};
            elseif strcmp(this.bm, 'AFC') && spec_i == 2
                interval_LR = {[10,30]};
            elseif strcmp(this.bm, 'WT')
                interval_LR = {[4,4.9], [9,9.9], [14,14.9]};
            end

            % scan_interval = zeros(1, this.traj_sz);
            % neg_interval = zeros(1, this.traj_sz);
            % for scan_i = 1: numel(interval_LR)
            %     LR = interval_LR{1, scan_i};
            %     LR = LR./this.Ts;
            %     scan_interval(1, LR(1,1)+1:LR(1,2)+1) = 1;
            % end
            % 
            % if strcmp(this.bm, 'ACC') && spec_i == 1
            %     left_neg_interval = sigMatch(Br, "d_rel") - 1.4 * sigMatch(Br, "v_ego") < 10;
            %     right_neg_interval = sigMatch(Br, "v_ego") > 30.1;
            %     neg_idx = (left_neg_interval + right_neg_interval > 0);
            %     neg_interval(1, neg_idx) = 1;
            % elseif strcmp(this.bm, 'ACC') && spec_i == 2
            %     delta = sigMatch(Br, "d_rel") - 1.4 * sigMatch(Br, "v_ego");
            %     % i == 500, i = 501, true
            %     for i = 1: 451
            %         if delta(1, i) < 12
            %             i_behind = i + 50;
            %             delta_behind = delta(1, i:i_behind);
            %             if ~any(delta_behind >= 12)
            %                 neg_interval(1, i_behind) = 1;
            %             end
            %         end
            %     end
            % elseif strcmp(this.bm, 'AFC') && spec_i == 1
            %     af = sigMatch(Br, "AF");
            %     mu = abs(af - 14.7)/14.7;
            %     neg_interval(mu >= 0.2) = 1;
            % elseif strcmp(this.bm, 'AFC') && spec_i == 2
            %     af = sigMatch(Br, "AF");
            %     % i == 300, i = 301, true
            %     for i = 101: 286
            %         if af(1, i) >= 1.1*14.7 || af(1, i) <= 0.9*14.7
            %             i_behind = i + 15;
            %             af_behind = af(1, i:i_behind);
            %             if ~any(af_behind <= 1.1*14.7 & af_behind >= 0.9 *14.7)
            %                 neg_interval(1, i_behind) = 1;
            %             end
            %         end
            %     end
            % elseif strcmp(this.bm, 'WT')
            %     h_error = abs(sigMatch(Br, "h_error"));
            %     neg_interval(h_error > 0.86) = 1;
            % end
            % int_interval = neg_interval .* scan_interval;
            % if ~ismember(1,int_interval)
            %     sys_time = datevec(datestr(now));
            %     % record current bug
            %     buglog_filename = 'buglog';
            %     for i = 1:numel(sys_time)
            %         buglog_filename = strcat(buglog_filename, '_', num2str(sys_time(1,i)));
            %     end
            %     buglog_filename = [buglog_filename, '.mat'];
            %     save(buglog_filename, 'Br', 'mdl', 'in_sig');
            %     disp('robustness > 0');
            %     rob = rob + 100000;
            %     tau_s = this.T;
            %     return;
            % end
            % negl_idx = find(int_interval == 1);
            % %
            % tau_s = (negl_idx(1,1) - 1) * this.Ts;
            
            if strcmp(this.bm, 'WT')
                % 对于WT这个spec来说，首先根据taus确定错误所在的大区间，然后根据大区间去精细化要分析的区间。这样可以过滤掉很多噪音
                % 分析rob的trace是否在一直下降。如果是，那么就大区间的开始去构建exe_spectrum；
                % 否则直接从rob下降的地方开始采集数据。所以可以分成两种情况讨论
                % (1) 在子interval_LR的第一个时间点违反，此时直接把当前的子interval的开始作为开始区间。
                % (2) 在interval_LR的第一个时间点之后违反，此时rob_up使用第一个时间点的rob
                % 之后，取得第一个出现小于这个值的时间点。这个点作为window的起始点。
                % 上面(2)的表述不太准确，更新之后：
                % 应该是在interval_LR的第一个时间点之后违反，此时选取violation之前
                % （也就是rob_up还大于0的区间）的最后一个平台期的最后一个点作为开始观察的起点。
                % TF = islocalmax(A,'FlatSelection','all');

                h_error = abs(sigMatch(Br, "h_error"));
                % narrow the search space to a sub interval
                for i = 1:numel(interval_LR)
                    if diagInfo.tau_s >= interval_LR{1,i}(1,1) && diagInfo.tau_s <= interval_LR{1,i}(1,2)
                        LR = interval_LR{1,i};
                        LR = LR./this.Ts;
                        roi_rob_idxl = LR(1,1) + 1;
                        roi_rob_idxr = LR(1,2) + 1;
                    end
                end

                % (1) 在子interval_LR的第一个时间点违反，此时直接把当前的子interval的开始作为开始区间
                if h_error(1, roi_rob_idxl) > 0.86
                    start_idx = roi_rob_idxl - 40;
                else
                    % (2) 在interval_LR的第一个时间点之后违反，此时rob_up使用第一个时间点的rob
                    % 之后，取得第一个出现小于这个值的时间点。这个点作为window的起始点。
                    rob_up = 0.86 - h_error(1, roi_rob_idxl);
                    for roii = roi_rob_idxl + 1:roi_rob_idxr
                        if 0.86 - h_error(1, roii) < rob_up
                            start_idx = roii;
                            break;
                        end
                    end
                end
            end

        end

        function var = initializeCell(this, str)
            % initializeCell function can initialize the corresponding cell
            % according to str.
            %
            % Inputs:
            %   str: cell name
            % Outputs:
            %   intialized variable

            switch str
                case 'weight2NeuronSnapShot'
                    % share the same structure with this.weight
                    var = {};
                    for li = 1:this.layer_num + 1
                        if li == 1
                            var{end+1} = zeros(size(this.net.IW{1,1}));
                        else
                            var{end+1} = zeros(size(this.net.LW{li,li-1}));
                        end
                    end
                case 'neuronOutSnapShot'
                    var = {};
                    for li = 1: this.layer_num + 1
                        var{end+1} = zeros(this.nn_stru(1,li), 1);
                    end
                case 'neuron2OutSnapShot'
                    var = {};
                    for li = 1: this.layer_num
                        var{end+1} = zeros(this.nn_stru(1,li), 1);
                    end
                case 'weight2OutSnapShot'
                    var = {};
                    for li = 1:this.layer_num + 1
                        if li == 1
                            var{end+1} = zeros(size(this.net.IW{1,1}));
                        else
                            var{end+1} = zeros(size(this.net.LW{li,li-1}));
                        end
                    end
                case 'weight2OutSnapShotBinary'
                    var = {};
                    for li = 1:this.layer_num + 1
                        if li == 1
                            var{end+1} = zeros(size(this.net.IW{1,1}));
                        else
                            var{end+1} = zeros(size(this.net.LW{li,li-1}));
                        end
                    end
                case 'exe_spectrum'
                    var = {};
                    for li = 1: this.layer_num + 1
                        if li == 1
                            [li_row, li_column] = size(this.net.IW{1,1});
                            layer_exe_spectrum = cell(li_row, li_column);
                            for i = 1: li_row
                                for j = 1: li_column
                                    layer_exe_spectrum{i,j} = zeros(1,4);
                                end
                            end
                            var{end+1} = layer_exe_spectrum;
                        else
                            [li_row, li_column] = size(this.net.LW{li,li-1});
                            layer_exe_spectrum = cell(li_row, li_column);
                            for i = 1: li_row
                                for j = 1: li_column
                                    layer_exe_spectrum{i,j} = zeros(1,4);
                                end
                            end
                            var{end+1} = layer_exe_spectrum;
                        end
                    end
                case 'sps_score'
                    var = {};
                    for li = 1: this.layer_num + 1
                        if li == 1
                            var{end+1} = zeros(size(this.net.IW{1,1}));
                        else
                            var{end+1} = zeros(size(this.net.LW{li,li-1}));
                        end
                    end
                otherwise
                    error('Check cell name!');
            end
        end
        function [weight2Neuron, neuronOut] = weightToNeuron(this, input)
            % weightToNeuron function calculates the first part of the feed forward impact.
            % (a weight of a neuron to the output of this neuron)
            %
            % Inputs:
            %   input: input signal of nn at a timestamp (at a certain frame)
            % Outputs:
            %   weight2Neuron: the impact of a weight of a neuron to the output of this neuron during a system execution
            %   neuronOut: the output of each neuron during a system execution

            % calculate the frame number in a system execution (tau_s/Ts)
            [row, column] = size(input);
            % initialize the neuronOut to store neuronOutSnapShot
            neuronOut = {};
            % initialize the weight2Neuron to store weight2NeuronSnapShot
            weight2Neuron = {};

            for j = 1: column
                % record the output of each neuron at a certain frame
                neuronOutSnapShot = this.initializeCell('neuronOutSnapShot');
                % record the impact of a weight of a neuron to the output of this neuron at a certain frame
                weight2NeuronSnapShot = this.initializeCell('weight2NeuronSnapShot');
                for i = 1: this.net.numLayers
                    if i == 1
                        transFcn = this.net.layers{i}.transferFcn();
                        transFcn = str2func(transFcn);
                        neuronOutSnapShot{1,1} = transFcn(this.weight{1,1} * input(:,j) + this.bias{1, i});
                        % calculate the impact of an input weight of a neuron to the output of this neuron
                        for ni = 1: this.nn_stru(1,i)
                            % the absolute sum of current neuron
                            weightoutSum = abs(this.weight{1,1}(ni,:)) * abs(input(:,j));
                            % calculate weight2NeuronSnapShot one by one
                            for wi = 1:row
                                weight2NeuronSnapShot{1,1}(ni,wi) = abs(this.weight{1,1}(ni,wi)) * abs(input(wi,1))/weightoutSum;
                            end
                        end
                    else
                        transFcn = this.net.layers{i}.transferFcn();
                        transFcn = str2func(transFcn);
                        neuronOutSnapShot{1,i} = transFcn(this.weight{1,i} * neuronOutSnapShot{1,i-1} + this.bias{1, i});
                        % calculate the impact of a layer weight of a neuron to the output of this neuron.
                        for ni = 1: this.nn_stru(1,i)
                            % the absolute sum of current neuron
                            weightoutSum = abs(this.weight{1,i}(ni,:)) * abs(neuronOutSnapShot{1,i-1});
                            % calculate weight2NeuronSnapShot one by one
                            for wi = 1:this.nn_stru(1,i-1)
                                weight2NeuronSnapShot{1,i}(ni,wi) = abs(this.weight{1,i}(ni,wi)) * abs(neuronOutSnapShot{1,i-1}(wi,1))/weightoutSum;
                            end
                        end
                    end
                end
                neuronOut{end+1} = neuronOutSnapShot;
                weight2Neuron{end+1} = weight2NeuronSnapShot;
            end
        end

        function [output, doutdneout] = neuronToOutputGradient(this, layerOut, layer_idx)
            % neuronToOutputGradient function calculates the gradient of
            % the final output to the output of a neuron at a certain layer.
            %
            % Inputs:
            %   layerOut: the output of the neurons at a certain layer
            %   layer_idx: range: [1, this.layer_num]
            % Outputs:
            %   output: final output in the form of dlarray
            %   doutdneout: the derivative of the final output to the
            %   output of an neuron at a certain layer.

            % assign neuronOut to a temp variable a_prev
            a_prev = layerOut;
            % perform inference to show how neuronOut turn into output
            for li = layer_idx: this.layer_num
                transFcn = this.net.layers{li+1}.transferFcn();
                transFcn = str2func(transFcn);
                a_prev = transFcn(this.weight{1, li+1} * a_prev + this.bias{1, li+1});
            end

            % assign the final temp variable a_prev to output
            output = a_prev;
            % calculate the gradient of the final output to the output of
            % the neurons at a certain layer
            doutdneout = dlgradient(output, layerOut);
        end

        function [neuron2Out] = neuronToOutput(this, input, neuronOut)
            % neuronToOutput function calculates the second part of the feed forward impact.
            % (the impact of a neuron's output at a certain layer to the final output)
            %
            % Inputs:
            %   input: input signal of nn at a timestamp (at a certain frame)
            %   neuronOut: the output of each neuron during a system execution
            % Outputs:
            %   neuron2Out: the gradient of the final output to the output of a neuron at a certain layer, i.e.,
            %   the impact of the output of a neuron to the final output during a system execution

            % calculate the frame number in a system execution (tau_s/Ts)
            [~, column] = size(input);
            % initialize the neuron2Out to store neuron2OutSnapShot
            neuron2Out = {};

            for j = 1:column
                % record the impact of the output of a neuron to the final output at a certain frame
                neuron2OutSnapShot = this.initializeCell('neuron2OutSnapShot');
                for li = 1:this.layer_num
                    layerOut = dlarray(neuronOut{1,j}{1,li});
                    [fval, gradval] = dlfeval(@this.neuronToOutputGradient, layerOut, li);
                    % dlarray to mat
                    neuron2OutSnapShot{1,li} = extractdata(gradval);
                end
                neuron2Out{end+1} = neuron2OutSnapShot;
            end
        end

        function diagInfo = forwardImpactAnalysis(this, mdl, in_sig, spec_i)
            % Given an external input signal of the system, forwardImpactAnalysis function returns the diagnostic information
            % of current system execution.
            % Inputs:
            %   mdl: simulink modelthe diagnostic information of the forward  impactof current system execution.
            %   in_sig: the input signals of the simulink model
            %   spec_i: specification index
            % Outputs:
            %   diagInfo: a struct store the diagnostic information of the forward impact of current system execution.
            %   diagInfo.Br: BreachSimulinkSystem of the Simulink model
            %   diagInfo.rob: robustness value
            %   diagInfo.state: the state of current system execution, passed or failed
            %   diagInfo.tau_s: the first timestamp at which the robustness value turns negative
            %   diagInfo.ic_sig_val: input signal of controller during [0, tau_s]
            %   diagInfo.oc_sig_val: output signal of controller during [0, tau_s]
            %   diagInfo.weight2Neuron: the impact of a weight of a neuron to the output of this neuron during a system execution
            %   diagInfo.neuronOut: the output of each neuron during a system execution
            %   diagInfo.neuron2Out: the gradient of the final output to the output of a neuron at a certain layer, i.e., the impact of the output of a neuron to the final output during a system execution
            %   diagInfo.weight2Out: the forward impact of each weight to the final output during a system execution

            % perform signal diagnosis
            [rob, Br, tau_s, ic_sig_val, oc_sig_val] = this.signalDiagnosis(mdl, in_sig, spec_i);

            % calculate the frame number in a system execution (tau_s/Ts)
            [~, frame_num] = size(ic_sig_val);

            % initialize weight2Out
            weight2Out = cell(1, frame_num);

            % obtain the state of the system execution
            if rob >= 0
                state = 1;
            else
                state = 0;
            end

            % calculate the first part
            [weight2Neuron, neuronOut] = this.weightToNeuron(ic_sig_val);
            % calculate the second part
            [neuron2Out] = this.neuronToOutput(ic_sig_val, neuronOut);
            % calculate weight2Out by first part * second part
            for fi = 1:frame_num
                % record the impact of a weight of a neuron to the final output at a certain frame
                weight2OutSnapShot = this.initializeCell('weight2OutSnapShot');
                for li = 1:this.layer_num + 1
                    if li < this.layer_num + 1
                        [~,col] = size(weight2Neuron{1,fi}{1,li});
                        % expand
                        expandedNeuron2Out = repmat(neuron2Out{1,fi}{1,li}, 1, col);
                        % multiply the corresponding positions of the
                        % weight2Neuron matrix and the expanded neuron2Out vector
                        weight2OutSnapShot{1,li} = weight2Neuron{1,fi}{1,li} .* expandedNeuron2Out;
                    else
                        weight2OutSnapShot{1,li} = weight2Neuron{1,fi}{1,li};
                    end
                end
                weight2Out{1, fi} = weight2OutSnapShot;
            end

            diagInfo.Br = Br;
            diagInfo.rob = rob;
            diagInfo.state = state;
            diagInfo.tau_s = tau_s;
            diagInfo.ic_sig_val = ic_sig_val;
            diagInfo.oc_sig_val = oc_sig_val;
            diagInfo.weight2Neuron = weight2Neuron;
            diagInfo.neuronOut = neuronOut;
            diagInfo.neuron2Out = neuron2Out;
            diagInfo.weight2Out = weight2Out;
        end

        function exe_spectrum = accordinglyAssign(this, exe_spectrum, diagInfo, hc, in_i)
            % accordinglyAssign function merges the execution spectrum of current system execution with
            % the execution spectrum of the system executions before.
            % Since online monitoring need more details about Br, state and
            % weight2Out are replace by diagInfo. But state and weight2Out
            % can be accessed via diagInfo.
            %
            % Inputs:
            %   exe_spectrum: the execution spectrum of the system executions before. For each weight, [passed&&topk, passed&&nottopk, failed&&topk, failed&&nottopk]
            %   state: passed or failed
            %   weight2Out: forward impact of each weight to the final output.
            %   diagInfo: 
            %   hc: a hyper parameter config
            %   in_i: the idx of current test
            % Outputs:
            %   exe_spectrum: execution spectrum of each weight after assignment
                
            state = diagInfo.state;    
            weight2Out = diagInfo.weight2Out;

            % frame number
            frame_num = numel(weight2Out);
            % initialize weight2OutBinary by converting weight2Out in a binary form. if topk, 1; else, 0.
            weight2OutBinary = cell(1, frame_num);
            for fi = 1: frame_num
                if in_i == 0
                    [~, weight2OutSnapShotBinary] = this.selectActWeight(weight2Out{1, fi});
                elseif strcmp(hc.es_mode, 'topk') && strcmp(hc.topk_mode, 'fixedNum')
                    % select topk weights for each frame according to hc, for ef. In this work, we only consider ef.
                    [~, weight2OutSnapShotBinary] = this.selectDescTopkWeightFixedNum(weight2Out{1, fi}, hc.topk);
                elseif strcmp(hc.es_mode, 'topk')
                    % select topk weights according to hc, for ef. In this work, we only consider ef.
                    [~, weight2OutSnapShotBinary] = this.selectDescTopkWeightFixedRatio(weight2Out{1, fi}, hc, in_i);
                elseif strcmp(hc.es_mode, 'activated')
                    % select activated weights for each frame according to hc
                    [~, weight2OutSnapShotBinary] = this.selectActWeight(weight2Out{1, fi});
                end
                weight2OutBinary{1, fi} = weight2OutSnapShotBinary;
            end

            if state == 1 
                % equal weights
                w = ones(frame_num, 1);
            elseif state == 0 && hc.window == 0
                w = this.traj_sz * ones(frame_num, 1)/frame_num;
            elseif state == 0 && hc.window == 1
                % hamming window  weight the weight2OutBinary (just a heuristic idea)
                init_w = hamming(frame_num * 2);
                scale = this.traj_sz/sum(init_w(1:frame_num, 1));
                w = scale * init_w;
            elseif state == 0 && hc.window == 2
                if hc.win_sz <= frame_num * this.Ts
                    % taus >= fixed_win
                    init_w = zeros(frame_num, 1);
                    start_idx = frame_num - hc.win_sz/this.Ts + 1;
                    init_w(start_idx:end, 1) = 1;
                    scale = this.traj_sz/(hc.win_sz/this.Ts);
                    w = scale * init_w;
                else
                    % taus < fixed_win
                    w = this.traj_sz * ones(frame_num, 1)/frame_num;
                end
            % 
            elseif state == 0 && hc.window == 3
                % for i=10:-1:1 end
                init_w = zeros(frame_num, 1);
                start_idx = this.monitorRob(diagInfo);
                win_sz = frame_num - start_idx + 1;
                init_w(start_idx:end, 1) = 1;
                scale = this.traj_sz/win_sz;
                w = scale * init_w;
            end

            for frame_i = 1:frame_num
                weight2OutSnapShotBinary = weight2OutBinary{1,frame_i};
                for li = this.roil_l:this.roil_r
                    [row, column] = size(weight2OutSnapShotBinary{1,li});
                    for i = 1:row
                        for j = 1:column
                            if state == 1 && weight2OutSnapShotBinary{1,li}(i,j) == 1
                                exe_spectrum_snapshot = w(frame_i, 1) * [1,0,0,0];
                            elseif state == 1 && weight2OutSnapShotBinary{1,li}(i,j) == 0
                                exe_spectrum_snapshot = w(frame_i, 1) * [0,1,0,0];
                            elseif state == 0 && weight2OutSnapShotBinary{1,li}(i,j) == 1
                                exe_spectrum_snapshot = w(frame_i, 1) * [0,0,1,0];
                            elseif state == 0 && weight2OutSnapShotBinary{1,li}(i,j) == 0
                                exe_spectrum_snapshot = w(frame_i, 1) * [0,0,0,1];
                            else
                                error('check your state or weight2OutSnapShotBinary!');
                            end
                            exe_spectrum{1,li}{i,j} = exe_spectrum{1,li}{i,j} + exe_spectrum_snapshot;
                        end
                    end
                end
            end
        end

        function [act_weight_info, weight2OutSnapShotBinary] = selectActWeight(this, weight2OutSnapShot)
            %  selectActWeight function returns the activated weights.
            %
            % Inputs:
            %   weight2OutSnapShot: execution spectrum at a certain frame
            % Outputs:
            %   act_weight_info: includes current weight's weight2OutSnapShot, layer_idx, right_endpoint_idx, left_endpoint_idx
            %   weight2OutSnapShotBinary: if a weight's forward impact is greater than 0, its weight2OutSnapShotBinary is 1; otherwise, 0.

            % initialize act_weight_info
            act_weight_info = {};
            for li = 1:this.layer_num + 1
                act_weight_info{end+1} = [];
            end

            % record the binary impact of a weight to the final output at a certain frame
            weight2OutSnapShotBinary = weight2OutSnapShot;

            % record the activated weights
            for li = this.roil_l:this.roil_r
                [row, column] = size(weight2OutSnapShot{1,li});
                for i = 1:row
                    for j = 1:column
                        if weight2OutSnapShot{1,li}(i,j) ~= 0
                            act_weight_info{1, li}(end+1, :) = [abs(weight2OutSnapShot{1,li}(i,j)), li, i, j];
                            weight2OutSnapShotBinary{1, li}(i, j) = 1;
                        end
                    end
                end
            end
        end

        function [topk_weight_info, weight2OutSnapShotBinary] = selectDescTopkWeightFixedNum(this, weight2OutSnapShot, topk)
            % selectDescTopkWeightFixedNum function returns the weights whose forward impact are within 5% in each layer. (ef)
            %
            % Inputs:
            %   weight2OutSnapShot: execution spectrum at a certain frame
            %   topk: the number of weights ranked in the topk at each layer
            % Outputs:
            %   topk_weight_info: includes singleWeight2OutSnapShot, layer_idx, right_endpoint_idx, left_endpoint_idx
            %   weight2OutSnapShotBinary: if a weight at a certain frame belongs topk_weight, 1; else, 0.
            % initialize topk_weight_info
            topk_weight_info = {};
            for li = 1:this.layer_num + 1
                topk_weight_info{end+1} = zeros(topk(1, li), 4);
            end
            % record the binary impact of a weight to the final output at a certain frame
            weight2OutSnapShotBinary = this.initializeCell('weight2OutSnapShotBinary');
            % select and record the topk weights
            for li = this.roil_l:this.roil_r
                [row, column] = size(weight2OutSnapShot{1,li});
                for i = 1:row
                    for j = 1:column
                        if abs(weight2OutSnapShot{1,li}(i,j)) > topk_weight_info{1,li}(topk(1,li), 1)
                            topk_weight_info{1,li}(topk(1,li), :) = [abs(weight2OutSnapShot{1,li}(i,j)), li, i, j];
                            topk_weight_info{1,li} = sortrows(topk_weight_info{1, li}, -1);
                        end
                    end
                end
            end
            for li = 1:numel(topk)
                for i = 1:topk(1,li)
                    % in case topk is set too large or the case that
                    % the number of positive values in weight2OutSnapShot is less than topk
                    if topk_weight_info{1,li}(i,1) > 0
                        weight2OutSnapShotBinary{1, topk_weight_info{1,li}(i,2)}(topk_weight_info{1,li}(i,3),topk_weight_info{1,li}(i,4)) = 1;
                    end
                end
            end
        end

        function [mut_posNum_list] = obtainposNum4TS(this, simlogFolder, tsf_size)
            % obtainposNum4TS function returns the number of the weights with a non-zero FI of all mutants. 
            %
            % Inputs:
            %   simlogFolder: the folder with the simlog files
            %   tsf_size: the size of the test suite for fault localization
            % Outputs:
            %   mut_posNum_list: record the number of the weights with a non-zero FI of all mutants
            
            dirOutput = dir(fullfile(simlogFolder,'*_M_*.mat'));
            fileNames = {dirOutput.name};
            mut_num = numel(fileNames);
            % record the mut_posNum info for each mutant
            mut_posNum_list = cell(1, mut_num);
            % initialize the struct mut_posNum to record the posNum details for each mutant
            % posNum4Test: record the number of the weights with a non-zero FI for each test, cell
            % mut_posNum_list{1,mut_i}.posNum4Test{1,in_i}(1,li)
            mut_posNum.posNum4Test = cell(1, tsf_size);
            % posNum4TS: record the details of the weights with a non-zero FI for the whole test suite, array
            % mut_posNum_list{1,mut_i}.posNum4TS(1,li)
            mut_posNum.posNum4TS = zeros(1, this.layer_num + 1);

            % initialize mut_posNum_list
            for mut_i = 1:mut_num
                mut_posNum_list{1,mut_i} = mut_posNum;
            end

            % check if a parallel pool exists
            pool = gcp('nocreate');
            % if a parallel pool exists, delete it
            if ~isempty(pool)
            	delete(pool);
            end
            % create a new parallel pool with the desired configuration
            pool = parpool(this.core_num);

            parfor mut_i = 1:numel(fileNames)
                str = fileNames{1, mut_i};
                simlog = load(str);
                % record the sum of the FI of the weights for each test
                ts_fi_sum_list = cell(1, tsf_size);
                % record the sum of the FI of the weights across the entire test suite, for mut_posNum.posNum4TS
                ts_fi_sum = this.initializeCell('weight2OutSnapShot');
                
                for in_i = 1: tsf_size
                    % record the sum of the FI of the weights for each test
                    cur_fi_sum = this.initializeCell('weight2OutSnapShot');
                    % frame number
                    frame_num = numel(simlog.cur_diagInfo_suite{1, in_i}.weight2Out);
                    % sum fi of each weights based on current system execution
                    for fi = 1:frame_num
                        for li = this.roil_l:this.roil_r
                            cur_fi_sum{1,li} = cur_fi_sum{1,li} + abs(simlog.cur_diagInfo_suite{1,in_i}.weight2Out{1,fi}{1,li});
                            ts_fi_sum{1,li} = ts_fi_sum{1,li} + abs(simlog.cur_diagInfo_suite{1,in_i}.weight2Out{1,fi}{1,li});
                        end
                    end
                    ts_fi_sum_list{1, in_i} = cur_fi_sum;
                    % count the number of the weights with a non-zero FI for each test
                    cur_posNum = zeros(1, this.layer_num + 1);
                    for li = this.roil_l: this.roil_r
                        cur_posNum(1,li) = nnz(cur_fi_sum{1,li});
                    end
                    mut_posNum_list{1,mut_i}.posNum4Test{1,in_i} = cur_posNum;
                end
                % count the number of the weights with a non-zero FI across the entire test suite
                for li = this.roil_l: this.roil_r
                    mut_posNum_list{1,mut_i}.posNum4TS(1,li) = nnz(ts_fi_sum{1,li});
                end
            end
            this.mut_posNum_list = mut_posNum_list;
        end

        function [topk_weight_info, weight2OutSnapShotBinary] = selectDescTopkWeightFixedRatio(this, weight2OutSnapShot, hc, in_i)
            % selectDescTopkWeightFixedRatio4Infer function returns topk weights from the weights with a positive forward impact. (ef)
            %
            % Inputs:
            %   weight2OutSnapShot: execution spectrum at a certain frame
            %   topk: the number of weights ranked in the topk at each hidden layer
            % Outputs:
            %   topk_weight_info: includes singleWeight2OutSnapShot, layer_idx, right_endpoint_idx, left_endpoint_idx
            %   weight2OutSnapShotBinary: if a weight at a certain frame belongs topk_weight, 1; else, 0.

            % initialize topk_weight_info
            topk_weight_info = cell(1, this.layer_num + 1);
            % record the binary impact of a weight to the final output at a certain frame
            weight2OutSnapShotBinary = this.initializeCell('weight2OutSnapShotBinary');

            % select and record the topk weights
            for li = this.roil_l:this.roil_r
                [row, column] = size(weight2OutSnapShot{1,li});
                % obtain the number of weights with a positive forward impact at current frame
                if strcmp(hc.topk_mode, 'fixedRatio4Infer')
                    posNum = sum(sum(abs(weight2OutSnapShot{1,li}) > 0));
                elseif strcmp(hc.topk_mode, 'fixedRatio4Test')
                    posNum = this.mut_posNum.posNum4Test{1,in_i}(1,li);
                elseif strcmp(hc.topk_mode, 'fixedRatio4TS')
                    posNum = this.mut_posNum.posNum4TS(1,li);
                else
                    error('Check current topk_mode!');
                end
                li_topk_weight_info = zeros(ceil(posNum * hc.topk(1,li)), 4);
                topk_weight_info{1,li} = li_topk_weight_info;
                if posNum == 0
                    % this situation is potentially plausible
                    continue;
                end
                
                for i = 1:row
                    for j = 1:column
                        if abs(weight2OutSnapShot{1,li}(i,j)) > topk_weight_info{1,li}(end,1)
                            topk_weight_info{1,li}(end,:) = [abs(weight2OutSnapShot{1,li}(i,j)), li, i, j];
                            topk_weight_info{1,li} = sortrows(topk_weight_info{1,li},-1);
                        end
                    end
                end
            end

            for li = this.roil_l:this.roil_r
                for i = 1:size(topk_weight_info{1,li},1)
                    % % in case topk is set too large or the case that
                    % % the number of positive values in weight2OutSnapShot is less than topk
                    if topk_weight_info{1,li}(i,1) > 0
                        weight2OutSnapShotBinary{1, topk_weight_info{1,li}(i,2)}(topk_weight_info{1,li}(i,3),topk_weight_info{1,li}(i,4)) = 1;
                    end
                end
            end
        end

        function [topk_weight_info, weight2OutSnapShotBinary] = selectAscTopkWeight(this, weight2OutSnapShot, topk)
            % selectAscTopkWeight function returns the weights whose forward impact are within 95% in each layer. (nf)
            %
            % Inputs:
            %   weight2OutSnapShot: execution spectrum at a certain frame
            %   topk: the number of weights ranked in the topk at each layer
            % Outputs:
            %   topk_weight_info: includes singleWeight2OutSnapShot, layer_idx, right_endpoint_idx, left_endpoint_idx
            %   weight2OutSnapShotBinary: if a weight at a certain frame belongs topk_weight, 1; else, 0.

            % initialize topk_weight_info
            topk_weight_info = {};
            % initialize a new topk
            new_topk = zeros(size(topk));
            for li = 1:this.layer_num + 1
                % the sum of weight num of current layer minus topk
                if li == 1
                    new_topk(1,li) = numel(this.net.IW{1,1}) - topk(1, li);
                    topk_weight_info{end+1} = zeros(new_topk(1,li), 4);
                else
                    new_topk(1,li) = numel(this.net.LW{li,li-1}) - topk(1, li);
                    topk_weight_info{end+1} = zeros(new_topk(1,li), 4);
                end
            end

            % record the binary impact of a weight to the final output at a certain frame
            weight2OutSnapShotBinary = this.initializeCell('weight2OutSnapShotBinary');

            % select and record the topk weights
            for li = this.roil_l: this.roil_r
                [row, column] = size(weight2OutSnapShot{1,li});
                for i = 1:row
                    for j = 1:column
                        if abs(weight2OutSnapShot{1,li}(i,j)) > topk_weight_info{1,li}(new_topk(1,li), 1)
                            topk_weight_info{1, li}(new_topk(1,li), :) = [abs(weight2OutSnapShot{1,li}(i,j)), li, i, j];
                            topk_weight_info{1, li} = sortrows(topk_weight_info{1, li}, -1);
                        end
                    end
                end
            end

            for li = 1: numel(new_topk)
                for i = 1:new_topk(1,li)
                    % in case topk is set too small or the case that
                    % the number of positive values in weight2OutSnapShot is less than topk
                    if topk_weight_info{1,li}(i,1) > 0
                        weight2OutSnapShotBinary{1, topk_weight_info{1,li}(i,2)}(topk_weight_info{1,li}(i,3),topk_weight_info{1,li}(i,4)) = 1;
                    end
                end
            end
        end

        function [sps_weight_info, sps_score] = SBFL(this, exe_spectrum, hc, mi)
            % SBFL function performs a spectrum-based fault localization.
            %
            % Inputs:
            %   exe_spectrum: execution spectrum of all weights
            %   hc: current hyper config, including hc.window, hc.es_mode, hc.topk,
            %   hc.sps_metric, hc.tops
            %   mi: the index of current suspiciousness metric
            % Outputs:
            %   sps_weight_info: includes suspiciousness score of a suspicious weight, weight value, layer_idx, right_endpoint_idx, left_endpoint_idx
            %   sps_score: suspiciousness scores of all weights

            % initialize sps_weight_info
            sps_weight_info = repmat(-inf, hc.tops, 5);
            % initialize sps_score
            sps_score = this.initializeCell('sps_score');
            % calculate the suspiciousness score for each weight and record the tops weights
            for li = this.roil_l: this.roil_r
                [row, column] = size(exe_spectrum{1,li});
                for i = 1:row
                    for j = 1:column
                        cs = exe_spectrum{1,li}{i,j};
                        single_sps_score = spsCalculator(cs(1,1), cs(1,2), cs(1,3), cs(1,4), hc.sps_metric{1, mi});
                        % update the score of current weight
                        sps_score{1,li}(i,j) = single_sps_score;
                        % for NaN 
                        if isnan(single_sps_score)  
                            single_sps_score = 0;
                        end
                        % select the tops suspicious weights
                        if single_sps_score > sps_weight_info(hc.tops, 1)
                            sps_weight_info(hc.tops, :) = [single_sps_score, this.weight{1,li}(i,j), li, i, j];
                            sps_weight_info = sortrows(sps_weight_info, -1);
                        end
                    end
                end
            end
            % restore the scores of the weights whose previous scores are NaN
            for i = 1:hc.tops
                sps_weight_info(i,1) = sps_score{1,sps_weight_info(i,3)}(sps_weight_info(i,4),sps_weight_info(i,5));
            end
        end

        function [hyper_all_sps_weight, hyper_all_sps_score, state_set, hyper_exe_spectrum, diagInfo_suite] = AutoFL(this, mdl, spec_i, test_suite, hyper_config)
            % AutoFL function returns the information of topk suspicious weights and suspiciousness scores of all weights
            % under different hyper parameter configs. AutoFL function is a core function of our fault localization method.
            %
            % Inputs:
            %   mdl: simulink model
            %   spec_i: specification index
            %   test_suite: test suite used for fault localization
            %   hyper_config: different combinations of hyper parameters of our fault localization method,
            %   i.e., {hc_1, hc_2, hc_3, ..., hc_n}.
            %   hc_i.window = 0, hc_i.topk = [0,5,5,0],
            %   hc_i.sps_metric = {sm_1, sm_2, sm_n}, hc_i.tops = [0,5,5,0]
            % Outputs:
            %   hyper_all_sps_weight: includes the information of suspicious weights obtained by different suspiciousness measures under different hyper parameter configs.
            %   Each all_sps_weight includes topk suspicious weights calculated by different suspiciousness measures that should be responsible for the violation.
            %   each row in sps_weight_info is composed of suspiciousness score of a suspicious weight, weight value, layer_idx, right_endpoint_idx, left_endpoint_idx.
            %   hyper_all_sps_score: includes suspiciousness scores obtained by different suspiciousness measures under different hyper parameter configs.
            %   Each all_sps_score includes suspiciousness scores of each weight calculated by different suspiciousness measures.
            %   state_set: record the state of each system execution. (0: failed; 1: passed)
            %   hyper_exe_spectrum: includes the execution spectrums of all weights based on the test suite under different hyper parameter configs.
            %   each exe_spectrum includes the execution spectrum of all weights based on the test suite under a hyper parameter config.
            %   diagInfo_suite: includes the diagnostic information of the forward impact of each system execution.

            % we also need to ensure weight_num > hc.topk(i) or total_weight_num > topk, a potential bug

            % the number of hyper configs
            hc_num = numel(hyper_config);
            % initialize state_set
            state_set = zeros(numel(test_suite), 1);
            % initialize hyper_exe_spectrum (one hyper_config_i, one exe_spectrum)
            hyper_exe_spectrum = cell(1, hc_num);
            for hi = 1:hc_num
                % initialize exe_spectrum
                exe_spectrum = this.initializeCell('exe_spectrum');
                hyper_exe_spectrum{1, hi} = exe_spectrum;
            end
            % obtain execution spectrum of all weights based on the given test suite
            diagInfo_suite = cell(1, numel(test_suite));
            for in_i = 1: numel(test_suite)
                in_sig = test_suite{1, in_i};
                diagInfo = this.forwardImpactAnalysis(mdl, in_sig, spec_i);
                state_set(in_i, 1) = diagInfo.state;
                % calculate exe_spectrum one by one
                for hi = 1:hc_num
                    exe_spectrum = this.accordinglyAssign(hyper_exe_spectrum{1, hi}, diagInfo, hyper_config{1, hi});
                    hyper_exe_spectrum{1, hi} = exe_spectrum;
                end
                diagInfo_suite{1, in_i} = diagInfo;
            end
            % perform SBFL
            hyper_all_sps_weight = cell(1, hc_num);
            hyper_all_sps_score = cell(1, hc_num);
            for hi = 1: hc_num
                hc = hyper_config{1, hi};
                all_sps_weight = cell(1, numel(hc.sps_metric));
                all_sps_score = cell(1, numel(hc.sps_metric));
                for mi = 1: numel(hc.sps_metric)
                    [sps_weight_info, sps_score] = this.SBFL(hyper_exe_spectrum{1, hi}, hc, mi);
                    all_sps_weight{1, mi} = sps_weight_info;
                    all_sps_score{1, mi} = sps_score;
                end
                hyper_all_sps_weight{1, hi} = all_sps_weight;
                hyper_all_sps_score{1, hi} = all_sps_score;
            end
        end

        function [weight_candidate, result] = RQ1(this, spec_i, tsf_size, tsf_mode, wsel_mode, budget, behavior_change_rate, hyper_config, specl_l, specl_r)
            % RQ1 function aims to validate the effectiveness of our fault localization method, according to RQ1 of Arachne.
            % (only for debugging, UseParallel: 'no')
            %
            % Inputs:
            %   spec_i: specification index
            %   tsf_size: the size of the test suite for fault localization
            %   tsf_mode: the mode of building the test suite for fault localization
            %   wsel_mode: 'all': select all the weights; 'average': only select the top 50% weights whose
            %   forward impact is greater than the average. (1) value
            %   average; (2) number average. According to the source code of
            %   Arachne, it should be the first case.
            %   budget: the size of guassian distribution mean set
            %   behavior_change_rate: behavior change rate after inserting a bug
            %   hyper_config: different combinations of hyper parameters of our fault localization method,
            %   i.e., {config_1, config_2, config_3, ..., config_n}.
            %   specl_l: split task and speed up calculation
            %   specl_r: split task and speed up calculation
            % Outputs:
            %   weight_candidate: the information of fault localization
            %   result: the information of fault localization in the table form

            % generate test suite
            test_suite = generateTestSuite(this.D, tsf_size, tsf_mode);
            tsf_size = numel(test_suite);

            weight_candidate = [];
            if strcmp(wsel_mode, 'average')
                % select the weight whose forward impact is greater than the average
                for in_i = 1: numel(test_suite)
                    in_sig = test_suite{1, in_i};
                    diagInfo = this.forwardImpactAnalysis(this.mdl, in_sig, spec_i);
                    % obtain top50_weight of current system execution
                    top50_weight = selectTop50Weight(diagInfo.weight2Out, this.roil_l, this.roil_r, wsel_mode);
                    weight_candidate = [weight_candidate; top50_weight];
                end
                % remove duplicated rows
                weight_candidate = unique(weight_candidate(:,1:3), 'rows');
            elseif strcmp(wsel_mode, 'all')
                for li = this.roil_l:this.roil_r
                    [rows, cols] = size(this.weight{1, li});
                    for i = 1:rows
                        for j = 1:cols
                            weight_candidate = [weight_candidate; li, i, j];
                        end
                    end
                end
            elseif strcmp(wsel_mode, 'specify')
                for li = specl_l:specl_r
                    [rows, cols] = size(this.weight{1, li});
                    for i = 1:rows
                        for j = 1:cols
                            weight_candidate = [weight_candidate; li, i, j];
                        end
                    end
                end
            elseif strcmp(wsel_mode, 'scattered')
                file = load('candidate.mat');
                weight_candidate = file.candidate;
            else
                error('Check your weight selection mode!');
            end

            candidate_num = size(weight_candidate, 1);
            % obtain FL_info of original mdl
            [ori_hyper_all_sps_weight, ori_hyper_all_sps_score, ori_state_set, ori_hyper_exe_spectrum, ori_diagInfo_suite] = this.AutoFL(this.mdl, spec_i, test_suite, hyper_config);
            % calculate safety rate
            ori_safety_rate = numel(find(ori_state_set > 0));
            % save FL info of original mdl
            FL_info_file = [this.mdl, '_selflag_', num2str(this.sel_flag), '_size_', num2str(tsf_size), '_mode_', tsf_mode, '_FL_Info.mat'];
            save(FL_info_file, 'ori_hyper_all_sps_weight', 'ori_hyper_all_sps_score', 'ori_state_set', 'ori_safety_rate', 'ori_hyper_exe_spectrum', 'ori_diagInfo_suite');

            % the number of hyper configs
            hc_num = numel(hyper_config);
            % record weight bugs for each weight
            hyper_layer_idx_cell = cell(1, hc_num);
            hyper_right_idx_cell = cell(1, hc_num);
            hyper_left_idx_cell = cell(1, hc_num);
            hyper_bug_value_cell = cell(1, hc_num);
            % record delta state (failed -> passed; passed -> failed)
            hyper_delta_state_cell = cell(1, hc_num);
            % record the number of passed cases
            hyper_safety_rate_cell = cell(1, hc_num);
            % record the changes of safety rate
            hyper_delta_safety_cell = cell(1, hc_num);
            % record the suspiciousness scores for comparison
            hyper_ori_sps_score_cell = cell(1, hc_num);
            hyper_cur_sps_score_cell = cell(1, hc_num);
            % record the ranks of the suspiciousness scores for comparison
            hyper_pre_rank_cell = cell(1, hc_num);
            hyper_cur_rank_cell = cell(1, hc_num);
            % is detectable or not
            hyper_is_detected_cell = cell(1, hc_num);

            for hi = 1:hc_num
                % record weight bugs for each weight
                hyper_layer_idx_cell{1, hi} = cell(1, candidate_num);
                hyper_right_idx_cell{1, hi} = cell(1, candidate_num);
                hyper_left_idx_cell{1, hi} = cell(1, candidate_num);
                hyper_bug_value_cell{1, hi} = cell(1, candidate_num);
                % record delta state (failed -> passed; passed -> failed)
                hyper_delta_state_cell{1, hi} = cell(1, candidate_num);
                % record the number of passed cases
                hyper_safety_rate_cell{1, hi} = cell(1, candidate_num);
                % record the changes of safety rate
                hyper_delta_safety_cell{1, hi} = cell(1, candidate_num);
                % record the suspiciousness scores for comparison
                hyper_ori_sps_score_cell{1, hi} = cell(1, candidate_num);
                hyper_cur_sps_score_cell{1, hi} = cell(1, candidate_num);
                % record the ranks of the suspiciousness scores for comparison
                hyper_pre_rank_cell{1, hi} = cell(1, candidate_num);
                hyper_cur_rank_cell{1, hi} = cell(1, candidate_num);
                % is detectable or not
                hyper_is_detected_cell{1, hi} = cell(1, candidate_num);
            end

            % % check if a parallel pool exists
            % pool = gcp('nocreate');
            % % if a parallel pool exists, delete it
            % if ~isempty(pool)
            % 	delete(pool);
            % end
            % % create a new parallel pool with the desired configuration
            % pool = parpool(this.core_num);

            % perform fault localization after inserting a bug
            for wi = 1: candidate_num
                disp(wi);
                flag = 1;
                bug_i = 1;
                guassianMean = [];
                % gaussian distribution mean value set used to generate bugs
                gap = 1; % initial gap
                cur_gm = 0; % current gm
                gm_r = [];
                while numel(gm_r) < budget/2
                    gm_r = [gm_r, cur_gm];
                    cur_gm = cur_gm + gap; % update gm
                    gap = gap * 2; % update gap
                end

                gm_l = -gm_r;
                % mean value set of gaussian distribution used to generate bugs
                gm = [gm_l', gm_r'];
                for r = 1:budget/2
                    guassianMean = [guassianMean, gm(r,:)];
                end

                while flag
                    li = weight_candidate(wi, 1);
                    i = weight_candidate(wi, 2);
                    j = weight_candidate(wi, 3);
                    % insert a bug
                    cur_bug = [li, i, j, normrnd(guassianMean(1, bug_i), 1)];
                    % buggy model name
                    bug_mdl = [this.mdl_m, '_', num2str(li), '_', num2str(i), '_', num2str(j), '_bug_', num2str(bug_i)];
                    % insert the above weight bug to simulink model
                    insertWeightBug(this.mdl_m, bug_mdl, cur_bug);
                    % restore this.weight, copy the net's weight to
                    % this.weight and insert cur_bug to this.weight
                    this.weight = weightReset(this.net, cur_bug);
                    % obtain FL_info of mdl with an inserted bug
                    [cur_hyper_all_sps_weight, cur_hyper_all_sps_score, cur_state_set, cur_hyper_exe_spectrum, cur_diagInfo_suite] = this.AutoFL(bug_mdl, spec_i, test_suite, hyper_config);
                    % calculate delta state
                    delta_state = nnz(ori_state_set ~= cur_state_set);
                    % calculate safety rate
                    cur_safety_rate = numel(find(cur_state_set > 0));

                    hyper_all_pre_rank = cell(1, hc_num);
                    hyper_all_cur_rank = cell(1, hc_num);
                    hyper_all_is_detected = cell(1, hc_num);
                    for hi = 1: hc_num
                        % obtain current hyper parameter config
                        hc = hyper_config{1, hi};
                        sm_num = numel(hc.sps_metric);
                        hyper_layer_idx_cell{1, hi}{1, wi} = [hyper_layer_idx_cell{1, hi}{1, wi}; li];
                        hyper_right_idx_cell{1, hi}{1, wi} = [hyper_right_idx_cell{1, hi}{1, wi}; i];
                        hyper_left_idx_cell{1, hi}{1, wi} = [hyper_left_idx_cell{1, hi}{1, wi}; j];
                        hyper_bug_value_cell{1, hi}{1, wi} = [hyper_bug_value_cell{1, hi}{1, wi}; cur_bug(1, 4)];
                        hyper_delta_state_cell{1, hi}{1, wi} = [hyper_delta_state_cell{1, hi}{1, wi}; delta_state];
                        hyper_safety_rate_cell{1, hi}{1, wi} = [hyper_safety_rate_cell{1, hi}{1, wi}; cur_safety_rate];
                        hyper_delta_safety_cell{1, hi}{1, wi} = [hyper_delta_safety_cell{1, hi}{1, wi}; cur_safety_rate - ori_safety_rate];

                        ori_sps_score = [];
                        cur_sps_score = [];
                        for mi = 1: numel(hc.sps_metric)
                            ori_sps_score = [ori_sps_score, ori_hyper_all_sps_score{1,hi}{1,mi}{1,li}(i,j)];
                            cur_sps_score = [cur_sps_score, cur_hyper_all_sps_score{1,hi}{1,mi}{1,li}(i,j)];
                        end
                        hyper_ori_sps_score_cell{1, hi}{1, wi} = [hyper_ori_sps_score_cell{1, hi}{1, wi}; ori_sps_score];
                        hyper_cur_sps_score_cell{1, hi}{1, wi} = [hyper_cur_sps_score_cell{1, hi}{1, wi}; cur_sps_score];
                        % obtain the rank of original weight
                        all_pre_rank = zeros(1, sm_num);
                        for mi = 1: sm_num
                            pre_rank = calWeightRank(ori_hyper_all_sps_score{1, hi}{1, mi}, li, i, j, this.roil_l, this.roil_r);
                            all_pre_rank(1, mi) = pre_rank;
                        end
                        hyper_all_pre_rank{1, hi} = all_pre_rank;
                        hyper_pre_rank_cell{1, hi}{1, wi} = [hyper_pre_rank_cell{1, hi}{1, wi}; all_pre_rank];

                        % obtain the rank of current weight
                        all_cur_rank = zeros(1, sm_num);
                        for mi = 1: sm_num
                            cur_rank = calWeightRank(cur_hyper_all_sps_score{1, hi}{1, mi}, li, i, j, this.roil_l, this.roil_r);
                            all_cur_rank(1, mi) = cur_rank;
                        end
                        hyper_all_cur_rank{1, hi} = all_cur_rank;
                        hyper_cur_rank_cell{1, hi}{1, wi} = [hyper_cur_rank_cell{1, hi}{1, wi}; all_cur_rank];

                        all_is_detected = zeros(1, sm_num);
                        for mi = 1: sm_num
                            if any(ismember(cur_hyper_all_sps_weight{1,hi}{1,mi}{1,li}(:,3:5), weight_candidate(wi, :), 'rows'))
                                all_is_detected(1, mi) = 1;
                            end
                        end
                        hyper_all_is_detected{1, hi} = all_is_detected;
                        hyper_is_detected_cell{1, hi}{1, wi} = [hyper_is_detected_cell{1, hi}{1, wi}; all_is_detected];
                    end

                    % update flag and bug_i
                    if delta_state >= behavior_change_rate * tsf_size
                        flag = 0;
                        % save FL info of current bug_mdl
                        FL_info_file = [bug_mdl, '_selflag_', num2str(this.sel_flag), '_size_', num2str(tsf_size), '_mode_', tsf_mode, '_wselmode_', wsel_mode, '_budget_', num2str(budget), '_bcr_', num2str(behavior_change_rate), '_FL_Info.mat'];
                        save(FL_info_file, 'cur_bug', 'cur_hyper_all_sps_weight', 'cur_hyper_all_sps_score', 'cur_state_set', 'delta_state', 'cur_safety_rate', 'cur_hyper_exe_spectrum', 'cur_diagInfo_suite', 'hyper_all_pre_rank', 'hyper_all_cur_rank', 'hyper_all_is_detected');
                    else
                        if bug_i == budget
                            flag = 0;
                            % save FL info of current bug_mdl
                            FL_info_file = [bug_mdl, '_selflag_', num2str(this.sel_flag), '_size_', num2str(tsf_size), '_mode_', tsf_mode, '_wselmode_', wsel_mode, '_budget_', num2str(budget), '_bcr_', num2str(behavior_change_rate), '_FL_Info.mat'];
                            save(FL_info_file, 'cur_bug', 'cur_hyper_all_sps_weight', 'cur_hyper_all_sps_score', 'cur_state_set', 'delta_state', 'cur_safety_rate', 'cur_hyper_exe_spectrum', 'cur_diagInfo_suite', 'hyper_all_pre_rank', 'hyper_all_cur_rank', 'hyper_all_is_detected');
                        end
                        bug_i = bug_i + 1;
                    end
                    delete(fullfile(['breach/Ext/ModelsData/', bug_mdl, '_breach.slx']));
                    delete(fullfile([bug_mdl, '.slx']));
                    delete(fullfile([bug_mdl, '.slx.r202*']));
                    delete(fullfile([bug_mdl, '_breach.slxc']));
                end
            end
            % merge FL_info into a separate table for each hyper_config
            for hi = 1: hc_num
                hc = hyper_config{1, hi};
                % record the weight bugs
                layer_idx_set = [];
                right_idx_set = [];
                left_idx_set = [];
                bug_value_set = [];
                % record delta state (failed -> passed; passed -> failed)
                delta_state_set = [];
                % record the number of passed cases
                safety_rate_set = [];
                % record the changes of safety rate
                delta_safety_set = [];
                % record the suspiciousness scores for comparison
                ori_sps_score_set = [];
                cur_sps_score_set = [];
                % record the ranks of the suspiciousness scores for comparison
                pre_rank_set = [];
                cur_rank_set = [];
                % is detectable or not
                is_detected_set = [];

                for wi = 1: candidate_num
                    layer_idx_set = [layer_idx_set; hyper_layer_idx_cell{1, hi}{1, wi}];
                    right_idx_set = [right_idx_set; hyper_right_idx_cell{1, hi}{1, wi}];
                    left_idx_set = [left_idx_set; hyper_left_idx_cell{1, hi}{1, wi}];
                    bug_value_set = [bug_value_set; hyper_bug_value_cell{1, hi}{1, wi}];
                    delta_state_set = [delta_state_set; hyper_delta_state_cell{1, hi}{1, wi}];
                    safety_rate_set = [safety_rate_set; hyper_safety_rate_cell{1, hi}{1, wi}];
                    delta_safety_set = [delta_safety_set; hyper_delta_safety_cell{1, hi}{1, wi}];
                    ori_sps_score_set = [ori_sps_score_set; hyper_ori_sps_score_cell{1, hi}{1, wi}];
                    cur_sps_score_set = [cur_sps_score_set; hyper_cur_sps_score_cell{1, hi}{1, wi}];
                    pre_rank_set = [pre_rank_set;  hyper_pre_rank_cell{1, hi}{1, wi}];
                    cur_rank_set = [cur_rank_set; hyper_cur_rank_cell{1, hi}{1, wi}];
                    is_detected_set = [is_detected_set; hyper_is_detected_cell{1, hi}{1, wi}];
                end
                % save result as a csv file
                varNames = {'layer_idx_set','right_idx_set','left_idx_set', 'bug_value_set', 'delta_state_set', 'safety_rate_set', 'delta_safety_set', 'ori_sps_score_set', 'cur_sps_score_set', 'pre_rank_set', 'cur_rank_set', 'is_detected_set'};
                result = table(layer_idx_set, right_idx_set, left_idx_set, bug_value_set, delta_state_set, safety_rate_set, delta_safety_set, ori_sps_score_set, cur_sps_score_set, pre_rank_set, cur_rank_set, is_detected_set, 'VariableNames',varNames);
                RQ1_result_name = [this.mdl, '_selflag_', num2str(this.sel_flag), '_isweighted_', num2str(hc.window), '_topk_', num2str(hc.topk), '_size_', num2str(tsf_size), '_mode_', tsf_mode, '_wselmode_', wsel_mode, '_bcr_', num2str(behavior_change_rate), '_RQ1_result.csv'];
                writetable(result, RQ1_result_name);
            end
        end

        function addOriHyperConfig(this, oriSimFolder, flFolder, tsf_size, hyper_config)
            % Given new hyper configs, addOriHyperConfig function returns the
            % fault localization information based on the simulation logs.
            %
            % Inputs:
            %   oriSimFolder: the folder which stores the simulation log of the original model
            %   flFolder: the folder to stores the FL info of the original model
            %   tsf_size: the size of the test suite for fault localization
            %   hyper_config: new combinations of hyper parameters for fault localization
            % Outputs:
            %   new RQ result files

            % the number of hyper configs
            hc_num = numel(hyper_config);
            %% obtain the simulation log list
            dirOutput = dir(fullfile(oriSimFolder,'*.mat'));
            fileNames = dirOutput(~contains({dirOutput.name}, '_M_'));
            str = fileNames.name;
            load([oriSimFolder, '/', str]);
            %% initialize hyper_exe_spectrum (one hyper_config_i, one exe_spectrum)
            hyper_exe_spectrum = cell(1, hc_num);
            for hi = 1:hc_num
                % initialize exe_spectrum
                exe_spectrum = this.initializeCell('exe_spectrum');
                hyper_exe_spectrum{1, hi} = exe_spectrum;
            end
            %% update
            for in_i = 1:tsf_size
                for hi = 1: hc_num
                    %% update ori_hyper_exe_spectrum
                    exe_spectrum = this.accordinglyAssign(hyper_exe_spectrum{1, hi}, ori_diagInfo_suite{1, in_i}, hyper_config{1, hi}, 0);
                    hyper_exe_spectrum{1, hi} = exe_spectrum;
                end
            end
            %% update hyper_all_sps_score and hyper_all_sps_weight
            % perform SBFL
            hyper_all_sps_weight = cell(1, hc_num);
            hyper_all_sps_score = cell(1, hc_num);
            for hi = 1: hc_num
                hc = hyper_config{1, hi};
                all_sps_weight = cell(1, numel(hc.sps_metric));
                all_sps_score = cell(1, numel(hc.sps_metric));
                for mi = 1: numel(hc.sps_metric)
                    [sps_weight_info, sps_score] = this.SBFL(hyper_exe_spectrum{1, hi}, hc, mi);
                    all_sps_weight{1, mi} = sps_weight_info;
                    all_sps_score{1, mi} = sps_score;
                end
                hyper_all_sps_weight{1, hi} = all_sps_weight;
                hyper_all_sps_score{1, hi} = all_sps_score;
            end
            ori_hyper_exe_spectrum = hyper_exe_spectrum;
            ori_hyper_all_sps_score = hyper_all_sps_score;
            ori_hyper_all_sps_weight = hyper_all_sps_weight;

            new_str = strrep(str, 'Sim_Log', 'FL_Info');
            final_str = fullfile(flFolder, new_str);
            save(final_str, 'hyper_config', 'ori_hyper_all_sps_score', 'ori_hyper_all_sps_weight', 'ori_hyper_exe_spectrum');
        end

        function addMutHyperConfig(this, oriFLFolder, curSimFolder, tsf_size, hyper_config)
            % Given new hyper configs, addMutHyperConfig function returns the
            % fault localization information based on the simulation logs.
            %
            % Inputs:
            %   oriFLFolder: the folder where the fault localization result of original model is located
            %   curSimFolder: the folder where the simulation logs of mutated model are located
            %   tsf_size: the size of the test suite for fault localization
            %   hyper_config: new combinations of hyper parameters for fault localization
            % Outputs:
            %   new RQ result files

            % the number of hyper configs
            hc_num = numel(hyper_config);
            %% obtain the FL info of original model
            oriDirOutput = dir(fullfile(oriFLFolder,'*.mat'));
            oriFileNames = oriDirOutput(~contains({oriDirOutput.name}, '_M_'));
            ori_str = oriFileNames.name;
            load(ori_str);
            %% obtain the simulation log list of mutants
            mutDirOutput = dir(fullfile(curSimFolder,'*_M_*.mat'));
            mutFileNames = {mutDirOutput.name};
            % traverse files
            for idx = 1:numel(mutFileNames)
                mut_str = mutFileNames{1, idx};
                load(mut_str);
                %% update cur_hyper_exe_spectrum
                % initialize hyper_exe_spectrum (one hyper_config_i, one exe_spectrum)
                hyper_exe_spectrum = cell(1, hc_num);
                for hi = 1:hc_num
                    % initialize exe_spectrum
                    exe_spectrum = this.initializeCell('exe_spectrum');
                    hyper_exe_spectrum{1, hi} = exe_spectrum;
                end
                %% update
                for in_i = 1:tsf_size
                    for hi = 1: hc_num
                        %% update cur_hyper_exe_spectrum
                        exe_spectrum = this.accordinglyAssign(hyper_exe_spectrum{1, hi}, cur_diagInfo_suite{1, in_i}, hyper_config{1, hi});
                        hyper_exe_spectrum{1, hi} = exe_spectrum;
                    end
                end
                %% update cur_hyper_all_sps_score and cur_hyper_all_sps_weight
                % perform SBFL
                hyper_all_sps_weight = cell(1, hc_num);
                hyper_all_sps_score = cell(1, hc_num);
                for hi = 1: hc_num
                    hc = hyper_config{1, hi};
                    all_sps_weight = cell(1, numel(hc.sps_metric));
                    all_sps_score = cell(1, numel(hc.sps_metric));
                    for mi = 1: numel(hc.sps_metric)
                        [sps_weight_info, sps_score] = this.SBFL(hyper_exe_spectrum{1, hi}, hc, mi);
                        all_sps_weight{1, mi} = sps_weight_info;
                        all_sps_score{1, mi} = sps_score;
                    end
                    hyper_all_sps_weight{1, hi} = all_sps_weight;
                    hyper_all_sps_score{1, hi} = all_sps_score;
                end
                cur_hyper_exe_spectrum = hyper_exe_spectrum;
                cur_hyper_all_sps_score = hyper_all_sps_score;
                cur_hyper_all_sps_weight = hyper_all_sps_weight;
                %% update hyper_all_pre_rank, hyper_all_cur_rank, hyper_all_is_detected
                pattern = 'M_(.*?)_spec';
                matches = regexp(mut_str, pattern, 'tokens');

                if ~isempty(matches)
                    weight_idx_str = matches{1}{1};
                end

                weight_idx = split(weight_idx_str, '_');

                li = str2num(weight_idx{1,1});
                i = str2num(weight_idx{2,1});
                j = str2num(weight_idx{3,1});

                hyper_all_pre_rank = cell(1, hc_num);
                hyper_all_cur_rank = cell(1, hc_num);
                hyper_all_is_detected = cell(1, hc_num);

                for hi = 1: hc_num
                    % obtain current hyper parameter config
                    hc = hyper_config{1, hi};
                    sm_num = numel(hc.sps_metric);
                    % obtain the rank of original weight
                    all_pre_rank = zeros(1, sm_num);
                    for mi = 1: sm_num
                        pre_rank = calWeightRank(ori_hyper_all_sps_score{1, hi}{1, mi}, li, i, j, this.roil_l, this.roil_r);
                        all_pre_rank(1, mi) = pre_rank;
                    end
                    hyper_all_pre_rank{1, hi} = all_pre_rank;
                    % obtain the rank of current weight
                    all_cur_rank = zeros(1, sm_num);
                    for mi = 1: sm_num
                        cur_rank = calWeightRank(cur_hyper_all_sps_score{1, hi}{1, mi}, li, i, j, this.roil_l, this.roil_r);
                        all_cur_rank(1, mi) = cur_rank;
                    end
                    hyper_all_cur_rank{1, hi} = all_cur_rank;

                    all_is_detected = zeros(1, sm_num);
                    for mi = 1: sm_num
                        if any(ismember(cur_hyper_all_sps_weight{1,hi}{1,mi}(:,3:5), [li, i, j], 'rows'))
                            all_is_detected(1, mi) = 1;
                        end
                    end
                    hyper_all_is_detected{1, hi} = all_is_detected;
                end

                new_mut_str = strrep(mut_str, 'Sim_Log', 'FL_Info');
                save(new_mut_str, 'cur_bug', 'hyper_config', 'cur_hyper_all_sps_score', 'cur_hyper_all_sps_weight', 'cur_hyper_exe_spectrum', ...
                    'hyper_all_pre_rank', 'hyper_all_cur_rank', 'hyper_all_is_detected');
            end
        end

        function suc_weight = randFL(this, simlogFolder, tsf_size, bcr, tops, rep_num)
            % Given a set of tops, randFL function randomly selects weights
            % as suspicious weights and repeats this process rep_num times.
            %
            % Inputs:
            %   simlogFolder: the folder where the simulation logs of mutated model are located
            %   tsf_size: the size of the test suite for fault localization
            %   bcr: behavior change rate
            %   tops: tops suspicious weights
            %   rep_num: the number of repetitions

            % check if a parallel pool exists
            pool = gcp('nocreate');
            % if a parallel pool exists, delete it
            if ~isempty(pool)
                delete(pool);
            end
            % create a new parallel pool with the desired configuration
            pool = parpool(this.core_num);

            weight_pool = [];
            for li = this.roil_l:this.roil_r
                [rows, cols] = size(this.weight{1, li});
                for i = 1:rows
                    for j = 1:cols
                        weight_pool = [weight_pool; li, i, j];
                    end
                end
            end
            weight_num = size(weight_pool, 1);

            weight_cell = cell(1, weight_num);

            % a set of weights that are mutated successfully
            suc_weight = [];

            dirOutput = dir(fullfile(simlogFolder, '/', '*_M_*Sim_Log.mat'));
            fileNames = {dirOutput.name};

            parfor wi = 1: weight_num
                li = weight_pool(wi, 1);
                i = weight_pool(wi, 2);
                j = weight_pool(wi, 3);
                % obtain the simulation log
                cur_weight_list = {};
                cur_weight_str = ['_M_', num2str(li), '_', num2str(i), '_', num2str(j), '_'];
                for idx = 1:numel(fileNames)
                    str = fileNames{1, idx};
                    if contains(str, cur_weight_str)
                        cur_weight_list{end+1} = str;
                    end
                end
                if numel(cur_weight_list) == 0 || numel(cur_weight_list) > 1
                    % error('Check your simlog file!');
                    continue;
                end
                simlog_str = fullfile(simlogFolder, '/', cur_weight_list{1,1});
                simlog = load(simlog_str);
                if simlog.delta_state < bcr * tsf_size || simlog.delta_state > (1 - bcr) * tsf_size
                    continue;
                end
                % randomly select
                suc_rep = [];
                for tops_i = 1: numel(tops)
                    for ri = 1: rep_num
                        weight_idx = randperm(weight_num, tops(1, tops_i));
                        rand_weight = weight_pool(weight_idx, :);
                        if ismember([li, i, j], rand_weight, 'rows')
                            suc_rep(end+1) = 1;
                        else
                            suc_rep(end+1) = 0;
                        end
                    end
                end

                % suc_weight = [suc_weight; li, i, j, suc_rep];
                weight_cell{1, wi} = [li, i, j, suc_rep];
                disp(wi);
            end
            % add current weight to suc_weight
            for wi = 1:weight_num
                if ~isempty(weight_cell{1, wi})
                    suc_weight = [suc_weight; weight_cell{1, wi}];
                end
            end
        end


        function [suc_weight] = arachneFL(this, simlogFolder, tsf_size, bcr, tops)
            % arachneFL function performs a naive fault localisation approach
            % based on the simulation logs. (derived from archne)
            %
            % Inputs:
            %   simlogFolder: the folder which stores the simulation logs of mutated model
            %   tsf_size: the size of the test suite for fault localization
            %   bcr: behavior change rate
            %   tops: the number of weights that should be responsible for the failure
            % Outputs:
            %   suc_weight: a set of weights that are mutated successfully,
            %   including their positions and the info about whether they
            %   are detected or not.

            % check if a parallel pool exists
            pool = gcp('nocreate');
            % if a parallel pool exists, delete it
            if ~isempty(pool)
                delete(pool);
            end
            % create a new parallel pool with the desired configuration
            pool = parpool(this.core_num);

            weight_pool = [];
            for li = this.roil_l:this.roil_r
                [rows, cols] = size(this.weight{1, li});
                for i = 1:rows
                    for j = 1:cols
                        weight_pool = [weight_pool; li, i, j];
                    end
                end
            end
            weight_num = size(weight_pool, 1);
            weight_cell = cell(1, weight_num);

            % a set of weights that are mutated successfully
            suc_weight = [];

            % obtain the simulation log list of mutants
            mutDirOutput = dir(fullfile(simlogFolder,'*_M_*.mat'));
            mutFileNames = {mutDirOutput.name};

            parfor wi = 1: weight_num
                disp(wi);
                % obtain the simulation log
                cur_weight_list = {};
                cur_weight_str = ['_M_', num2str(weight_pool(wi,1)), '_', num2str(weight_pool(wi,2)), '_', num2str(weight_pool(wi,3)), '_'];
                for idx = 1:numel(mutFileNames)
                    str = mutFileNames{1, idx};
                    if contains(str, cur_weight_str)
                        cur_weight_list{end+1} = str;
                    end
                end
                if numel(cur_weight_list) == 0 || numel(cur_weight_list) > 1
                    % error('Check your simlog file!');
                    continue;
                end
                simlog_str = fullfile(simlogFolder, '/', cur_weight_list{1,1});
                simlog = load(simlog_str);
                if simlog.delta_state < bcr * tsf_size || simlog.delta_state > (1 - bcr) * tsf_size
                    continue;
                end

                % initialize the passed_fi_sum and failed_fi_sum
                failed_fi_sum = this.initializeCell('weight2OutSnapShot');
                passed_fi_sum = this.initializeCell('weight2OutSnapShot');
                % update failed_fi_sum and passed_fi_sum
                for in_i = 1: tsf_size
                    % frame number
                    frame_num = numel(simlog.cur_diagInfo_suite{1, in_i}.weight2Out);
                    cur_fi_sum = this.initializeCell('weight2OutSnapShot');
                    % sum fi of each weights based on current system execution
                    for fi = 1:frame_num
                        for li = this.roil_l:this.roil_r
                            cur_fi_sum{1, li} = cur_fi_sum{1, li} + abs(simlog.cur_diagInfo_suite{1, in_i}.weight2Out{1, fi}{1,li});
                        end
                    end
                    for li = this.roil_l:this.roil_r
                        if simlog.cur_diagInfo_suite{1,in_i}.rob < 0
                            failed_fi_sum{1,li} = failed_fi_sum{1,li} + cur_fi_sum{1,li};
                        else
                            passed_fi_sum{1,li} = passed_fi_sum{1,li} + cur_fi_sum{1,li};
                        end
                    end
                end
                % FI of each weight
                FI = this.initializeCell('weight2OutSnapShot');
                for li = this.roil_l:this.roil_r
                    FI{1,li} = failed_fi_sum{1,li}./(1 + passed_fi_sum{1, li});
                end

                tops_res = zeros(1, numel(tops));

                for tops_i = 1:numel(tops)
                    rank = calWeightRank(FI, weight_pool(wi,1), weight_pool(wi,2), weight_pool(wi,3), this.roil_l, this.roil_r);
                    if tops(1, tops_i) >= rank
                        tops_res(1, tops_i) = 1;
                    end
                end

                weight_cell{1, wi} = [weight_pool(wi,:), tops_res];
            end

            % add current weight to suc_weight
            for wi = 1:weight_num
                if ~isempty(weight_cell{1, wi})
                    suc_weight = [suc_weight; weight_cell{1, wi}];
                end
            end
        end
    end
end
