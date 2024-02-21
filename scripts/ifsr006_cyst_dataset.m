clear all, close all, clc
addpath('D:/Itamar/Field_II_ver_3_30_windows/')
field_init(0)

possible_r = [2, 3, 4, 8]/1000;
possible_c = 1420:10:1600;
possible_latpos = (-16:2:16)/1000;
possible_axpos = (40:2.5:70)/1000;
set_field('show_times', 0); % do not show calculation time 

n_simulations = 2;
for idx = 1:n_simulations
    tic
    fprintf('Simulation %.5d\n', idx)
    % Get random parameters
    r = possible_r(randi(numel(possible_r)));
    c = possible_c(randi(numel(possible_c)));
    lat_pos = possible_latpos(randi(numel(possible_latpos)));
    ax_pos = possible_axpos(randi(numel(possible_axpos)));
    
    % Simulation parameters
    fc  = 5.5e6;            % Transducer center frequency [Hz]
    fs  = 22e6;            % Sampling frequency [Hz]
    set_field('fs', fs);
    set_field('c', c);      % Speed of sound [m/s]
    
    % Transducer parameters
    height      = 7/1000;           % Height of element [m]
    width       = 0.24/1000;        % Width of element [m]
    kerf        = 0.06/1000;        % Distance between transducer elements [m]
    num_elem    = 128;              % Number of elements
    focus       = [0 0 10000]/1000; % Initial electronic focus
    angles      = [0];
    num_sub_x    = 1;
    num_sub_y    = 5;
    
    % Simulating phantom
    nscatters = 50000;
    [phantom_info] = phantom_lesion_cylindrical_cyst(idx, nscatters, lat_pos, ax_pos, r);
    phantom_amplitudes = phantom_info.phantom_amplitudes;
    phantom_positions = phantom_info.phantom_positions;

    % Define impulse response and excitation
    impulse_response=sin(2*pi*fc*(0:1/fs:2/fc));
    impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
    excitation=sin(2*pi*fc*(0:1/fs:2/fc));

    % Define Rx transducer and set impulse
    ThRx = xdc_linear_array (num_elem, width, height, kerf, num_sub_x, num_sub_y, focus);
    xdc_impulse (ThRx, impulse_response);
    xdc_focus(ThRx, 0, focus)
    rect = xdc_get(ThRx);
    ele_pos = unique(rect(24,:));

    % Creating scats
    ThTx = xdc_linear_array (num_elem, width, height, kerf, num_sub_x, num_sub_y, focus);
    xdc_impulse(ThTx, impulse_response);
    xdc_excitation(ThTx, excitation);
    xdc_focus(ThTx, 0, focus)
    [signal,time_zero]=calc_scat_multi(ThTx, ThRx, phantom_positions, phantom_amplitudes);

    xdc_free (ThTx)
    xdc_free (ThRx)

    % saving info
    savedir = 'D:\Itamar\datasets\fieldII\simulation\nair2020\raw\';
%     str_param = sprintf('simu%.5d_r_%.1f_c_%.1f_latpos_%.1f_axpos_%.1f', ...
%                         idx, r*1000, c,lat_pos*1000, ax_pos*1000);
    simu_counter = sprintf('/simu%.5d', idx);
    filename = [savedir 'dataset.h5'];
    
    h5create(filename, [simu_counter '/fc'], size(fc))
    h5create(filename, [simu_counter '/fs'], size(fs))
    h5create(filename, [simu_counter '/r'], size(r))
    h5create(filename, [simu_counter '/c'], size(c))
    h5create(filename, [simu_counter '/ele_pos'], size(ele_pos))
    h5create(filename, [simu_counter '/lat_pos'], size(lat_pos))
    h5create(filename, [simu_counter '/ax_pos'], size(ax_pos))
    h5create(filename, [simu_counter '/signal'], size(signal))
    h5create(filename, [simu_counter '/time_zero'], size(time_zero))

    %%%%%%%%%%%%%%%% write %%%%%%%%%%%%%%%%
    h5write(filename, [simu_counter '/fc'], fc)
    h5write(filename, [simu_counter '/fs'], fs)
    h5write(filename, [simu_counter '/r'], r)
    h5write(filename, [simu_counter '/c'], c)
    h5write(filename, [simu_counter '/ele_pos'], ele_pos)
    h5write(filename, [simu_counter '/lat_pos'], lat_pos)
    h5write(filename, [simu_counter '/ax_pos'], ax_pos)
    h5write(filename, [simu_counter '/signal'], signal)
    h5write(filename, [simu_counter '/time_zero'], time_zero)

    toc
%     return
end
field_end
fprintf('DONE\n')
