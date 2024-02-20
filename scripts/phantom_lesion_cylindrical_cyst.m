function [phantom_info] = phantom_lesion_cylindrical_cyst(idx, nscatters,lat_pos, ...
    ax_pos, r)
%   Detailed explanation goes here
    seed = idx;
    rng(seed)

    x_size = 40/1000;       % Width of phantom [m]
    y_size = 7/1000;        % Transverse width of phantom [m]
    z_size = 50/1000;       % Height of phantom [m]
    z_start = 30/1000;      % Start of phantom surface [m];

    % Create the general scatterers
    x = (rand (nscatters,1)-0.5)*x_size;
    y = (rand (nscatters,1)-0.5)*y_size;
    z = rand (nscatters,1)*z_size + z_start;

    % Generate the amplitudes with a Gaussian distribution
    amp=randn(nscatters,1);
    
    % Make the cyst and set the amplitudes to zero inside
    lesion_pos = [lat_pos, ax_pos];
    ntargets = size(lesion_pos, 1);
    for idx = 1:size(lesion_pos,1)
        lesion_xc = lesion_pos(idx,1);
        lesion_zc = lesion_pos(idx,2);
        assert(abs(lesion_xc)<=x_size/2)
        assert(lesion_zc>=z_start)
        assert(lesion_zc<=(z_start+z_size))
        is_inside = ( ((x-lesion_xc).^2 + (z-lesion_zc).^2) < r^2);
        amp = amp .* (1-is_inside);
    end
    positions=[x y z];

    phantom_info = struct( ...
    'phantom_ntargets', ntargets, ...
    'phantom_target_posX',lesion_pos(:,1), ...
    'phantom_target_posZ', lesion_pos(:,2), ...
    'phantom_target_radius', r, ...
    'phantom_x_size', x_size, ...
    'phantom_y_size', y_size, ...
    'phantom_z_size', z_size, ...
    'phantom_z_start', z_start, ...
    'phantom_amplitudes', amp, ...
    'phantom_positions', positions);
end



