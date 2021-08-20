classdef utils
    methods(Static)

        function plot_waveform(waveform, sample_rate, fig_title, fig)
            if ~exist('fig_title', 'var')
                fig_title = '';
            end
            if ~exist('fig', 'var')
                f = figure();
            end
            if size(waveform, 2) > size(waveform,1)
                waveform = permute(waveform, [2,1]);
                warning('Waveform appears to be in format (channels, timesteps), permutting for plot')
            end
            
            time_axis = linspace(0, (size(waveform,1) - 1) / sample_rate, size(waveform,1));
            ax = plot(time_axis, waveform);
            ylabel('Time [s]');
            title(fig_title)
        end
    end
end
