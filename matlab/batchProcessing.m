% inputs: [nInputs x nSamples]
% predictionMode: 
%   '1'    (exactly 1 slope) 
%   '2'    (exactly 2 slopes)
%   '3'    (exactly 3 slopes)
%   'dfn'  (let DecayFitNet predict how many slopes there are, between 1 and 3)
%   'ibic' (use IBIC with DecayFitNet(1), DecayFitNet(2), and DecayFitNet(3)
%           to determine best fitting model order)
%
% Georg Götz, October 2023, Aalto University

function [tPredictions, aPredictions, nPredictions, normVals, dbMSE] = batchProcessing(inputs, predictionMode, maxOrder, fs, filterFrequency, inputsAreEDFs, plotFits)
switch predictionMode
    case '1'
        assert(maxOrder==1, "maxOrder must be 1 if prediction mode is '1'");
        net = DecayFitNetToolbox(1, fs, filterFrequency);
    case '2'
        assert(maxOrder==2, "maxOrder must be 2 if prediction mode is '2'");
        net = DecayFitNetToolbox(2, fs, filterFrequency);
    case '3'
        assert(maxOrder==3, "maxOrder must be 3 if prediction mode is '3'");
        net = DecayFitNetToolbox(3, fs, filterFrequency);
    case 'dfn'
        assert(maxOrder==3, "maxOrder must be 3 if prediction mode is 'dfn'");
        net = DecayFitNetToolbox(0, fs, filterFrequency);
    case 'ibic'
        net = ibicBasedDecayFitNet(maxOrder, fs, filterFrequency);
    otherwise
        error('Prediction mode %s is not supported.', predictionMode);
end

% Initialize arrays
nInputs = size(inputs, 1);
tPredictions = zeros(nInputs, maxOrder);
aPredictions = zeros(nInputs, maxOrder);
nPredictions = zeros(nInputs, 1);
normVals = zeros(nInputs, 1);
mseVals = zeros(nInputs, 1);

timeAxis = linspace(0, (size(inputs, 2)-1)/fs, size(inputs, 2));
timeAxis_ds = linspace(0, (size(inputs, 2)-1)/fs, 100);

bb=0; % for progress bar
for mIdx=1:nInputs
    % Progress bar
    pstr = sprintf('Progress: Measurement %d / %d [%.02f %%%%]. \n', mIdx, nInputs, 100*mIdx/nInputs);
    fprintf([repmat('\b',[1 bb]),pstr]) %erase current line with backspaces
    bb = length(pstr)-1;
    
    % Estimate decay parameters
    thisOmniRIR = inputs(mIdx, :).';
    [tVals, aVals, nVal, normVal] = net.estimateParameters(thisOmniRIR, inputsAreEDFs);
    tPredictions(mIdx, 1:length(tVals)) = tVals;
    aPredictions(mIdx, 1:length(aVals)) = aVals;
    nPredictions(mIdx) = nVal;
    normVals(mIdx) = normVal;

    % Get model fit from parameters
    fittedEDF_norm = generateSyntheticEDCs(tVals, aVals, nVal, timeAxis).';
    fittedEDF_norm_db_ds = resample(10*log10(fittedEDF_norm), 100, length(fittedEDF_norm), 0, 5);
    fittedEDF_db_ds = fittedEDF_norm_db_ds + 10*log10(normVal);

    % Calculate MSEs
    edf_norm = rir2decay(thisOmniRIR, fs, filterFrequency, true, true, true); % doBackwardsInt, analyseFullRIR, normalize
    edf_norm_db_ds = resample(10*log10(edf_norm), 100, length(edf_norm), 0, 5);
    edf_db_ds = edf_norm_db_ds + 10*log10(normVal);
    mseVals(mIdx) = mseLoss(fittedEDF_norm_db_ds(1:95), edf_norm_db_ds(1:95));

    % Plot if desired
    if plotFits
        if mIdx==1
            yMax = ceil(log10(normVal)+1)*10;
            yMin = yMax - 130;

            figure;
            hold on;
            h_edf = plot(timeAxis_ds, edf_db_ds, 'LineWidth', 1.5);
            h_fittedEDF = plot(timeAxis_ds, fittedEDF_db_ds, 'LineWidth', 1.5);
            ylim([yMin, yMax]);
            yticks(yMin:20:yMax);
            legend('True EDF', 'Fitted EDF');
            xlabel('Time [in s]');
            ylabel('Energy [in dB]');
            set(gca, 'FontSize', 14', 'FontName', 'CMU Serif');
            grid on;
        else
            set(h_edf, 'YData', edf_db_ds);
            set(h_fittedEDF, 'YData', fittedEDF_db_ds);
        end
        title(sprintf('DecayFitNet fit for measurement %d (%s mode)', mIdx, predictionMode));
        drawnow;
    end
end

% Build up MSE struct: mean, median, 95 quant., all MSEs
mseQuantiles = quantile(mseVals, [0.5, 0.95]);
dbMSE.mseVals = mseVals;
dbMSE.avgMSE = mean(mseVals);
dbMSE.medianMSE = mseQuantiles(1);
dbMSE.q95MSE = mseQuantiles(2);

end