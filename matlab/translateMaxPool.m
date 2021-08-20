function [NNTLayer, issues] = translateMaxPool(node, LayerName, OpsetVersion)

%   Copyright 2018-2020 The MathWorks, Inc.

issues = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "auto_pad"          "STRING"    true    "NOTSET"
    "kernel_shape"      "INTS"      false   []
    "pads"              "INTS"      true    []
    "storage_order"     "INT"       true    []
    "strides"           "INTS"      true    []
    });
% Parse the attributes
[auto_pad,  kernel_shape, pads, storage_order, strides] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);

% kernel_shape
FilterSize = kernel_shape; %[h w] or [h w d]

% Determine whether the input is 2d or 3d
has3dImageInput = (numel(FilterSize) == 3);

% Handle padding
if ~isempty(pads)
    Padding = pads;
    Padding = iConvertONNXToMATLABPadding(Padding, has3dImageInput);
    if auto_pad~="NOTSET"
        issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
            message('nnet_cnn_onnx:onnx:AutoPadAndPadDefined'))]; 
    end
else
    switch auto_pad
        case 'SAME_UPPER'
            Padding = 'same';
        case 'SAME_LOWER'
            Padding = 'same';
            issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
                message('nnet_cnn_onnx:onnx:AutoPadSameLower', LayerName))];
        case 'VALID'
            Padding = getDefaultPadding(has3dImageInput);
        case 'NOTSET'
            % Pads is not explicitly set at this point so default is used
            Padding = getDefaultPadding(has3dImageInput);
        otherwise
            issues = [issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
                message('nnet_cnn_onnx:onnx:UnknownAttributeSetting', 'auto_pad'))];
    end
end

% storage_order
if storage_order==1
    issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message('nnet_cnn_onnx:onnx:MaxPoolStorageOrder'))];
    NNTLayer = [];
    return;
end

% strides
if isempty(strides)
    if has3dImageInput
        strides = [1 1 1];
    else
        strides = [1 1];        
    end
end

if OpsetVersion >= 8
    % storage_order
    if storage_order==1
        % column-major not supported
        issues = [issues nnet.internal.cnn.onnx.NodeTranslationError(node,...
            message('nnet_cnn_onnx:onnx:MaxPoolStorageOrder'))];
        NNTLayer = [];
        return;
    end
end

args = {FilterSize, 'Stride', strides, 'Padding', Padding, 'Name', LayerName};
if has3dImageInput
    [NNTLayer, constructionIssue] = nnet.internal.cnn.onnx.constructLayer('maxPooling3dLayer', LayerName, node, args{:});
else
    [NNTLayer, constructionIssue] = nnet.internal.cnn.onnx.constructLayer('maxPooling2dLayer', LayerName, node, args{:});
end
issues = [issues constructionIssue];
end

function Padding = iConvertONNXToMATLABPadding(Padding, has3dImageInput)
    if has3dImageInput
        % ONNX:   [H_b,W_b,D_b,H_end,W_end,D_end] ==> [t l f b r k]
        % MATLAB: [t l f; b r k]
        Padding = Padding([1 3 5; 2 4 6]);
    else
        if length(Padding) == 2  % HAck for 1d padding
            warning('Using my Custom padding to load 1d Padding layers')
            % ONNX:   [H_b,H_end,] ==> [l r]
            % MATLAB: [0 0 l r]
            Padding = [0, 0, Padding([1,2])];
        else
            % ONNX:   [H_b,W_b,H_end,W_end] ==> [t l b r]
            % MATLAB: [t b l r]
            Padding = Padding([1,3,2,4]);
        end
    end
end 

function Padding = getDefaultPadding(has3dImageInput)
    if has3dImageInput
        Padding = [0 0 0; 0 0 0]; % [t l f; b r k]
    else
        Padding = [0 0 0 0]; % [t b l r]
    end
end