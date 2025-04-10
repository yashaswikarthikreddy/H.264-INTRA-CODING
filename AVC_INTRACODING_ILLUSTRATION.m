clc; clear; close all;

blockSize = 16;
subBlockSize = 4;
lambda = 0.85 * 2^((22 - 12)/3); % QP = 22
Qstep = 10;

img = imread('photo_954.jpg');
img = im2gray(img);
img = double(img);
[H, W] = size(img);

% Pad image
Hpad = ceil(H/blockSize)*blockSize;
Wpad = ceil(W/blockSize)*blockSize;
img = padarray(img, [Hpad-H, Wpad-W], 'replicate', 'post');

decisionMap = zeros(Hpad/blockSize, Wpad/blockSize); % 1 = use 4x4, 0 = use 16x16
modeMap = zeros(Hpad/blockSize, Wpad/blockSize, 16);   % Store selected modes (one per 4x4 sub-block)

%% Main Loop: Process each macroblock (16x16)
for i = 1:blockSize:Hpad
    for j = 1:blockSize:Wpad
        block = img(i:i+blockSize-1, j:j+blockSize-1);

        %% ----- Try 16×16 prediction -----
        if i == 1
            top = 128 * ones(1, blockSize);
        else
            top = img(i-1, j:j+blockSize-1);
        end
        if j == 1
            left = 128 * ones(blockSize, 1);
        else
            left = img(i:i+blockSize-1, j-1);
        end

        bestJ16 = Inf;
        bestMode16 = 0;
        for mode = 0:3
            pred = simple_intra(mode, top, left, blockSize);
            residual = block - pred;
            coeffs = dct2(residual);
            quant = round(coeffs / Qstep);
            R = 3 + 1.5 * nnz(quant);
            recon = pred + idct2(quant * Qstep);
            D = sum((block(:) - recon(:)).^2);
            J = D + lambda * R;
            if J < bestJ16
                bestJ16 = J;
                bestMode16 = mode;
            end
        end

        %% ----- Try 4×4 block-wise prediction -----
        bestJ4 = 0;
        localModeMap = zeros(4, 4); % Temporary storage for modes for the 16 sub-blocks
        for x = 0:3
            for y = 0:3
                bx = i + x * subBlockSize;
                by = j + y * subBlockSize;
                b = img(bx:bx+subBlockSize-1, by:by+subBlockSize-1);

                if bx == 1
                    t = 128 * ones(1, subBlockSize);
                else
                    t = img(bx-1, by:by+subBlockSize-1);
                end
                if by == 1
                    l = 128 * ones(subBlockSize, 1);
                else
                    l = img(bx:bx+subBlockSize-1, by-1);
                end

                bestJblock = Inf;
                bestMode = 0;
                % Try all 9 intra prediction modes (0 to 8)
                for m = 0:8  
                    pred = simple_intra(m, t, l, subBlockSize);
                    r = b - pred;
                    c = dct2(r);
                    q = round(c / Qstep);
                    R = 3 + 1.5 * nnz(q);
                    recon = pred + idct2(q * Qstep);
                    D = sum((b(:) - recon(:)).^2);
                    J = D + lambda * R;
                    if J < bestJblock
                        bestJblock = J;
                        bestMode = m;
                    end
                end
                bestJ4 = bestJ4 + bestJblock;
                localModeMap(x+1, y+1) = bestMode;
            end
        end

        %% Compare and store decision and mode(s)
        mbRow = (i-1)/blockSize + 1;
        mbCol = (j-1)/blockSize + 1;
        
        if bestJ4 < bestJ16
            decisionMap(mbRow, mbCol) = 1;
            modeMap(mbRow, mbCol, :) = reshape(localModeMap, 1, 1, []);
        else
            decisionMap(mbRow, mbCol) = 0;
            modeMap(mbRow, mbCol, :) = bestMode16 * ones(1, 1, 16);
        end
    end
end

%% Display block outlines over the original image
figure;
imshow(img / 255); hold on;
[macroRows, macroCols] = size(decisionMap);

for i = 1:macroRows
    for j = 1:macroCols
        x = (j - 1) * blockSize + 1;
        y = (i - 1) * blockSize + 1;
        
        if decisionMap(i, j) == 1
            % For 4x4 blocks, draw green rectangles for each sub-block
            for dx = 0:3
                for dy = 0:3
                    sx = x + dx * subBlockSize;
                    sy = y + dy * subBlockSize;
                    rectangle('Position', [sx, sy, subBlockSize, subBlockSize], ...
                              'EdgeColor', 'g', 'LineWidth', 0.1);
                end
            end
        else
            % For 16x16 block prediction, draw one blue rectangle
            rectangle('Position', [x, y, blockSize, blockSize], ...
                      'EdgeColor', 'b', 'LineWidth', 0.1);
        end
    end
end

%% Generate and display the color map of prediction modes
modeImage = zeros(Hpad, Wpad);
for i = 1:macroRows
    for j = 1:macroCols
        x = (j - 1) * blockSize;
        y = (i - 1) * blockSize;
        for dx = 0:3
            for dy = 0:3
                blockIdx = dx * 4 + dy + 1;
                val = modeMap(i, j, blockIdx);
                modeImage(y + dy*subBlockSize + 1 : y + dy*subBlockSize + subBlockSize, ...
                          x + dx*subBlockSize + 1 : x + dx*subBlockSize + subBlockSize) = val;
            end
        end
    end
end

figure;
imagesc(modeImage);
colormap(jet(9));   % Use 9 distinct colors for modes 0–8
colorbar;
caxis([0 8]);       % Fix color axis from 0 to 8
title('Intra Prediction Mode Map (Modes 0–8)');

%% Intra Prediction Function
function pred = simple_intra(mode, top, left, size)
    switch mode
        case 0 % Vertical
            pred = repmat(top, size, 1);

        case 1 % Horizontal
            pred = repmat(left, 1, size);

        case 2 % DC
            dc = round((sum(top) + sum(left)) / (2 * size));
            pred = dc * ones(size);

        case 3 % Diagonal Down-Left
            pred = zeros(size);
            ext = [top, top(end), top(end)];
            for i = 1:size
                for j = 1:size
                    idx = i + j - 1;
                    pred(i,j) = ext(min(idx, numel(ext)));
                end
            end

        case 4 % Diagonal Down-Right
            pred = zeros(size);
            ext = [left(end); flipud(left); left(1)];
            for i = 1:size
                for j = 1:size
                    idx = i - j + size;
                    idx = max(1, min(length(ext), idx));
                    pred(i,j) = ext(idx);
                end
            end

        case 5 % Vertical Right (approximate)
            pred = zeros(size);
            ext = [top, top(end), top(end)];
            for i = 1:size
                for j = 1:size
                    idx = i + floor(j/2);
                    pred(i,j) = ext(min(idx, numel(ext)));
                end
            end

        case 6 % Horizontal Down (approximate)
            pred = zeros(size);
            ext = [left; left(end); left(end)];
            for i = 1:size
                for j = 1:size
                    idx = j + floor(i/2);
                    pred(i,j) = ext(min(idx, numel(ext)));
                end
            end

        case 7 % Vertical Left (approximate)
            pred = zeros(size);
            ext = [top, top(end), top(end)];
            for i = 1:size
                for j = 1:size
                    idx = i - floor(j/2);
                    idx = max(1, idx);
                    pred(i,j) = ext(min(idx, numel(ext)));
                end
            end

        case 8 % Horizontal Up (approximate)
            pred = zeros(size);
            ext = [left; left(end); left(end)];
            for i = 1:size
                for j = 1:size
                    idx = j - floor(i/2);
                    idx = max(1, idx);
                    pred(i,j) = ext(min(idx, numel(ext)));
                end
            end

        otherwise
            pred = 128 * ones(size);
    end
end
