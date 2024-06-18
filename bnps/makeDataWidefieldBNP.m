function [Data, Im] = makeDataWidefield(Freq, IntegTime, Step, Width, Offset, numFrames, Path)
% makeData_Widefield takes photon arrival times and generates the format used by BNP code.
% for use in BNP2-FLIM by MohamadFazel on GitHub 
%
% INPUT:
%   Freq: laser frequency in MHz
%   IntegTime: integration time (time for a measurements at a given gate step) in ms
%   Step: gate step size between successive integration times in ns
%   Width: width of a single gate in ns
%   Offset: offset from pulse signal to open first gate in ps
%   NRows: number of rows in the image data frame
%   NCols: number of columns in the image data frame
%   NFrames: total number of full gate sequences taken (NOT image frames)
%   Path: Name of file to access data from, string ending in .tiff
%
% OUTPUT:
%   Data: Data in a format used by BNP, which is a structure array where
%         every element corresponds to a pixel. The fields are:
%      Dt: photon arrival times (with respect to start of pulse) 
%          associated to the pixel (ns).    
%      W_Success: Number of detected photons for a particular pixel
%      W_Trial: Number of total pulses, empty and non-empty, for a
%                particular pixel
%   Im: Image frame of the data where pixel values are the number of
%       detected photons for pixels
%

% Read TIFF and initialize shape variables
info = imfinfo(Path);
numImages = numel(info);
numRows = info(1).Height;
numCols = info(1).Width;
imageData = cell(1, numImages);
for i = 1:numImages
    imageData{k} = imread(Path,'Index', k);
end

% Initialize output structures
Data(numRows, numCols).Dt = [];
Data(numRows, numCols).W = [];
Im = zeros(NLine, NColumn);
   
% Iterate over each image within TIFF movie
for image = 1:numImages
    pulseImage = mod(image, numImages/numFrames);
    delay = 1000*Offset+ pulseImage*Step - (Width/2);
    for i = 1:numRows
        for j = 1:numCols
            for obs = 1:imageData{image}(i, j)
                Data(i,j).Dt = cat(1, Data(i,j).Dt, delay);
            end
        end
    end
end

% Assign "confocal" centers for each pixel and fill W field
for i = 1:numRows
    for j = 1:numCols
        Data(i,j).X_Confocal = j - 0.5;
        Data(i,j).Y_Confocal = i - 0.5;
        Data(i,j).Z_Confocal = 0;

        Data(i,j).W_Success = length(Data(i,j).Dt);
        Data(i,j).W_Trial = 1000*IntegTime*Freq*numImages;
    end   
end

end
