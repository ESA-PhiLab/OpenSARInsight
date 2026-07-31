function [rangeCompressedData, rangeRefFunctSamples] = range_compression(header, rawData)
%--------------------------------------------------------------------------
% Name: range_compression.m
% 
% Author: Juan M Cuerda (INTA)
% Date: 05/06/2025
% 
% Function: Perform range compression of Sentinel 1 decoded L0 data using
%   as range reference function a Nominal Image Replica generated from 
%   L0 header parameters as described in RD-1. Required input variables
%   have been generated using INTA's L0 reader function.
%
% [RD-1]: S1-TN-MDA-52-7445. Sentinel-1 Level 1 Detailed Algorithm Definition 
%
% Inputs: 
%   header:  cell matrix from SentinelHeader_IWX-Burst_[bust#]-[polarization].mat (.npi for Python version) file 
%   rawData: cell matrix from SentinelData_IWX-Burst_[bust#]-[polarization].mat (.npi for Python version) file 

% Outputs 
%   rangeCompressedData: nxm (azimuthLines x rangeSamples) [complex double] matrix with range compressed data
%   rangeRefFunctSamples: n [double]. Length of range reference function 
%       corresponding to partially focused samples at the end of range 
%       dimension. To be removed from the generated rangeCompressedData matrix
%--------------------------------------------------------------------------
% Read Burst Info
radarConfig = [header.RadarConfiguration];            % header structure
txpl        = unique([radarConfig.TxPulseLength]);    % Tx pulse length (s)
fs          = unique([radarConfig.SamplingFreq]);     % Sampling frequency (Hz)
txprr       = unique([radarConfig.TxPulseRampRate]);  % TX pulse ramp rate (Hz/s)
txpsf       = unique([radarConfig.TxPulseStartFreq]); % Tx pulse start freq (Hz)

rangeRefFunctSamples = round(txpl * fs); % (s)
samples = (0:rangeRefFunctSamples-1)-rangeRefFunctSamples/2; % vector of sample number
tn       = samples/fs; % range time for samples (s)

% Replica construction
phi1 = txpsf -txprr * (-txpl/2);
phi2 = txprr/2;

nomChirpImage = 1/rangeRefFunctSamples * exp(2i*pi * (phi1 * tn + phi2*tn.^2));

% Compression
aSize = size(rawData,1); 
rSize = size(rawData,2); 
nFFT = 2^nextpow2(rSize+rangeRefFunctSamples+1);
compressedRangeLine = zeros(aSize,nFFT);

for i = 1: aSize
    rangeLine = rawData(i,:);    
    rangeSpectrum = fft(rangeLine,nFFT);  
    chirpSpectrum = fft(nomChirpImage,nFFT);
    compressedSpectrum = rangeSpectrum.*conj(chirpSpectrum);
    compressedRangeLine(i,:) = ifft(compressedSpectrum);    
end

% Crop to original size
rangeCompressedData = compressedRangeLine(:,1:rSize); % Remove zero padding