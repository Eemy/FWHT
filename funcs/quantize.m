function mat_q = quantize(mat,maxVal,NB)
%Quantize input matrix 'mat' to NB values, return output matrix 'mat_q'.
    %Clip raw values
    if maxVal < 1
        maxVal = 1;
    end

    minVal = -maxVal;
    mat = min(max(mat,minVal),maxVal);

    %Convert to bit values
    int_bits = ceil(log(maxVal)/log(2));
    mat_q = round(mat*2^(NB-(int_bits+1))); %this may vary depending on max value..

    %Clip to actual bit limits
    mat_q = min(max(mat_q,-2^(NB-1)), 2^(NB-1)-1);
end