struct MFABC
    parameter_sampler::Function # Parameter sampler
    lofi::Function              # Map parameter to draw (lofi model) in summary statistic space
    hifi::Function              # Map parameter to draw (hifi model) in summary statistic space (accepts output from lofi for coupling)
    distance::Function          # Distance in summary statistic space from synthetic data
end

function runpair(mfabc::MFABC, i::Int64=1; timed_flag::Bool=true)
    
    k = mfabc.parameter_sampler()

    if timed_flag
    # Simulate low-fidelity and complete high-fidelity
        (yc,pass),ctilde = @timed mfabc.lofi(k)
        (yf),cc = @timed mfabc.hifi(k,pass)
    else
        yc, pass = mfabc.lofi(k)
        yf = mfabc.hifi(k,pass)
    end
    
    # Find out distances from data
    dc = mfabc.distance(yc)
    df = mfabc.distance(yf)

    if timed_flag
        return (k..., dc, df, ctilde, cc)
    else
        return (k..., dc, df)
    end
end

using Distributed
using DelimitedFiles
function get_benchmark(mfabc::MFABC, N::Int64=10, outfile::String="./trial_output.txt")
    output = pmap(i->runpair(mfabc,i), 1:N)
    writedlm(outfile,output)
    return output
end