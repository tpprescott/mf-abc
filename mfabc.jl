export MFABC, Particle, Cloud
export runpair, get_benchmark

##### Everything below assumes exactly two fidelities: 
# Future work will adapt methodology to a true multifidelity approach.

struct MFABC
    parameter_sampler::Function # Parameter sampler
    lofi::Function              # Map parameter to distance from observed data (low fidelity model) and coupling output
    hifi::Function              # Map parameter and coupling input to distance from observed data (high fidelity model)
end

mutable struct Particle
    k::Parameters                # Record the parameter values
    sim_flag::Tuple{Bool,Bool}   # Record whether lofi and hifi simulations
    dist::Tuple{Float64,Float64} # Record the distances (if simulated)
    cost::Tuple{Float64,Float64} # Record the costs (if simulated)
end
Cloud = Array{Particle, 1}

function runpair(mfabc::MFABC, i::Int64=1)
   
    k::Parameters = mfabc.parameter_sampler()
    (d_lo,pass),c_lo = @timed mfabc.lofi(k)
    (d_hi),c_hi = @timed mfabc.hifi(k,pass)
    
    return Particle(k, (true,true), (d_lo, d_hi), (c_lo, c_hi))
end

using Distributed
using DelimitedFiles
function get_benchmark(mfabc::MFABC, N::Int64=10, outfile::String="./trial_output.txt")::Cloud
    output = pmap(i->runpair(mfabc,i), 1:N)
#    writedlm(outfile,output)
    return output
end
