export MFABC, Particle, Cloud
export runpair, get_benchmark

##### Everything below assumes exactly two fidelities: 
# Future work will adapt methodology to a true multifidelity approach.

struct MFABC
    parameter_sampler::Function # Parameter sampler
    lofi::Function              # Map parameter to distance from observed data (low fidelity model) and coupling output
    hifi::Function              # Map parameter and coupling input to distance from observed data (high fidelity model)
end

struct Particle
    k::Parameters           # Record the parameter values
    sim_flag::Array{Bool,1} # Record whether lofi and hifi simulations
    dist::Array{Float64,1}  # Record the distances (if simulated)
    cost::Array{Float64,1}  # Record the costs (if simulated)
end
Cloud = Array{Particle, 1}

function runpair(mfabc::MFABC, i::Int64=1)
   
    k::Parameters = mfabc.parameter_sampler()
    (d_lo,pass),c_lo = @timed mfabc.lofi(k)
    (d_hi),c_hi = @timed mfabc.hifi(k,pass)
    
    return Particle(k, [true,true], [d_lo, d_hi], [c_lo, c_hi])
end

using DelimitedFiles
function write_cloud(cld::Cloud, outdir::String="./output/")
    mkpath(outdir)
    for fn in fieldnames(Particle)
        cloud_field = [getfield(particle, fn) for particle in cld]
        writedlm(outdir*string(fn)*".txt", cloud_field)
    end
end
function read_cloud(indir::String="./output/")::Cloud
    fn_list = fieldnames(Particle)
    input = map(fn->readdlm(indir*string(fn)*".txt"), fn_list)
    cld = Cloud()
    for i in 1:size(input[1],1)
        raw_entries = [fields[i,:] for fields in input]
        push!(cld, Particle(Parameters(raw_entries[1]), Bool.(raw_entries[2]), raw_entries[3], raw_entries[4]))
    end
    return cld
end


using Distributed
function get_benchmark(mfabc::MFABC, N::Int64=10, outdir::String="./output/")::Cloud
    if nworkers()>1
        output = pmap(i->runpair(mfabc,i), 1:N)
    else
        output = map(i->runpair(mfabc,i), 1:N)
    end
    write_cloud(output, outdir)
    return output
end