export MFABC, Particle, Cloud, BenchmarkCloud, MFABCParticle, MFABCCloud
export get_eta, phi, MakeMFABCCloud, MakeBenchmarkCloud

##### Everything below assumes exactly two fidelities: 
# Future work will adapt methodology to a true multifidelity approach.

struct MFABC
    parameter_sampler::Function # Parameter sampler
    lofi::Function              # Map parameter to distance from observed data (low fidelity model) and coupling output
    hifi::Function              # Map parameter and coupling input to distance from observed data (high fidelity model)
end

struct Particle{N}
    k::Parameters
    dist::NTuple{N,Float64}
    cost::NTuple{N,Float64}
end
struct MFABCParticle
    p::Particle
    eta::Float64
    w::Float64
end

BenchmarkCloud = Array{Particle{2}, 1}
MFABCCloud = Array{MFABCParticle, 1}
Cloud = Union{BenchmarkCloud, MFABCCloud}

######## Running simulations
function BenchmarkParticle(mfabc::MFABC, i::Int64=1)
   
    k::Parameters = mfabc.parameter_sampler()
    (d_lo,pass),c_lo = @timed mfabc.lofi(k)
    (d_hi),c_hi = @timed mfabc.hifi(k,pass)
    
    return Particle(k, (d_lo, d_hi), (c_lo, c_hi))
end

function MFABCParticle(mfabc::MFABC, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}, i::Int64=1)
    k::Parameters = mfabc.parameter_sampler()
    (d_lo,pass),c_lo = @timed mfabc.lofi(k)
    eta, w = (d_lo < epsilons[1]) ? (etas[1], 1) : (etas[2], 0)
    
    if rand()<eta
        (d_hi), c_hi = @timed mfabc.hifi(k,pass)
        close = ((d_lo, d_hi) .< epsilons)
        if xor(close...)
            w += (close[2]-close[1])/eta
        end
        p = Particle(k, (d_lo,d_hi), (c_lo,c_hi))
    else
        p = Particle(k, (d_lo,), (c_lo,))
    end
    return MFABCParticle(p,eta,w)
end

########## Take a benchmark, find the optimal eta (for a specified method) and get the MFABC cloud.

function sample_properties(s::BenchmarkCloud, epsilons::Tuple{Real,Real}, Fweights::Array{Float64,1}=[1.0])
    
    O_lo = [p.dist[1]<epsilons[1] for p in s]
    O_hi = [p.dist[2]<epsilons[2] for p in s]

    p_tp = mean(Fweights .* (O_lo .& O_hi))
    p_fp = mean(Fweights .* (O_lo .& .~O_hi))
    p_fn = mean(Fweights .* (.~O_lo .& O_hi))
    ct = mean([p.cost[1] for p in s])
    c_p = mean([p.cost[2] for p in s if p.dist[1]<epsilons[1]]) * mean(O_lo)
    c_n = mean([p.cost[2] for p in s if p.dist[1]>=epsilons[1]]) * mean(.~O_lo)

    return p_tp, p_fp, p_fn, ct, c_p, c_n
end
function sample_properties(s::BenchmarkCloud, epsilons::Tuple{Real,Real}, parameterFun::Function)    
    F = [parameterFun(p.k) for p in s]
    Fbar = mean([parameterFun(p.k) for p in s if p.dist[2]<epsilons[2]])
    Fweights = (F .- Fbar).^2
    return sample_properties(s, epsilons, Fweights)
end
function sample_properties(s::MFABCCloud, epsilons::Tuple{Real,Real})
    
    s_bm = filter(x->isa(x, Particle{2}), [pp.p for pp in s])
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(BenchmarkCloud(s_bm), epsilons)

    rhom = mean([pp.p.dist[1]<epsilons[1] for pp in s])
    rhok = mean([p.dist[2]<epsilons[2] for p in s_bm])

    ct = mean([pp.p.cost[1] for pp in s])
    c_p *= rhom/rhok
    c_n *= (1-rhom)/(1-rhok)

    p_tp *= rhom/rhok
    p_fp *= rhom/rhok
    p_fn *= (1-rhom)/(1-rhok)

    return p_tp, p_fp, p_fn, ct, c_p, c_n
end

function phi(eta, p_tp::Float64, p_fp::Float64, p_fn::Float64, ct::Float64, c_p::Float64, c_n::Float64)
    return (p_tp - p_fp + (p_fp/eta[1]) + (p_fn/eta[2])) * (ct + c_p*eta[1] + c_n*eta[2])
end
function phi(eta, s::Cloud, epsilons::Tuple{Float64,Float64})
    return phi(eta, sample_properties(s, epsilons)...)
end
function phi(eta, s::BenchmarkCloud, epsilons::Tuple{Float64,Float64}, parameterFun::Function)
    return phi(eta, sample_properties(s, epsilons, parameterFun)...)
end


### The following version of get_eta uses optimisation (requiring the derivative of phi too)
# No longer called
#=
function dphi!(storage, eta, p_tp::Float64, p_fp::Float64, p_fn::Float64, ct::Float64, c_p::Float64, c_n::Float64)
    storage[1] = c_p * (p_tp - p_fp + (p_fn/eta[2])) - (p_fp/(eta[1]^2))*(ct + c_n*eta[2])
    storage[2] = c_n * (p_tp - p_fp + (p_fp/eta[1])) - (p_fn/(eta[2]^2))*(ct + c_p*eta[1])
end
function dphi!(storage, eta, s::BenchmarkCloud, epsilons::Tuple{Float64,Float64})
    dphi!(storage, eta, sample_properties(s, epsilons)...)
end
function dphi!(storage, eta, s::BenchmarkCloud, epsilons::Tuple{Float64,Float64}, parameterFun::Function)
    dphi!(storage,eta, sample_properties(s, epsilons, parameterFun)...)
end
function dphi!(storage, eta, s::BenchmarkCloud, epsilons::Tuple{Float64,Float64}, cf::Array{Bool,1})
    dphi!(storage,eta, sample_properties(s, epsilons, cf)...)
end
using Optim
function get_eta(f::Function, g!::Function; method::String="mf")
    if method=="mf"
        initial_x = [0.5, 0.5]
        od = OnceDifferentiable(f, g!, initial_x)
        lower = [0.0, 0.0]
        upper = [1.0, 1.0]
        inner_optimizer = GradientDescent()
        result = optimize(od, lower, upper, initial_x, Fminbox(inner_optimizer))
        return eta=(Optim.minimizer(result)[1], Optim.minimizer(result)[2])

    elseif method=="er"
        result = optimize(x2->f([1,x2]), 0.0, 1.0, Brent())
        return eta=(1.0, Optim.minimizer(result))

    elseif method=="ed"
        result = optimize(xx->f([xx,xx]), 0.0, 1.0, Brent())
        return eta=(Optim.minimizer(result), Optim.minimizer(result))

    else
        return eta=(1.0,1.0)
    end
end
=#

#### The rest of the get_eta are the ones used
function get_eta(p_tp::Float64, p_fp::Float64, p_fn::Float64, ct::Float64, c_p::Float64, c_n::Float64; method::String="mf")
    Rp = p_fp/(c_p/ct)
    Rn = p_fn/(c_n/ct)
    R0 = p_tp - p_fp
    if R0<=0
        return (1.0,1.0)
    end

    eta_1 = sqrt(Rp/R0)
    eta_2 = sqrt(Rn/R0)

    eta_1_bar = minimum([1.0, eta_1 / sqrt((1+p_fn/R0)/(1+c_n/ct))])
    eta_2_bar = minimum([1.0, eta_2 / sqrt((1+p_fp/R0)/(1+c_p/ct))])

    if method=="mf"
        if (eta_1<=1.0)&(eta_2<=1.0)
            return (eta_1,eta_2)
        else
            phi_1 = phi((eta_1_bar, 1.0), p_tp, p_fp, p_fn, ct, c_p, c_n)
            phi_2 = phi((1.0, eta_2_bar), p_tp, p_fp, p_fn, ct, c_p, c_n)
            if phi_2 < phi_1
                return (1.0, eta_2_bar)
            else
                return (eta_1_bar, 1.0)
            end
        end
    elseif method=="er"
        return (1.0, eta_2_bar)
    elseif method=="ed"
        eta = sqrt((ct*(p_fp+p_fn))/(R0*(c_p+c_n)))
        return (eta, eta)
    else
        return (1.0, 1.0)
    end
end
function get_eta(s::Cloud, epsilons::Tuple{Float64,Float64}; method::String="mf")
    # Keyword "method" can be chosen from "mf" (multifidelity), "er" (early rejection), "ed" (early decision)
    # Any other keyword will give the ABC approach with no early rejection or acceptance at all.
    
    f(x) = phi(x,s,epsilons)
    eta = get_eta(sample_properties(s,epsilons)..., method=method)
    return eta, f(eta)
end
function get_eta(s::BenchmarkCloud, epsilons::Tuple{Float64,Float64}, parameterFun::Function; method::String="mf")
    # Keyword "method" can be chosen from "mf" (multifidelity), "er" (early rejection), "ed" (early decision)
    # Any other keyword will give the ABC approach with no early rejection or acceptance at all.
    
    f(x) = phi(x,s,epsilons,parameterFun)
    eta = get_eta(sample_properties(s,epsilons,parameterFun)..., method=method)
    return eta, f(eta)
end

######## Creating clouds

using DelimitedFiles
function write_cloud(cld::BenchmarkCloud, outdir::String="./output/") where {T}
    mkpath(outdir)
    for fn in fieldnames(Particle)
        cloud_field = [getfield(particle, fn) for particle in cld]
        writedlm(outdir*string(fn)*".txt", cloud_field)
    end
end

using Distributed
function MakeBenchmarkCloud(mfabc::MFABC, N::Int64=10, outdir::String="./output/")::BenchmarkCloud
# Simulate a Benchmark cloud for fixed continuation probabilities
    if nworkers()>1
        output = pmap(i->BenchmarkParticle(mfabc,i), 1:N)
    else
        output = map(i->BenchmarkParticle(mfabc,i), 1:N)
    end
    write_cloud(output, outdir)
    return output
end

function MakeBenchmarkCloud(indir::String="./output/")::BenchmarkCloud
# Read a previously simulated Benchmark Cloud
    fn_list = fieldnames(Particle{2})
    input = map(fn->readdlm(indir*string(fn)*".txt"), fn_list)
    cld = BenchmarkCloud()
    for i in 1:size(input[1],1)
        raw_entries = [fields[i,:] for fields in input]
        push!(cld, Particle{2}(Parameters(raw_entries[1]), NTuple{2,Float64}(raw_entries[2]), NTuple{2,Float64}(raw_entries[3])))
    end
    return cld
end

function MakeMFABCCloud(mfabc::MFABC, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}, N::Int64=10)::MFABCCloud
# Multifidelity ABC algorithm with known continuation probabilities (which can therefore be parallelised)
# N specifies the number of particles to simulate
    if nworkers()>1
        return pmap(i->MFABCParticle(mfabc, epsilons, etas, i), 1:N)
    else
        return map(i->MFABCParticle(mfabc, epsilons, etas, i), 1:N)
    end
end

function MakeMFABCCloud(mfabc::MFABC, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}, budget::Float64)::MFABCCloud
    # Multifidelity ABC algorithm with known continuation probabilities
    # budget specifies the point (in seconds) at which the algorithm ends
    # Could feasibly parallelise by keeping track of a parallel running cost for each worker in the cluster and ending the worker when that's exceeded
        running_cost = 0.0
        cloud = MFABCCloud()
        while running_cost < budget
            pp = MFABCParticle(mfabc, epsilons, etas)
            push!(cloud, pp)
            running_cost += sum(pp.p.cost)
        end
        return cloud
    end

# Still missing: MFABCCloud out of the model specification (MFABC type), starting with unknown continuation probabilities
# This approach is no longer parallelisable, as eta has to adapt.
# We could give each worker its own eta and adapt that?


###### Converting benchmarks to multifidelity (known and unknown etas)

function MFABCParticle(p::Particle{2}, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64})
    close = (p.dist.<epsilons)
    eta, w = (close[1]) ? (etas[1], 1) : (etas[2], 0)
    if rand()<eta
        if xor(close...)
            w += (close[2]-close[1])/eta
        end
        return MFABCParticle(p, eta, w)
    else
        q = Particle{1}(p.k, (p.dist[1],), (p.cost[1],))
        return MFABCParticle(q, eta, w)
    end
end

function MakeMFABCCloud(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64})::MFABCCloud
    return map(p->MFABCParticle(p, epsilons, etas), s)
end

function MakeMFABCCloud(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}; method::String="mf", kwargs...)
# Continuation probabilities are either inferred from the entire benchmark set (posthoc), or sequentially (after a burn-in period)

    kwargs = Dict(kwargs)
    if haskey(kwargs,:burnin)
        n_b = kwargs[:burnin]::Int64
        return MakeMFABCCloud_(s, epsilons, n_b, method=method)
    else
    # Convert a benchmark cloud into a MFABC cloud using the entire benchmark set's information to get an optimal eta
        etas, phi = get_eta(s, epsilons, method=method)
        return MakeMFABCCloud(s, epsilons, etas)
    end
    
end

function MakeMFABCCloud_(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}, burnsize::Int64; method::String="mf")
# Convert a benchmark cloud into a MFABC cloud sequentially, using only the preceding information to inform eta at each iteration
    if burnsize>length(s)
        error("Not enough sample points for specified burn-in")
    end
    
    out = MFABCCloud()
    etas = (1.0, 1.0)
    
    for (n,p) in enumerate(s)
        mfp = MFABCParticle(p, epsilons, etas)
        push!(out, mfp)
        if n>burnsize
            etas, phi = get_eta(out, epsilons, method=method)
        end
    end
    return out, etas
end