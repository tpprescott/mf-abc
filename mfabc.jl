export MFABC, Particle, Cloud, BenchmarkCloud, MFABCParticle, MFABCCloud
export get_eta, phi, MakeMFABCCloud, MakeBenchmarkCloud, cost, length
export write_cloud

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

import Base.length
length(p::Particle) = length(p.cost)
length(pp::MFABCParticle) = length(pp.p)
length(c::Cloud, i::Integer) = length(filter(p->(length(p)==i), c))
cost(p::Particle) = sum(p.cost)
cost(p::Particle, i::Integer) = (i<=length(p) ? p.cost[i] : 0)
cost(pp::MFABCParticle) = cost(pp.p)
cost(pp::MFABCParticle, i) = cost(pp.p, i)
cost(c::Cloud) = sum(cost.(c))
cost(c::Cloud, i) = sum(cost.(c, i))

accept(p::Particle, epsilon::Float64) = (p.dist[2]<=epsilon)
accept_rate(c::BenchmarkCloud, epsilon::Float64) = count(p->accept(p,epsilon), c)/length(c)


######## Running simulations: benchmark particles ignore MFABC
function BenchmarkParticle(mfabc::MFABC, i::Int64=1)::Particle{2}
   
    k::Parameters = mfabc.parameter_sampler()
    (d_lo,pass),c_lo = @timed mfabc.lofi(k)
    (d_hi),c_hi = @timed mfabc.hifi(k,pass)
    
    return Particle{2}(k, (d_lo, d_hi), (c_lo, c_hi))
end

using Distributed
function MakeBenchmarkCloud(mfabc::MFABC, N::Int64)::BenchmarkCloud
# Simulate a Benchmark cloud for fixed continuation probabilities
    if nworkers()>1
        return pmap(i->BenchmarkParticle(mfabc,i), 1:N)
    else
        return map(i->BenchmarkParticle(mfabc,i), 1:N)
    end
end
function MakeBenchmarkCloud(mfabc::MFABC, N::Int64, fn::String)::BenchmarkCloud
    output = MakeBenchmarkCloud(mfabc, N)
    write_cloud(output, fn)
    return output
end

######## Running simulations: apply MFABC with known eta
# Apply either to:
# de novo simulations (mfabc::MFABC) or;
# previously completed simulations (p::Particle{2})

############
# PARTICLES
############

# De novo
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
    return MFABCParticle(p,eta,w), sum(p.cost)
end

# Convert
function MFABCParticle(p::Particle{2}, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}, i::Int64=1)
    close = (p.dist.<epsilons)
    etas==(1.0, 1.0) && (return MFABCParticle(p, 1.0, 1.0*close[2]), sum(p.cost))
    eta, w = (close[1]) ? (etas[1], 1.0) : (etas[2], 0.0)
    if rand()<eta
        if xor(close...)
            w += (close[2]-close[1])/eta
        end
        return MFABCParticle(p, eta, w), sum(p.cost)
    else
        q = Particle{1}(p.k, (p.dist[1],), (p.cost[1],))
        return MFABCParticle(q, eta, w), p.cost[1]
    end
end

############
# CLOUDS
############

## Known, fixed eta (no dependence on a function to be estimated, either)

# de novo
function MakeMFABCCloud(mfabc::MFABC, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}; kwargs...)::MFABCCloud
# Multifidelity ABC algorithm with known continuation probabilities (which can therefore be parallelised)
    kwargs = Dict(kwargs)
    if haskey(kwargs, :N)
        N = Integer(kwargs[:N]) # Specifies the number of particles to simulate
        if nworkers()>1
            return pmap(i->MFABCParticle(mfabc, epsilons, etas, i)[1], 1:N)
        else
            return map(i->MFABCParticle(mfabc, epsilons, etas, i)[1], 1:N)
        end
    elseif haskey(kwargs, :budget)
        b = Float64(kwargs[:budget])
        run_cost = 0.0
        cloud = MFABCCloud()
        while 0<1
            pp, c = MFABCParticle(mfabc, epsilons, etas)
            run_cost += c
            (run_cost < b) ? push!(cloud, pp) : return cloud 
        end
    else
        error("No size indication (N or budget)")
    end
end

# convert
function MakeMFABCCloud(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64})::MFABCCloud
    return map(p->MFABCParticle(p, epsilons, etas)[1], s) # First component is the particle (second is cost of particle)
end
function MakeMFABCCloud(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}, budget::Float64)::MFABCCloud
    function truncate(cld, b, args...)
        return cld[1:searchsortedlast(cumsum(cost.(cld, args...)), b)]
    end
    mf = MakeMFABCCloud(truncate(s,budget,1), epsilons, etas)
    return truncate(mf, budget)
end

## Unknown eta: need to take into account finding a good eta, potentially given a function to be estimated

# de novo: this has to create a burn-in cloud and then adapt eta afterwards
function MakeMFABCCloud(mfabc::MFABC, epsilons::Tuple{Float64, Float64}; burnin::Int64, method::String="mf", kwargs...)::MFABCCloud
    s = MakeBenchmarkCloud(mfabc, burnin)
    return MakeMFABCCloud(mfabc, s, epsilons; method=method, kwargs...)
end
function MakeMFABCCloud(mfabc::MFABC, s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}; method::String="mf", kwargs...)
# Here s is a calculated burn-in set of hi/lo-fi simulation pairs
    kd = Dict(kwargs)
    cloud = MakeMFABCCloud(s, epsilons, (1.0, 1.0))
    burnin=length(s)
    if haskey(kd, :N)
        NN = Integer(kd[:N])-burnin # Input specifies the total number of particles (takes precedence over budget), NN here calculates additional particles needed
        for n in 1:NN
            etas = get_eta(cloud, epsilons; method=method, kwargs...)[1]
            push!(cloud, MFABCParticle(mfabc, epsilons, etas)[1])
        end
    elseif haskey(kd, :budget)
        b = Float64(kd[:budget])
        run_cost = cost(s)
        while run_cost<b
            etas = get_eta(cloud, epsilons; method=method, kwargs...)[1]
            pp, c = MFABCParticle(mfabc, epsilons, etas)
            run_cost += c
            push!(cloud, pp) 
        end
    end
    return cloud
end

# convert (fixed eta or unknown eta: determined by specifying a burnin or otherwise)

function MakeMFABCCloud(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}; method::String="mf", kwargs...)
# Continuation probabilities are either inferred from the entire benchmark set (posthoc), or sequentially (after a burn-in period)
    kd = Dict(kwargs)
    if haskey(kd,:burnin)
        n_b = kd[:burnin]::Int64
        return MakeMFABCCloud_(s, epsilons, n_b; method=method, kwargs...)
    else
    # Convert a benchmark cloud into a MFABC cloud using the entire benchmark set's information to get an optimal eta
        etas = get_eta(s, epsilons; method=method, kwargs...)[1]
        return MakeMFABCCloud(s, epsilons, etas)
    end    
end
function MakeMFABCCloud_(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}, burnin_size::Int64; method::String="mf", kwargs...)
# Convert a benchmark cloud into a MFABC cloud sequentially, using only the preceding information to inform eta at each iteration
# This is a very artificial approach, as we are ignoring information from simulations already carried out
    out = MakeMFABCCloud(s[1:burnin_size], epsilons, (1.0,1.0))
    
    for n in (burnin_size+1):length(s)
        etas = get_eta(out, epsilons; method=method, kwargs...)[1]
        push!(out, MFABCParticle(s[n], epsilons, etas))
    end
    return out
end

########## MFABC Clouds relies on "GET_ETA", calculated as follows:

function sample_properties(s::MFABCCloud, epsilons::Tuple{Float64, Float64}; kwargs...)
    kd = Dict(kwargs)

    if haskey(kd, :F)
        fun = kd[:F]::Function
        F_i = map(pp->fun(pp.p.k), s)
        w_i = [pp.w for pp in s]
        Fbar = sum(F_i .* w_i)/sum(w_i)
        Fweights = (F_i .- Fbar).^2
    else
        Fweights = ones(length(s))
    end

    is_bm = [isa(pp.p, Particle{2}) for pp in s]
    s2 = s[is_bm]
    Fweights2 = Fweights[is_bm]

    rhom = mean([pp.p.dist[1]<epsilons[1] for pp in s])
    rhok = mean([pp.p.dist[1]<epsilons[1] for pp in s2])
    
    O_lo = [pp.p.dist[1]<epsilons[1] for pp in s2]
    O_hi = [pp.p.dist[2]<epsilons[2] for pp in s2]
    
    p_tp = (rhom/rhok) * mean(Fweights2.*(O_lo .& O_hi))
    p_fp = (rhom/rhok) * mean(Fweights2.*(O_lo .& .~O_hi))
    p_fn = ((1-rhom)/(1-rhok)) * mean(Fweights2.*(.~O_lo .& O_hi))
    ct = mean([pp.p.cost[1] for pp in s])
    c_p = (rhom/rhok) * mean([pp.p.cost[2] for pp in s2] .* O_lo)
    c_n = ((1-rhom)/(1-rhok)) * mean([pp.p.cost[2] for pp in s2] .* .~O_lo)

    return p_tp, p_fp, p_fn, ct, c_p, c_n
end

function sample_properties(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}; kwargs...)
    kd = Dict(kwargs)
    if haskey(kd, :F)
        fun = kd[:F]::Function
        F_i = map(p->fun(p.k), s)
        w_i = [p.dist[2]<epsilons[2] for p in s]
        Fbar = sum(F_i .* w_i)/sum(w_i)
        Fweights = (F_i .- Fbar).^2
    else
        Fweights = ones(length(s))
    end

    O_lo = [p.dist[1]<epsilons[1] for p in s]
    O_hi = [p.dist[2]<epsilons[2] for p in s]
    
    p_tp = mean(Fweights.*(O_lo .& O_hi))
    p_fp = mean(Fweights.*(O_lo .& .~O_hi))
    p_fn = mean(Fweights.*(.~O_lo .& O_hi))
    ct = mean([p.cost[1] for p in s])
    c_p = mean([p.cost[2] for p in s] .* O_lo)
    c_n = mean([p.cost[2] for p in s] .* .~O_lo)

    return p_tp, p_fp, p_fn, ct, c_p, c_n
end

function phi(eta, p_tp::Float64, p_fp::Float64, p_fn::Float64, ct::Float64, c_p::Float64, c_n::Float64)
    return (p_tp - p_fp + (p_fp/eta[1]) + (p_fn/eta[2])) * (ct + c_p*eta[1] + c_n*eta[2])
end
function phi(eta, s::Cloud, epsilons::Tuple{Float64, Float64}; kwargs...)
    return phi(eta, sample_properties(s, epsilons; kwargs...)...)
end

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
        eta = minimum([1.0, sqrt((ct*(p_fp+p_fn))/(R0*(c_p+c_n)))])
        return (eta, eta)
    else
        return (1.0, 1.0)
    end
end

function get_eta(s::Cloud, epsilons::Tuple{Float64, Float64}; method::String="mf", lower_eta::Float64=0.01, kwargs...)
    # Keyword "method" can be chosen from "mf" (multifidelity), "er" (early rejection), "ed" (early decision)
    # Any other keyword will give the ABC approach with no early rejection or acceptance at all.
    
    sp = sample_properties(s, epsilons; kwargs...)
    eta = get_eta(sp..., method=method)
    eta = max.(lower_eta, eta)
    return eta, phi(eta, sp...)
end

######## Read/write (benchmark) clouds

using DelimitedFiles

function write_cloud(c::Array, name::String)
    open(name*".bm", "w") do io
        writedlm(io, c)
    end
    return nothing
end
function write_cloud(c::Array{<:Particle,1}, name::String)
    mkpath(name)
    for fieldname in fieldnames(Particle)
        cloud_field = [getfield(p, fieldname) for p in c]
        write_cloud(cloud_field, name*"/"*string(fieldname))
    end
    return nothing
end
function write_cloud(c::MFABCCloud, name::String)
    for fieldname in fieldnames(MFABCParticle)
        cloud_field = [getfield(mfp, fieldname) for mfp in c]
        write_cloud(cloud_field, name*"/"*string(fieldname))
    end
    N = length.(c)
    write_cloud(N, name*"/p/N")
    return nothing
end

function MakeBenchmarkCloud(indir::String)::BenchmarkCloud
# Read a previously simulated Benchmark Cloud
    k = readdlm(indir*"/k.bm")
    dist = readdlm(indir*"/dist.bm")
    cost = readdlm(indir*"/cost.bm")
    n = size(k,1)
    
    bm = BenchmarkCloud(undef, n)
    for i in 1:n
        bm[i] = Particle{2}(Parameters(k[i,:]), NTuple{2,Float64}(dist[i,:]), NTuple{2,Float64}(cost[i,:]))
    end
    return bm
end

function MakeMFABCCloud(indir::String)::MFABCCloud
    N = Integer.(readdlm(indir*"/p/N.bm"))
    k = readdlm(indir*"/p/k.bm")
    dist = readdlm(indir*"/p/dist.bm")
    cost = readdlm(indir*"/p/cost.bm")
    eta = readdlm(indir*"/eta.bm")
    w = readdlm(indir*"/w.bm")

    n = size(k,1)
    mf = MFABCCloud(undef, n)
    for i in 1:n
        pp = Particle{N[i]}(Parameters(k[i,:]), NTuple{N[i],Float64}(dist[i,1:N[i]]), NTuple{N[i],Float64}(cost[i,1:N[i]]))
        mf[i] = MFABCParticle(pp, eta[i], w[i])
    end
    return mf
end