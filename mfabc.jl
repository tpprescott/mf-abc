export MFABC, Particle, Cloud, BenchmarkParticle, BenchmarkCloud, MFABCParticle, MFABCCloud
export get_eta, phi

##### Everything below assumes exactly two fidelities: 
# Future work will adapt methodology to a true multifidelity approach.

struct MFABC
    parameter_sampler::Function # Parameter sampler
    lofi::Function              # Map parameter to distance from observed data (low fidelity model) and coupling output
    hifi::Function              # Map parameter and coupling input to distance from observed data (high fidelity model)
end

abstract type Particle end

struct BenchmarkParticle <: Particle
    k::Parameters           # Record the parameter values
    dist::Array{Float64,1}  # Record the two distances
    cost::Array{Float64,1}  # Record the two costs
end
struct MFABCParticle <: Particle
    k::Parameters   # Record the parameter values
    w::Float64      # Weighted (may be negative)
    c::Float64      # Total simulation cost
    cf::Bool        # Continuation Flag
end

Cloud{T} = Array{T, 1} where T<:Particle

######## Running simulations
function BenchmarkParticle(mfabc::MFABC, i::Int64=1)
   
    k::Parameters = mfabc.parameter_sampler()
    (d_lo,pass),c_lo = @timed mfabc.lofi(k)
    (d_hi),c_hi = @timed mfabc.hifi(k,pass)
    
    return BenchmarkParticle(k, [d_lo, d_hi], [c_lo, c_hi])
end

function MFABCParticle(mfabc::MFABC, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}, i::Int64=1)
    k::Parameters = mfabc.parameter_sampler()
    (d_lo,pass),c_lo = @timed mfabc.lofi(k)
    eta, w = (d_lo < epsilons[1]) ? (etas[1], 1) : (etas[2], 0)
    cf = (rand()<eta)
    if cf
        (d_hi), c_hi = @timed mfabc.hifi(k,pass)
        c = c_lo + c_hi
        close = ((d_lo, d_hi) .< epsilons)
        if xor(close...)
            w += (close[2]-close[1])/eta
        end
    else
        c = c_lo
    end
    return MFABCParticle(k, w, c, cf)
end

######## Converting benchmarks to MFABC (post-hoc)
function MFABCParticle(p::BenchmarkParticle, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64})
    close = (p.dist.<epsilons)
    eta, w = (close[1]) ? (etas[1], 1) : (etas[2], 0)
    cf = rand()<eta
    if cf
        c = sum(p.cost)
        if xor(close...)
            w += (close[2]-close[1])/eta
        end
    else
        c = p.cost[1]
    end
    return MFABCParticle(p.k, w, c, cf)
end

function MFABCCloud(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64})
    return map(p->MFABCParticle(p, epsilons, etas), s)
end

########## Take a benchmark, find the optimal eta (for a specified method) and get the MFABC cloud.

function sample_properties(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Real,Real}, Fweights::Array{Float64,1}=[1.0])
    
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
function sample_properties(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Real,Real}, parameterFun::Function)    
    F = [parameterFun(p.k) for p in s]
    Fbar = mean([parameterFun(p.k) for p in s if p.dist[2]<epsilons[2]])
    Fweights = (F .- Fbar).^2
    return sample_properties(s, epsilons, Fweights)
end
function sample_properties(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Real,Real}, cf::Array{Bool,1})
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s[cf], epsilons)
    rho_m = mean([p.dist[1]<epsilons[1] for p in s])
    rho_k = mean([p.dist[1]<epsilons[1] for p in s[cf]])

    ct = mean([p.cost[1] for p in s])
    c_p *= rho_m/rho_k
    c_n *= (1-rho_m)/(1-rho_k)
    p_tp *= rho_m/rho_k
    p_fp *= rho_m/rho_k
    p_fn *= (1-rho_m)/(1-rho_k)

    return p_tp, p_fp, p_fn, ct, c_p, c_n

end

function phi(eta, s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64,Float64})
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s, epsilons)
    return (p_tp - p_fp + (p_fp/eta[1]) + (p_fn/eta[2])) * (ct + c_p*eta[1] + c_n*eta[2])
end
function dphi!(storage, eta, s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64,Float64})
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s, epsilons)
    storage[1] = c_p * (p_tp - p_fp + (p_fn/eta[2])) - (p_fp/(eta[1]^2))*(ct + c_n*eta[2])
    storage[2] = c_n * (p_tp - p_fp + (p_fp/eta[1])) - (p_fn/(eta[2]^2))*(ct + c_p*eta[1])
end
function phi(eta, s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64,Float64}, parameterFun::Function)
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s,epsilons, parameterFun)
    return (p_tp - p_fp + (p_fp/eta[1]) + (p_fn/eta[2])) * (ct + c_p*eta[1] + c_n*eta[2])
end
function dphi!(storage, eta, s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64,Float64}, parameterFun::Function)
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s, epsilons, parameterFun)
    storage[1] = c_p * (p_tp - p_fp + (p_fn/eta[2])) - (p_fp/(eta[1]^2))*(ct + c_n*eta[2])
    storage[2] = c_n * (p_tp - p_fp + (p_fp/eta[1])) - (p_fn/(eta[2]^2))*(ct + c_p*eta[1])
end
function phi(eta, s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64,Float64}, cf::Array{Bool,1})
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s,epsilons,cf)
    return (p_tp - p_fp + (p_fp/eta[1]) + (p_fn/eta[2])) * (ct + c_p*eta[1] + c_n*eta[2])
end
function dphi!(storage, eta, s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64,Float64}, cf::Array{Bool,1})
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s,epsilons,cf)
    storage[1] = c_p * (p_tp - p_fp + (p_fn/eta[2])) - (p_fp/(eta[1]^2))*(ct + c_n*eta[2])
    storage[2] = c_n * (p_tp - p_fp + (p_fp/eta[1])) - (p_fn/(eta[2]^2))*(ct + c_p*eta[1])
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
function get_eta(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64,Float64}; method::String="mf")
    # Keyword "method" can be chosen from "mf" (multifidelity), "er" (early rejection), "ed" (early decision)
    # Any other keyword will give the ABC approach with no early rejection or acceptance at all.
    
    f(x) = phi(x,s,epsilons)
    g!(st,x) = dphi!(st,x,s,epsilons)

    eta = get_eta(f, g!, method=method)
    return eta, f(eta)
end
function get_eta(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64,Float64}, parameterFun::Function; method::String="mf")
    # Keyword "method" can be chosen from "mf" (multifidelity), "er" (early rejection), "ed" (early decision)
    # Any other keyword will give the ABC approach with no early rejection or acceptance at all.
    
    f(x) = phi(x,s,epsilons,parameterFun)
    g!(st,x) = dphi!(st,x,s,epsilons,parameterFun)

    eta = get_eta(f, g!, method=method)
    return eta, f(eta)
end
function get_eta(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64,Float64}, cf::Array{Bool,1}; method::String="mf")
    f(x) = phi(x,s,epsilons,cf)
    g!(st,x) = dphi!(st,x,s,epsilons,cf)

    eta = get_eta(f, g!, method=method)
    return eta, f(eta)
end


function MFABCCloud(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64, Float64}; method::String="mf")
    etas, phi = get_eta(s, epsilons, method=method)
    return MFABCCloud(s, epsilons, etas)
end


function MFABCCloud(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64, Float64}, burnsize::Int64; method::String="mf")
    if burnsize>length(s)
        error("Not enough sample points for specified burn-in")
    end

    out = Cloud{MFABCParticle}()
    cf = Array{Bool,1}()

    etas = (1.0, 1.0)
    for (n,p) in enumerate(s)
        mfp = MFABCParticle(p, epsilons, etas)
        push!(out, mfp)
        push!(cf, mfp.cf)
        if n>burnsize
            etas = get_eta(s[1:n], epsilons, cf, method=method)[1]
        end
    end

    return out, etas
  
end


######## Creating clouds

using DelimitedFiles
function write_cloud(cld::Cloud{T}, outdir::String="./output/") where {T}
    mkpath(outdir)
    for fn in fieldnames(T)
        cloud_field = [getfield(particle, fn) for particle in cld]
        writedlm(outdir*string(fn)*".txt", cloud_field)
    end
end

using Distributed
function BenchmarkCloud(mfabc::MFABC, N::Int64=10, outdir::String="./output/")::Cloud{BenchmarkParticle}
    if nworkers()>1
        output = pmap(i->BenchmarkParticle(mfabc,i), 1:N)
    else
        output = map(i->BenchmarkParticle(mfabc,i), 1:N)
    end
    write_cloud(output, outdir)
    return output
end

function BenchmarkCloud(indir::String="./output/")
    fn_list = fieldnames(BenchmarkParticle)
    input = map(fn->readdlm(indir*string(fn)*".txt"), fn_list)
    cld = Cloud{BenchmarkParticle}()
    for i in 1:size(input[1],1)
        raw_entries = [fields[i,:] for fields in input]
        push!(cld, BenchmarkParticle(Parameters(raw_entries[1]),raw_entries[2],raw_entries[3]))
    end
    return cld
end

function MFABCCloud(mfabc::MFABC, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}, N::Int64=10)::Cloud{MFABCParticle}
    if nworkers()>1
        return pmap(i->MFABCParticle(mfabc, epsilons, etas, i), 1:N)
    else
        return map(i->MFABCParticle(mfabc, epsilons, etas, i), 1:N)
    end
end

function MFABCCloud(mfabc::MFABC, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}, budget::Float64)::Cloud{MFABCParticle}
    running_cost = 0.0
    cloud = Cloud{MFABCParticle}()
    while running_cost < budget
        append!(cloud, [MFABCParticle(mfabc, epsilons, etas)])
        running_cost += cloud[end].c
    end
    return cloud        
end