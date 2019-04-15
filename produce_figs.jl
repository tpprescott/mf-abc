using DelimitedFiles
using StatsBase

struct Sample
    ksample::Array{Float64,2}
    dtilde::Array{Float64,1}
    d::Array{Float64,1}
    ctilde::Array{Float64,1}
    c::Array{Float64,1}

    function Sample(s::Array{Float64,2})
        new(s[:,1:end-4], s[:,end-3], s[:,end-2], s[:,end-1], s[:,end])
    end
end

function sample_properties(s::Sample,epsilons::Tuple{Float64,Float64})
    n = length(s.d)
    
    Ot_idx = (s.dtilde.<epsilons[1])
    O_idx = (s.d.<epsilons[2])

    p_tp = count(Ot_idx .& O_idx)/n
    p_fp = count(Ot_idx .& .~O_idx)/n
    p_fn = count(.~Ot_idx .& O_idx)/n
    ct = mean(s.ctilde)
    c_p = mean(s.c[Ot_idx]) * count(Ot_idx)/n
    c_n = mean(s.c[.~Ot_idx]) * count(.~Ot_idx)/n

    return p_tp, p_fp, p_fn, ct, c_p, c_n
end
function sample_properties(s::Sample,epsilons::Tuple{Float64,Float64},parameterFun::Function)
    n = length(s.d)
    
    Ot_idx = (s.dtilde.<epsilons[1])
    O_idx = (s.d.<epsilons[2])

    F = [parameterFun(s.ksample[j,:]) for j in 1:n]
    Fweight = (F .- mean(F[O_idx])).^2

    p_tp = mean(Fweight .* (Ot_idx .& O_idx))
    p_fp = mean(Fweight .* (Ot_idx .& .~O_idx))
    p_fn = mean(Fweight .* (.~Ot_idx .& O_idx))
    ct = mean(s.ctilde)
    c_p = mean(s.c[Ot_idx]) * count(Ot_idx)/n
    c_n = mean(s.c[.~Ot_idx]) * count(.~Ot_idx)/n

    return p_tp, p_fp, p_fn, ct, c_p, c_n
end

function phi(eta, s::Sample, epsilons::Tuple{Float64,Float64})
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s,epsilons)
    return (p_tp - p_fp + (p_fp/eta[1]) + (p_fn/eta[2])) * (ct + c_p*eta[1] + c_n*eta[2])
end
function dphi!(storage, eta, s::Sample, epsilons::Tuple{Float64,Float64})
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s,epsilons)
    storage[1] = c_p * (p_tp - p_fp + (p_fn/eta[2])) - (p_fp/(eta[1]^2))*(ct + c_n*eta[2])
    storage[2] = c_n * (p_tp - p_fp + (p_fp/eta[1])) - (p_fn/(eta[2]^2))*(ct + c_p*eta[1])
end

function phi(eta, s::Sample, epsilons::Tuple{Float64,Float64}, parameterFun::Function)
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s,epsilons, parameterFun)
    return (p_tp - p_fp + (p_fp/eta[1]) + (p_fn/eta[2])) * (ct + c_p*eta[1] + c_n*eta[2])
end
function dphi!(storage, eta, s::Sample, epsilons::Tuple{Float64,Float64}, parameterFun::Function)
    p_tp, p_fp, p_fn, ct, c_p, c_n = sample_properties(s, epsilons, parameterFun)
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
function get_eta(s::Sample, epsilons::Tuple{Float64,Float64}; method::String="mf")
    # Keyword "method" can be chosen from "mf" (multifidelity), "er" (early rejection), "ed" (early decision)
    # Any other keyword will give the ABC approach with no early rejection or acceptance at all.
    
    f(x) = phi(x,s,epsilons)
    g!(st,x) = dphi!(st,x,s,epsilons)

    eta = get_eta(f, g!, method=method)
    return eta, f(eta)
end
function get_eta(s::Sample, epsilons::Tuple{Float64,Float64}, parameterFun::Function; method::String="mf")
    # Keyword "method" can be chosen from "mf" (multifidelity), "er" (early rejection), "ed" (early decision)
    # Any other keyword will give the ABC approach with no early rejection or acceptance at all.
    
    f(x) = phi(x,s,epsilons,parameterFun)
    g!(st,x) = dphi!(st,x,s,epsilons,parameterFun)

    eta = get_eta(f, g!, method=method)
    return eta, f(eta)
end

function posthoc(s::Sample, epsilons::Tuple{Float64,Float64}, etas::Tuple{Float64, Float64})
    Ot = s.dtilde .< epsilons[1]
    O  = s.d .< epsilons[2]

    eta = (etas[1].*Ot) .+ (etas[2].*(.~Ot))
    cont_flags = rand(length(Ot)).<eta

    c_used = s.c .* cont_flags
    kweights = Ot + (cont_flags.*(O - Ot))./eta
    return kweights, c_used
end

function ESS(kweights::Array{Float64,1})
    return sum(kweights)^2/sum(kweights.^2)
end
function ESS_vs_times(kweights::Array{Float64,1}, c_used::Array{Float64,1}, ctilde::Array{Float64,1})
    return ESS(kweights), sum(ctilde)+sum(c_used)
end
function ESS_vs_times(s::Sample, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}=(1.0,1.0))
    kweights, c_used = posthoc(s, epsilons, etas)
    return ESS_vs_times(kweights, c_used, s.ctilde)
end
function efficiency(a,b,c)
    ESS_vs_times(a,b,c)[1]/ESS_vs_times(a,b,c)[2]
end

function estimate_mu(parameterFun::Function, kweights::Array{Float64,1}, ksample::Array{Float64,2})
    F = [parameterFun(ksample[j,:]) for j in 1:size(ksample,1)]
    return sum(kweights.*F)/sum(kweights)
end
function estimate_mu(parameterFun::Function, kweights::Array{Float64,1},s::Sample)
    return estimate_mu(parameterFun, kweights, s.ksample)
end
function estimate_mu(parameterFun::Function, s::Sample, epsilons::Tuple{Float64,Float64}, etas::Tuple{Float64, Float64}=(1.0,1.0))
    kweights, c_used = posthoc(s, epsilons, etas)
    return estimate_mu(parameterFun, kweights, s)
end

####################################

simset = readdlm("/scratch/prescott/output_190412_nc3_epsilon1e-2.txt")

# Make the benchmark
size_bm = 10^4
bm = Sample(simset[1:size_bm,:])
remaining_idx = size_bm+1:size(simset,1)

# Set the values of epsilon to use
epsilons = (40.0, 40.0)

# Get eta_1, eta_2 (continuation probabilities) optimized for ESS
eta_mf, phi_mf = get_eta(bm, epsilons, method="mf")
eta_er, phi_er = get_eta(bm, epsilons, method="er")
eta_ed, phi_ed = get_eta(bm, epsilons, method="ed")
eta_abc, phi_abc = get_eta(bm, epsilons, method="abc")

# Get some other continuation probabilities
eta_pp = 0.5.*eta_mf .+ 0.5.*(1,1)
    phi_pp = phi(eta_pp,bm,epsilons)
eta_pm = 0.5.*eta_mf .+ 0.5.*(1,0)
    phi_pm = phi(eta_pm,bm,epsilons)
eta_mp = 0.5.*eta_mf .+ 0.5.*(0,1)
    phi_mp = phi(eta_mp,bm,epsilons)
eta_mm = 0.5.*eta_mf .+ 0.5.*(0,0)
    phi_mm = phi(eta_mm,bm,epsilons)

# Get eta_1, eta_2 (continuation probabilities) optimized for specified functions
F = [k->Float64(1.9 < k[2] < 2.1),
     k->Float64(1.2 < k[2] < 1.4),
     k->k[2]]

eta_mf_Fi, phi_mf_Fi = zip(map(Fi->get_eta(bm, epsilons, Fi, method="mf"), F)...)
eta_er_Fi, phi_er_Fi = zip(map(Fi->get_eta(bm, epsilons, Fi, method="er"), F)...)
eta_ed_Fi, phi_ed_Fi = zip(map(Fi->get_eta(bm, epsilons, Fi, method="ed"), F)...)

# Split the remainder of the simulations into independent samples of fixed (common) size
size_s = 10^3
sset = [Sample(simset[idx,:]) for idx in Iterators.partition(remaining_idx,size_s)]

using Plots, StatsPlots, KernelDensity
function plot_efficiencies(bm::Sample, sset::Array{Sample,1}, epsilons::Tuple{Float64, Float64})
    eta_mf, phi_mf = get_eta(bm, epsilons, method="mf")
    eta_er, phi_er = get_eta(bm, epsilons, method="er")
    eta_ed, phi_ed = get_eta(bm, epsilons, method="ed")
    eta_abc, phi_abc = get_eta(bm, epsilons, method="abc")
    
    fig = plot()
    
    for etas in [eta_mf, eta_er, eta_ed, eta_abc]
        plot!(kde(map(s->efficiency(s,epsilons,etas), sset)))
    end
    return fig
end