using Distributed

function _batch!(
    mm::AbstractArray{M},
    ww::AbstractArray{Float64},
    y_obs::Y,
    q::AbstractGenerator{M},
    w::AbstractWeight{M},
)::NamedTuple where M where Y

    save1 = rand!(mm, q)
    save2 = weight!(ww, w, mm, y_obs)
    return merge(save1, save2)
end

export rejection_sample, importance_sample

function rejection_sample(
    y_obs::Y,
    q::AbstractGenerator{M},
    w::AbstractWeight{M},
    N::Int64,
) where M where Y
    mm = Array{M}(undef, N)
    ww = Array{Float64}(undef, N)
    save = _batch!(mm, ww, y_obs, q, w)
    output = merge((mm=mm, ww=ww), save)
    return output
end

function importance_sample(
    y_obs::Y,
    prior::AbstractGenerator{M},
    proposal::AbstractGenerator{M},
    w::AbstractWeight{M},
    N::Int64,
) where M where Y
    mm = Array{M, 1}(undef, N)
    ww = Array{Float64, 1}(undef, N)
    save = _batch!(mm, ww, y_obs, proposal, w)

    logpp = :logpp in keys(save) ? save[:logpp] : logpdf(prior, mm)
    logqq = :logqq in keys(save) ? save[:logqq] : logpdf(proposal, mm)

    ww .*= exp.(logpp)
    ww ./= exp.(logqq)
    return merge((mm=mm, ww=ww, logpp=logpp, logqq=logqq), save)
end

export AbstractMCMCChain, mcmc_sample
abstract type AbstractMCMCChain{M<:AbstractModel} end
function mcmc_sample(mcmc::AbstractMCMCChain{M}, y_obs, init_m::M) where M
    error("Implement me.")
end
function mcmc_sample(mcmc::AbstractMCMCChain{M}, y_obs, NChains::Int64; parallel::Bool=false) where M
    initial_vec = rand(mcmc.prior, NChains)[:mm]
    return mcmc_sample(mcmc, y_obs, initial_vec; parallel=parallel)
end
function mcmc_sample(mcmc::AbstractMCMCChain{M}, y_obs, initial_vec::Array{M,1}; parallel::Bool=false) where M
    F(init_m) = mcmc_sample(mcmc, y_obs, init_m)
    chain_vec = parallel ? pmap(F, initial_vec) : map(F, initial_vec)
    return chain_vec
end

export GaussianWalk
struct GaussianWalk{M<:AbstractModel, D<:DistributionGenerator{M}, W<:AbstractWeight{M}} <: AbstractMCMCChain{M}
    prior::D
    w::W
    N::Int64
    covMat::Matrix{Float64}
end
function mcmc_sample(mcmc::GaussianWalk{M}, y_obs, initial::M=rand(mcmc.prior)[:m]) where M
    mm = Array{M, 1}(undef, mcmc.N)
    logpp = Array{Float64, 1}(undef, mcmc.N)
    logww = Array{Float64, 1}(undef, mcmc.N)

    mm[1] = initial
    logpp[1] = logpdf(mcmc.prior, mm[1])[:logq]
    logww[1] = weight(mcmc.w, mm[1], y_obs)[:logw]
    
    # PerturbationKernel is MvNormal, centred on the current parameter value
    # Symmetric, therefore do not need to worry about ratio of proposal likelihoods
    K = PerturbationKernel(mm[1], mcmc.covMat, mcmc.prior)

    for i in 2:mcmc.N
        proposal = rand(K)
        mm[i] = proposal[:m]
        logpp[i] = proposal[:logp]
        logww[i] = weight(mcmc.w, mm[i], y_obs)[:logw]
        alpha = exp(logww[i] + logpp[i] - logww[i-1] - logpp[i-1])
        if (rand()>alpha)
            mm[i] = mm[i-1]
            logpp[i] = logpp[i-1]
            logww[i] = logww[i-1]
        else
            recentre!(K, mm[i])
        end
    end
    return (mm=mm, logww=logww, logpp=logpp)
end

struct Haario{M} <: AbstractMCMCChain{M}
    prior::AbstractGenerator{M}
    w::AbstractWeight{M}
    N::Int64
    covMat0::Matrix{Float64}
    t0::Int64
    s_d::Float64
    epsilon::Float64
end


