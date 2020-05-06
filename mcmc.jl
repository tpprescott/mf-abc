export MCMCProposal, propose, initial
export mcmc_sample

struct MCMCProposal{Θ<:AbstractModel, Π<:AbstractGenerator{Θ}, W<:LikelihoodObservationSet}
    prior::Π
    K::PerturbationKernel{Θ}
    lh_set::W
    function MCMCProposal(prior::Π, K::PerturbationKernel, lh_set::W) where Π <: AbstractGenerator{Θ} where Θ where W
        return new{Θ, Π, W}(prior, K, lh_set)
    end
    function MCMCProposal(prior::Π, C0::Matrix{Float64}, lh_set::W) where Π<:AbstractGenerator{Θ} where Θ where W
        init = rand(prior)
        K = PerturbationKernel(init.θ, C0)
        return new{Θ, Π, W}(prior, K, lh_set)
    end
end

MCMCProposal(
    prior::AbstractGenerator{Θ},
    K,
    L_set::NTuple{N, L where L<:AbstractLikelihoodFunction}, 
    y_obs_set::NTuple{N, X where X}
) where N where Θ = MCMCProposal(prior, K, (L_set, y_obs_set))

MCMCProposal(
    prior::AbstractGenerator{Θ}, 
    K,
    lh_fun::LikelihoodObservationPair...
) where Θ = MCMCProposal(prior, K, zip(lh_fun...)...)

MCMCProposal(
    prior::AbstractGenerator{Θ}, 
    K,
    w::AbstractLikelihoodFunction, 
    y_obs,
) where Θ = MCMCProposal(prior, K, ((w,), (y_obs,)))

function MCMCProposal(prior::AbstractGenerator{Θ}, args...) where Θ
error("$(args...) is not valid for creating an MCMC proposal")
end

function propose(
    Σ::MCMCProposal{Θ, Π, LikelihoodObservationSet{N, TLL, TXX}},
    L_pre...;
    kwargs...
) where Θ where Π where TLL<:NTuple{N,AbstractLikelihoodFunction} where TXX<:NTuple{N,Any} where N

    proposal = rand(Σ.K; prior=Σ.prior, kwargs...)
    if isfinite(proposal.logp)
        weight = loglikelihood(Σ.lh_set, proposal.θ, L_pre...; kwargs...)
    else
        L::TLL = Σ.lh_set[1]
        weight::NamedTuple{(:logw, :logww, :L), Tuple{Float64, NTuple{N, Float64}, TLL}} = (logw=-Inf, logww=Tuple(fill(-Inf, N)), L=L)
    end
    return merge(proposal, weight)
end

function initial(Σ::MCMCProposal; kwargs...)
    init_θ = rand(Σ.prior; kwargs...).θ
    recentre!(Σ.K, init_θ)
    proposal = propose(Σ)
    return isfinite(proposal.logw) ? proposal : initial(Σ; kwargs...)
end

# MCMC Sampling

import Base: IteratorSize, IsInfinite, IteratorEltype, HasEltype, eltype
IteratorSize(::Type{Σ}) where Σ<:MCMCProposal = IsInfinite()
IteratorEltype(::Type{Σ}) where Σ<:MCMCProposal = HasEltype()
function eltype(::Type{MCMCProposal{Θ, Π, LikelihoodObservationSet{N, TLL, TXX}}}) where Θ where Π where TXX<:NTuple{N, Any} where TLL<:NTuple{N, AbstractLikelihoodFunction} where N 
    return NamedTuple{(:θ, :θstar, :logp, :logw, :logww),
    Tuple{Θ, Θ, Float64, Float64, NTuple{N, Float64}}}
end

function Base.iterate(Σ::MCMCProposal)
    init = initial(Σ)
    out = (θ = init.θ, θstar = init.θ, logp = init.logp, logw = init.logw, logww = init.logww)
    recentre!(Σ.K, init.θ)
    return out, init
end

function Base.iterate(Σ::MCMCProposal, state::NamedTuple)
    proposal = propose(Σ, state.L)
    logα = metropolis_hastings(state, proposal)
    if log(rand())<logα 
        state = proposal
        recentre!(Σ.K, state.θ)
    end
    out = (θ = state.θ, θstar = proposal.θ, logp = proposal.logp, logw = proposal.logw, logww = proposal.logww)
    return out, state
end

function metropolis_hastings(state, proposal)
    # We are only working with symmetric proposal distributions
    return proposal[:logp] + proposal[:logw] - state[:logp] - state[:logw]
end

function mcmc_sample(
    Σ::MCMCProposal,
    numChain::Int64,
)   
    σ_n = Iterators.take(Σ, numChain)
    return table(collect(σ_n))
end