export importance_sample

export MonteCarloProposal
struct MonteCarloProposal{Θ<:AbstractModel, Π<:AbstractGenerator{Θ}, Q<:AbstractGenerator{Θ}, W<:LikelihoodObservationSet}
    prior::Π
    q::Q
    lh_set::W
end
MonteCarloProposal(prior::AbstractGenerator{Θ}, q::AbstractGenerator{Θ}, L_set::NTuple{N, AbstractLikelihoodFunction}, y_obs_set::NTuple{N, X where X}) where N where Θ = MonteCarloProposal(prior, q, (L_set, y_obs_set))
MonteCarloProposal(prior::AbstractGenerator{Θ}, q::AbstractGenerator{Θ}, lh_fun::LikelihoodObservationPair...) where Θ = MonteCarloProposal(prior, q, zip(lh_fun...)...)
MonteCarloProposal(prior::AbstractGenerator{Θ}, q::AbstractGenerator{Θ}, w::AbstractLikelihoodFunction, y_obs) where Θ = MonteCarloProposal(prior, q, (w, y_obs))
MonteCarloProposal(prior::AbstractGenerator{Θ}, args...) where Θ = MonteCarloProposal(prior, prior, args...)
function MonteCarloProposal(prior::AbstractGenerator{Θ}, q::AbstractGenerator{Θ}, args...) where Θ
    for x_i in args
        println(x_i)
    end
end

function (Σ::MonteCarloProposal)(; loglikelihood::Bool=false, kwargs...)
    proposal::NamedTuple{(:θ, :logq, :logp)} = rand(Σ.q; prior=Σ.prior, kwargs...)
    if isfinite(proposal.logp)
        weight = likelihood(Σ.lh_set, proposal.θ; loglikelihood=loglikelihood, kwargs...)
    else
        L = Σ.lh_set[1]
        N = length(L)
        weight = loglikelihood ? (logw=-Inf, w_components=Tuple(fill(-Inf, N)), L=L) : (w=0.0, w_components=Tuple(fill(0.0, N)), L=L)
    end
    return merge(proposal, weight)
end

function batch(Σ::MonteCarloProposal, N::Int64; kwargs...)
    I_Σ = Iterators.repeated(Σ)
    I_kw = Iterators.repeated(kwargs)
    I = zip(I_Σ, I_kw)
    
    b = pmap(_batch, Iterators.take(I, N))
    return b
end
_batch((Σ, kwargs)) = Σ(; kwargs...)

export StopCondition
struct StopCondition{F,N} 
    f::F
    n::N
end
function (stop::StopCondition)(sample)::Bool
    stop.f(sample) >= stop.n
end

function importance_sample(
    Σ::MonteCarloProposal,
    stop::StopCondition = StopCondition(length, 100);
    parallel::Bool = false,
    batch_size::Int64 = 10,
    kwargs...
)
    sample = batch(Σ, batch_size; parallel=parallel, kwargs...)
    while !stop(sample)
        append!(sample, batch(Σ, batch_size; parallel=parallel, kwargs...))
    end
    return sample
end
importance_sample(Σ::MonteCarloProposal, f, n; kwargs...) = importance_sample(Σ, StopCondition(f, n); kwargs...)

# MCMC Sampling

import Base: IteratorSize, IsInfinite, IteratorEltype, HasEltype, eltype
IteratorSize(::Type{Σ}) where Σ<:MonteCarloProposal = IsInfinite()
IteratorEltype(::Type{Σ}) where Σ<:MonteCarloProposal = HasEltype()
function eltype(::Type{MonteCarloProposal{
    Θ, Π, Q, Tuple{LH, Y}
}}) where {Θ, Π, Q, Y} where LH <: NTuple{N, AbstractLikelihoodFunction} where N
    return NamedTuple{(:θ, :θstar, :logq, :logp, :logw, :w_components),
    Tuple{Θ, Θ, Float64, Float64, Float64, NTuple{N, Float64}}}
end


function Base.iterate(Σ::MonteCarloProposal)
    initial_proposal = rand(Σ.prior)
    θ = initial_proposal[:θ]

    weight = likelihood(Σ.lh_set, θ; loglikelihood=true)
    isfinite(weight[:logw]) || (return Base.iterate(Σ))

    initial_state = merge(initial_proposal, weight)
    out = (θ = θ, θstar = θ, logq = initial_proposal.logq, logp = initial_proposal.logp, logw = weight.logw, w_components = weight.w_components)
    return out, initial_state
end

function Base.iterate(Σ::MonteCarloProposal, state::NamedTuple)
    recentre!(Σ.q, state[:θ])
    proposal = Σ(; loglikelihood=true)
    logα = metropolis_hastings(state, proposal)
    new_state = log(rand())<logα ? proposal : state
    out = (θ = new_state.θ, θstar = proposal.θ, logq = proposal.logq, logp = proposal.logp, logw = proposal.logw, w_components = proposal.w_components)
    return out, new_state
end

function metropolis_hastings(state, proposal)
    # Assuming we are only working with symmetric proposal distributions
    return proposal[:logp] + proposal[:logw] - state[:logp] - state[:logw]
end


export mcmc_sample
function mcmc_sample(
    Σ::MonteCarloProposal,
    stop::StopCondition = StopCondition(length, 100);
    batch_size = 100,
)   
    σ = Iterators.Stateful(Σ)
    sample = collect(Iterators.take(σ, batch_size))
    while !stop(sample)
        append!(sample, collect(Iterators.take(σ, batch_size)))
    end
    return sample
end
mcmc_sample(Σ::MonteCarloProposal, f, n; kwargs...) = mcmc_sample(Σ, StopCondition(f, n); kwargs...)
