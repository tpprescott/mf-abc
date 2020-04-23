using Distributed

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

function (Σ::MonteCarloProposal)(i=1; loglikelihood::Bool=false, kwargs...)
    proposal::NamedTuple{(:θ, :logq, :logp)} = rand(Σ.q; prior=Σ.prior, kwargs...)
    if isfinite(proposal.logp)
        weight = likelihood(Σ.lh_set, proposal[:θ]; loglikelihood=loglikelihood, kwargs...)
    else
        weight = loglikelihood ? (logw=-Inf, L=Σ.lh_set[1]) : (w=0.0, L=Σ.lh_set[1])
    end
    return merge(proposal, weight)
end

function batch(Σ::MonteCarloProposal, N::Int64; parallel::Bool=false, kwargs...)
    Σ_opt(i) = Σ(i; kwargs...)
    b = parallel ? pmap(Σ_opt, Base.OneTo(N)) : map(Σ_opt, Base.OneTo(N))
    return table(b)
end

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
        append!(rows(sample), rows(batch(Σ, batch_size; parallel=parallel, kwargs...)))
    end
    return sample
end
importance_sample(Σ::MonteCarloProposal, f, n; kwargs...) = importance_sample(Σ, StopCondition(f, n); kwargs...)

# MCMC Sampling
function Base.iterate(Σ::MonteCarloProposal)
    initial_proposal = rand(Σ.prior)
    θ = initial_proposal[:θ]

    weight = likelihood(Σ.lh_set, θ; loglikelihood=true)
    isfinite(weight[:logw]) || (return Base.iterate(Σ))

    initial_state = merge(initial_proposal, weight)
    out = merge((θ_accept = θ,), initial_state)
    return out, initial_state
end

function Base.iterate(Σ::MonteCarloProposal, state::NamedTuple)
    recentre!(Σ.q, state[:θ])
    proposal = Σ(; loglikelihood=true)
    logα = metropolis_hastings(state, proposal)
    new_state = log(rand())<logα ? proposal : state
    return merge((θ_accept = new_state[:θ],), proposal), new_state
end

function metropolis_hastings(state, proposal)
    # Assuming we are only working with symmetric proposal distributions
    return proposal[:logp] + proposal[:logw] - state[:logp] - state[:logw]
end

import Base: IteratorSize, IsInfinite, IteratorEltype, HasEltype, eltype
IteratorSize(::Type{MonteCarloProposal}) = IsInfinite()
IteratorEltype(::Type{MonteCarloProposal}) = HasEltype()
eltype(::Type{MonteCarloProposal}) = NamedTuple{(:θ_accept, :θ, :logq, :logp, :logw, :w_components, :L)}

export mcmc_sample
function mcmc_sample(
    Σ::MonteCarloProposal,
    stop::StopCondition = StopCondition(length, 100)
)   
    σ = Iterators.Stateful(Σ)

    sample = [first(σ)]
    while !stop(sample)
        push!(sample, first(σ))
    end
    return table(sample)
end
mcmc_sample(Σ::MonteCarloProposal, f, n; kwargs...) = mcmc_sample(Σ, StopCondition(f, n); kwargs...)
