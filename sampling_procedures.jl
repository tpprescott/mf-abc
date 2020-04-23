using Distributed

export rejection_sample, importance_sample

function batch(Σ::MonteCarloProposal, N::Int64; parallel::Bool=false)
    b = parallel ? pmap(Σ, Base.OneTo(N)) : map(Σ, Base.OneTo(N))
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
)
    sample = batch(Σ, batch_size, parallel=parallel)
    while !stop(sample)
        append!(rows(sample), rows(batch(Σ, batch_size, parallel=parallel)))
    end
    return sample
end
importance_sample(Σ::MonteCarloProposal, f, n; kwargs...) = importance_sample(Σ, StopCondition(f, n); kwargs...)

# MCMC Sampling
function Base.iterate(Σ::MonteCarloProposal)
    initial_proposal = rand(Σ.prior)
    b = weight(Σ.lh_set; initial_proposal...)
    logw = 0.0
    for b_i in b
        logw += b_i[:logw]
    end
    isfinite(logw) || (return Base.iterate(Σ))
    θ = initial_proposal[:θ]

    initial_state = merge(initial_proposal, (logw=logw, weight_components=b))
    out = merge((θ_accept = θ,), initial_state)
    return out, initial_state
end

function Base.iterate(Σ::MonteCarloProposal, state::NamedTuple)
    recentre!(Σ.q, state[:θ])
    proposal = Σ()
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
eltype(::Type{MonteCarloProposal}) = NamedTuple{(:θ_accept, :θ, :logq, :logp, :logw, :weight_components)}

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
