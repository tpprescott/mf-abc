# Importance sampling
export ISProposal

struct ISProposal{Θ<:AbstractModel, Π<:AbstractGenerator{Θ}, Q<:AbstractGenerator{Θ}, W<:LikelihoodObservationSet}
    prior::Π
    q::Q
    lh_set::W
    function ISProposal(prior::Π, q::Q, lh_set::W) where Π <: AbstractGenerator{Θ} where Q<:AbstractGenerator{Θ} where Θ where W
        return new{Θ, Π, Q, W}(prior, q, lh_set)
    end
end

ISProposal(
    prior::AbstractGenerator{Θ},
    q::AbstractGenerator{Θ},
    L_set::NTuple{N, L where L<:AbstractLikelihoodFunction}, 
    y_obs_set::NTuple{N, X where X}
) where N where Θ = ISProposal(prior, q, (L_set, y_obs_set))

ISProposal(
    prior::AbstractGenerator{Θ}, 
    q::AbstractGenerator{Θ},
    lh_fun::LikelihoodObservationPair...
) where Θ = ISProposal(prior, q, zip(lh_fun...)...)

ISProposal(
    prior::AbstractGenerator{Θ}, 
    q::AbstractGenerator{Θ},
    w::AbstractLikelihoodFunction, 
    y_obs,
) where Θ = ISProposal(prior, q, ((w,), (y_obs,)))

ISProposal(
    prior::AbstractGenerator{Θ}, 
    args...,
) where Θ = ISProposal(prior, prior, args...)

function ISProposal(prior::AbstractGenerator{Θ}, q::AbstractGenerator{Θ}, args...) where Θ
    error("$(args...) is not valid for creating an IS proposal")
end

function propose(
    Σ::ISProposal{Θ, Π, Q, LikelihoodObservationSet{N, TLL, TXX}}, 
    args...; kwargs...) where Π where Q where TLL where TXX where Θ where N

    proposal = rand(Σ.q; prior=Σ.prior, kwargs...)
    isfinite(proposal.logp) || (return propose(Σ, args...; kwargs...))
    logL = loglikelihood(Σ.lh_set, proposal.θ; kwargs...)
    return merge(proposal, (logw = logL.logw, logww = logL.logww, logtest = logL.logtest))
end

export importance_sample
function importance_sample(
    Σ::ISProposal,
    numSample::Int64;
    autoaccept_pc::Float64 = 0.05,
    scale::Float64 = 1.0,
    kwargs...
)

    I_Σ = Iterators.repeated(Σ)
    I_kw = Iterators.repeated(kwargs)
    I = zip(I_Σ, I_kw)
    
    b = @showprogress pmap(_batch, Iterators.take(I, numSample))
    t = table(b)
    
    # Importance weighting
    logw = select(t, :logw)
    logw .-= quantile(logw, 1.0-autoaccept_pc, sorted=false)
    logw .*= scale
    logu = log.(rand(numSample))
    acceptance = logu .< logw

    importance_weight = zeros(numSample)
    for (i, particle_i) in enumerate(b)
        if acceptance[i]
            importance_weight[i] = exp(max(0.0, logw[i]) + particle_i.logp - particle_i.logq)
        end
    end
    return transform(t, :weight => importance_weight)
end
function _batch((Σ, kwargs))
    propose(Σ; kwargs...)
end 