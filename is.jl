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
    if isfinite(proposal.logp)
        weight = loglikelihood(Σ.lh_set, proposal.θ; kwargs...)
    else
        L::TLL = Σ.lh_set[1]
        M = length(L)
        weight = (logw=-Inf, logww=Tuple(fill(-Inf, M)))
    end
    return merge(proposal, Base.structdiff(weight, NamedTuple{(:L,)}))
end

export importance_sample
function importance_sample(
    Σ::ISProposal,
    numSample::Int64;
    kwargs...
)

    I_Σ = Iterators.repeated(Σ)
    I_kw = Iterators.repeated(kwargs)
    I = zip(I_Σ, I_kw)
    
    b = pmap(_batch, Iterators.take(I, numSample))
    return table(b)
end
function _batch((Σ, kwargs))
    propose(Σ; kwargs...)
end 