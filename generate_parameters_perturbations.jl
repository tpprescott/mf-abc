using Distributions
import Distributions: rand

export PerturbationKernel, recentre!
struct PerturbationKernel{Θ} <: AbstractGenerator{Θ}
    d::MvNormal
end

function PerturbationKernel(θ::Θ, covMat) where Θ
    d = MvNormal([values(θ)...], covMat)
    return PerturbationKernel{Θ}(d)
end
function recentre!(K::PerturbationKernel{Θ}, θ::Θ) where Θ<:AbstractModel
    for (i,v) in enumerate(values(θ))
        K.d.μ[i] = v
    end
end

function (K::PerturbationKernel{Θ})(; prior::AbstractGenerator{Θ}, kwargs...) where Θ
    v = rand(K.d)
    θ = Θ(v)
    logp = logpdf(prior, θ)
#    isfinite(logp) || (return K(; prior=prior, kwargs...))
    logq = logpdf(K.d, v)
    return (θ=θ, logq=logq, logp=logp)
end
function logpdf(K::PerturbationKernel{Θ}, θ::Θ) where Θ
    v = make_array(θ)
    return logpdf(K.d, v)
end