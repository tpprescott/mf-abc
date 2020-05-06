using Distributions
import Distributions.rand, Distributions.rand!

export DistributionGenerator
struct DistributionGenerator{Θ, D<:Distribution} <: AbstractGenerator{Θ}
    d::D
    function DistributionGenerator(::Type{Θ}, x::D) where Θ where D<:Distribution
        N = length(fieldnames(Θ))
        length(x)==N || error("$N names for $(length(x)) variables")
        return new{Θ, D}(x)
    end
end

function (q::DistributionGenerator{Θ, D})(; prior::AbstractGenerator{Θ}, kwargs...) where Θ<:AbstractModel where D
    v = rand(q.d)
    θ = Θ(v)
    logq = logpdf(q.d, v)
    logp = prior==q ? logq : logpdf(prior, θ)
#    isfinite(logp) || (return q(; prior=prior, kwargs...))
    return (θ=θ, logq=logq, logp=logp)
end
function logpdf(q::DistributionGenerator{Θ, D}, θ::Θ) where Θ<:AbstractModel where D
    v = make_array(θ)
    return logpdf(q.d, v)
end

function make_array(θ::Θ) where Θ<:AbstractModel
    dim = length(θ)
    if dim==1
        return θ[1]
    else
        return [values(θ)...]
    end
end
function make_array(θ::AbstractArray{Θ,1}) where Θ<:AbstractModel
    dim = length(θ[1])
    N = length(θ)
    if dim==1
        return make_array.(θ)   
    else
        return hcat(make_array.(θ)...)
    end
end