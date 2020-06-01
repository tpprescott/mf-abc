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

function (q::DistributionGenerator{Θ, D})(; kwargs...) where Θ<:AbstractModel where D
    v = rand(q.d)
    θ = Θ(v)
    logq = logpdf(q.d, v)
    return θ, logq
end
function logpdf(q::DistributionGenerator{Θ}, θ::Θ) where Θ<:AbstractModel
    v = make_array(θ)
    return logpdf(q.d, v)
end

function make_array(θ::Θ) where Θ<:Union{NamedTuple, Tuple}
    dim = length(θ)
    if dim==1
        return θ[1]
    else
        return [values(θ)...]
    end
end
function make_array(θ::AbstractArray{Θ,1}) where Θ<:Union{NamedTuple, Tuple}
    dim = length(θ[1])
    if dim==1
        return make_array.(θ)   
    else
        return hcat(make_array.(θ)...)
    end
end

# Special case - perturbation kernel is multivariate normal. Allow translation.
export PerturbationKernel, recentre!
PerturbationKernel{Θ} = DistributionGenerator{Θ, <:MvNormal}

function PerturbationKernel(θ::Θ, covMat) where Θ<:AbstractModel
    d = MvNormal([values(θ)...], covMat)
    return DistributionGenerator(Θ, d)
end
function recentre!(K::PerturbationKernel{Θ}, θ::Θ) where Θ<:AbstractModel
    for (i,v) in enumerate(values(θ))
        K.d.μ[i] = v
    end
end