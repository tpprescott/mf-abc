using Distributions
import Distributions: rand, rand!

export PerturbationKernel, recentre!
struct PerturbationKernel{M, DP} <: AbstractGenerator{M}
    d::MvNormal
    prior::DP
end
function PerturbationKernel(m::M, covMat, prior::DistributionGenerator{M,D}) where M where D
    d = MvNormal([values(m)...], covMat)
    return PerturbationKernel{M,D}(d, prior.d)
end
function recentre!(K::PerturbationKernel{M,DP}, m::M) where M<:AbstractModel where DP
    for (i,v) in enumerate(values(m))
        K.d.μ[i] = v
    end
end

function rand(K::PerturbationKernel{M,DP}) where M where DP
    v = rand(K.d)
    logp = logpdf(K.prior, v)
    isfinite(logp) || (return rand(K))
    logq = logpdf(K.d, v)
    return (m=M(v), logq=logq, logp=logp)
end
function rand!(mm::AbstractArray{M}, q::AbstractGenerator{M})::NamedTuple where M<:AbstractModel
    N = length(mm)
    logpp = Array{Float64, 1}(undef, N)
    logqq = Array{Float64, 1}(undef, N)
    for i in eachindex(mm)
        mm[i], logqq[i], logpp[i] = values(rand(q))
    end
    return (logqq = logqq, logpp = logpp)
end

function logpdf(K::PerturbationKernel{M,DP}, m::M) where M where DP
    v = make_array(m)
    return logpdf(K.d, v)
end