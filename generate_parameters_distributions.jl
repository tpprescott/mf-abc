using Distributions
import Distributions.rand, Distributions.rand!

export DistributionGenerator
struct DistributionGenerator{M, D<:Distribution} <: AbstractGenerator{M}
    d::D
    function DistributionGenerator(::Type{M}, x::D) where M where D<:Distribution
        N = length(fieldnames(M))
        length(x)==N || error("$N names for $(length(x)) variables")
        return new{M,D}(x)
    end
end

function rand(q::DistributionGenerator{M, D}) where M<:AbstractModel where D
    v = rand(q.d)
    logq = logpdf(q.d, v)
    return (m=M(v), logq=logq)
end
function rand!(mm::AbstractArray{M,1}, q::DistributionGenerator{M,D}) where M<:AbstractModel where D
    logqq = Array{Float64}(undef, length(mm))
    vv = rand(q.d, length(mm))
    if length(q.d)==1
        logqq[:] .= logpdf.(q.d, vv)
        mm[:] .= M.(vv)
    else
        logqq = logpdf!(logqq, q.d, vv)
        for i in eachindex(mm)
            mm[i] = M(view(vv,:,i))
        end
    end
    return (logqq=logqq,)
end

function make_array(m::M) where M<:AbstractModel
    dim = length(m)
    if dim==1
        return m[1]
    else
        return [values(m)...]
    end
end
function make_array(mm::Array{M,1}) where M<:AbstractModel
    dim = length(mm[1])
    if dim==1
        return make_array.(mm)
    else
        value_array = make_array.(mm)
        return hcat(value_array...)
    end
end

function logpdf(q::DistributionGenerator{M,D}, m::M) where M where D
    v = make_array(m)
    return (logq = logpdf(q.d, v),)
end
function logpdf(q::DistributionGenerator{M,D}, mm::AbstractArray{M,1}) where M where D
    dim = length(mm[1])
    v = make_array(mm)
    if dim==1
        logqq = logpdf.(q.d, v)
    else
        logqq = logpdf(q.d, v)
    end
    return (logqq=logqq, )
end
function logpdf!(logqq, q::DistributionGenerator{M,D}, mm::AbstractArray{M,1}) where M where D
    dim = length(mm[1])
    v = make_array(mm)
    if dim==1
        broadcast!(logpdf, logqq, q.d, v)
    else
        logpdf!(logqq, q.d, v)
    end
    return NamedTuple()
end
