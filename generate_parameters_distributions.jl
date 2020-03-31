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

function (q::DistributionGenerator{M,D})()::M where M where D
    return M(rand(q.d))
end
function rand(q::DistributionGenerator{M, D}) where M<:AbstractModel where D
    return (m=q(),)
end

function rand!(mm::AbstractArray{M}, q::DistributionGenerator{M,D}) where M<:AbstractModel where D
    xx = rand(q.d, length(mm))
    if length(q.d)==1
        for i in eachindex(mm)
            mm[i] = M(xx[i])
        end
    else
        for i in eachindex(mm)
            mm[i] = M(view(xx,:,i))
        end
    end
    return NamedTuple()
end

function make_array(mm::Array{M}) where M<:AbstractModel
    dim = length(fieldnames(M))
    if dim==1
        T = fieldtypes(M)[1]
        out = Vector{T}(undef, length(mm))
        for i in eachindex(mm)
            out[i] = mm[i][1]
        end
    else
        T = promote_type(fieldtypes(M)...)
        out = Array{T,2}(undef, dim, length(mm))
        vals = values.(mm)
        for i in eachindex(mm)
            for j in 1:dim
                out[j,i] = vals[i][j]
            end
        end
    end
    return out
end

function unnormalised_likelihood(q::DistributionGenerator{M,D}, m::M) where M<:AbstractModel where D
    dim = length(q.d)
    p = (dim==1 ? pdf(q.d, values(m)[1]) : pdf(q.d, [values(m)...]))
    return (p=p,)
end
function unnormalised_likelihood!(pp::AbstractArray{Float64,1}, q::DistributionGenerator{M, D}, mm::AbstractArray{M}) where M where D
    dim = length(q.d)
    mm_array = make_array(mm)
    pp[:] .= (dim==1 ? pdf.(q.d, mm_array) : pdf(q.d, mm_array))
    return NamedTuple()
end
function unnormalised_likelihood(q::DistributionGenerator{M,D}, mm::AbstractArray{M}) where M<:AbstractModel where D
    pp = Array{Float64,1}(undef, size(mm))
    save = unnormalised_likelihood!(pp, q, mm)
    return merge((pp=pp,), save)
end
