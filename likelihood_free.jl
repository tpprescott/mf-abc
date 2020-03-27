module LikelihoodFree

export AbstractModel, AbstractGenerator, AbstractExperiment, AbstractWeight
# The model contains the parameters that will be inferred
abstract type AbstractModel end
# The generator will generate the model according to some distribution (rejection sampling or importance sampling)
abstract type AbstractGenerator{M<:AbstractModel} end
# The experiment contains the data (the space of which parameterises the type) and also conditions under which it was gathered (to control simulations if needed)
abstract type AbstractExperiment{D} end 
data_type(::AbstractExperiment{D}) where D = D
# The weighter is essentially a likelihood estimate
abstract type AbstractWeight{M<:AbstractModel, U<:AbstractExperiment} end

export MCOutput
MCOutput{M<:AbstractModel} = Array{Tuple{M, Float64}, 1}

# Basic experiment type is just data alone
export ExperimentalData
struct ExperimentalData{D} <: AbstractExperiment{D}
    y_obs::D
end

#######
# Generate models from the generator

import Base.rand
export rand

function rand(q::AbstractGenerator{M}; saved...)::M where M<:AbstractModel
    return q()
end
function rand!(x::AbstractArray{M}, q::AbstractGenerator{M}; saved...)::Nothing where M<:AbstractModel
    for i in eachindex(x)
        x[i] = rand(q)
    end
    return nothing
end
function rand(q::AbstractGenerator{M}, N::Vararg{Int64,K}; saved...)::Array{M,K} where M<:AbstractModel where K
    x = Array{M,K}(undef, N...)
    rand!(x, q; saved...)
    return x
end

# Apply a Monte Carlo weight based on the simulation output
export weight
function weight(w::AbstractWeight{M,U}, m::M, u::U; saved...)::Float64 where M where U
    return w(m,u)
end
function weight!(x::AbstractArray{Float64}, w::AbstractWeight{M,U}, mm::AbstractArray{M}, u::U; saved...)::Nothing where M where U
    size(x)==size(mm) || error("Mismatched sizes between preallocated array and data set")
    for i in eachindex(mm)
        x[i] = weight(w, mm[i], u; i=i, saved...)
    end
    return nothing
end
function weight(w::AbstractWeight{M,U}, mm::AbstractArray{M}, u::U; saved...)::AbstractArray{Float64} where M where U
    x = Array{Float64}(undef, size(mm))
    weight!(x, w, mm, u; saved...)
    return x
end

include("rejection_sampling.jl")

######### Now go likelihood-free: a subtype of AbstractWeight that will be dependent on a simulation
# Here defined for a single fidelity
export AbstractSummaryStatisticSpace
abstract type AbstractSummaryStatisticSpace end

export LikelihoodFreeWeight, AbstractSimulator, AbstractComparison
abstract type AbstractSimulator{M<:AbstractModel, U<:AbstractExperiment, Y<:AbstractSummaryStatisticSpace} end
abstract type AbstractComparison{U<:AbstractExperiment, Y<:AbstractSummaryStatisticSpace} end

struct LikelihoodFreeWeight{M, U, Y<:AbstractSummaryStatisticSpace, TF<:AbstractSimulator{M,U,Y}, TC<:AbstractComparison{U,Y}} <: AbstractWeight{M,U}
    F::TF
    C::TC
end

export simulate, compare
function simulate(F::AbstractSimulator{M,U,Y}, m::M, u::U; saved...)::Y where M where U where Y
    return F(m, u)
end
function simulate!(F::AbstractSimulator{M,U,Y}, mm::AbstractArray{M}, u::U; yy::AbstractArray{Y}, saved...) where M where U where Y
    for i in eachindex(mm)
        yy[i] = simulate(F, mm[i], u; i=i, saved...)
    end
    return nothing
end
function simulate(F::AbstractSimulator{M,U,Y}, mm::AbstractArray{M}, u::U; saved...)::AbstractArray{Y} where M where U where Y
    yy = Array{Y}(undef, size(mm))
    simulate!(F, mm, u; yy=yy, saved...)
    return yy
end

function compare(C::AbstractComparison{U,Y}, u::U, y::Y; saved...)::Float64 where U where Y
    return C(u, y)
end
function compare!(w::AbstractArray{Float64}, C::AbstractComparison{U,Y}, u::U; yy::AbstractArray{Y}, saved...) where U where Y
    for i in eachindex(yy)
        w[i] = compare(C, u, yy[i]; i=i, saved...)
    end
    return nothing
end
function compare(C::AbstractComparison{U,Y}, u::U; yy::AbstractArray{Y}, saved...) where U where Y
    w = Array{Float64}(undef, size(yy))
    compare!(w, C, u; yy=yy, saved...)
    return w
end

function weight(w::LikelihoodFreeWeight{M,U,Y}, m::M, u::U)::Float64 where M where U where Y
    y = simulate(w.F, m, u)
    out = compare(w.C, u, y)
    return out
end
function weight!(
    out::AbstractArray{Float64}, 
    w::LikelihoodFreeWeight{M,U,Y}, 
    mm::AbstractArray{M}, 
    u::U; 
    yy::AbstractArray{Y} = Array{Y}(undef, size(mm)), 
    saved...
)::Nothing where M where U where Y

    simulate!(w.F, mm, u; yy=yy, saved...)
    compare!(out, w.C, u; yy=yy, saved...)
    return nothing
end
function weight(w::LikelihoodFreeWeight{M,U,Y}, mm::AbstractArray{M}, u::U; yy::AbstractArray{Y}, saved...)::AbstractArray{Float64} where M where U where Y
    out = Array{Float64}(undef, size(mm))
    weight!(out, w, mm, u; yy=yy, saved...)
    return out
end

function sample(u::U, q::AbstractGenerator{M}, F::AbstractSimulator{M,U,Y}, C::AbstractComparison{U,Y}, N; saved...) where M where U where Y
    yy = Array{Y}(undef, N)
    return sample(u, q, LikelihoodFreeWeight(F, C), N; yy=yy, saved...)
end

export output_type, data_type
output_type(::AbstractSimulator{M,U,Y}) where M where U where Y = Y
output_type(::AbstractComparison{U,Y}) where U where Y = Y
output_type(::LikelihoodFreeWeight{M,U,Y}) where M where U where Y = Y
data_type(::AbstractSimulator{M,U,Y}) where M where U where Y = data_type(U)
data_type(::AbstractComparison{U,Y}) where U where Y = data_type(U)
data_type(::LikelihoodFreeWeight{M,U,Y}) where M where U where Y = data_type(U)

end