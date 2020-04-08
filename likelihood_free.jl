module LikelihoodFree

export AbstractModel, AbstractGenerator, AbstractExperiment, AbstractWeight
# The model contains the parameters that will be inferred
AbstractModel = NamedTuple
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

include("generate_parameters.jl")
include("sampling_procedures.jl")

# Need to apply a Monte Carlo weight to the generated parameters 
export weight
function weight(w::AbstractWeight{M,U}, m::M, u::U)::NamedTuple where M where U
    return (w = w(m,u),)
end
function weight!(ww::AbstractArray{Float64}, w::AbstractWeight{M,U}, mm::AbstractArray{M}, u::U)::NamedTuple where M where U
    size(ww)==size(mm) || error("Mismatched sizes between preallocated array and data set")
    for i in eachindex(mm)
        ww[i] = weight(w, mm[i], u)[:w]
    end
    return NamedTuple()
end
function weight(w::AbstractWeight{M,U}, mm::AbstractArray{M}, u::U)::NamedTuple where M where U
    ww = Array{Float64}(undef, size(mm))
    save = weight!(ww, w, mm, u)
    return merge((ww=ww,), save)
end


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
function simulate(F::AbstractSimulator{M,U,Y}, m::M, u::U)::NamedTuple where M where U where Y
    return (y = F(m, u),)
end
function simulate!(yy::AbstractArray{Y}, F::AbstractSimulator{M,U,Y}, mm::AbstractArray{M}, u::U)::NamedTuple where M where U where Y
    for i in eachindex(mm)
        yy[i] = simulate(F, mm[i], u)[:y]
    end
    return NamedTuple()
end
function simulate(F::AbstractSimulator{M,U,Y}, mm::AbstractArray{M,1}, u::U) where M where U where Y
    yy = Array{Y,1}(undef, length(mm))
    save = simulate!(yy, F, mm, u)
    return merge((yy=yy,), save)
end

# Multiple (K) simulations per parameter value
function simulate(F::AbstractSimulator{M,U,Y}, mm::AbstractArray{M,1}, u::U, K::Int64) where M where U where Y
    yy = Array{Array{Y,1},1}(undef, length(mm))
    save = simulate!(yy, F, mm, u, K)
    return merge((yy=yy,), save)
end
function simulate!(yy::Array{Array{Y,1},1}, F::AbstractSimulator{M,U,Y}, mm::AbstractArray{M,1}, u::U, K::Int64) where M where U where Y
    for i in eachindex(mm)
        yy[i] = simulate(F, mm[i], u, K)[:y]
    end
    return NamedTuple()
end
function simulate(F::AbstractSimulator{M,U,Y}, m::M, u::U, K::Int64) where M where U where Y
    y = Array{Y,1}(undef, K)
    save = simulate!(y, F, m, u)
    return merge((y=y,), save)
end
function simulate!(y::AbstractArray{Y,1}, F::AbstractSimulator{M,U,Y}, m::M, u::U) where M where U where Y
    for i in eachindex(y)
        y[i] = simulate(F, m, u)[:y]
    end
    return NamedTuple()
end

function compare(C::AbstractComparison{U,Y}, u::U, y::Y)::NamedTuple where U where Y
    return (w = C(u, y),)
end
function compare!(ww::AbstractArray{Float64}, C::AbstractComparison{U,Y}, u::U, yy::AbstractArray{Y})::NamedTuple where U where Y
    for i in eachindex(yy)
        ww[i] = compare(C, u, yy[i])[:w]
    end
    return NamedTuple()
end
function compare(C::AbstractComparison{U,Y}, u::U, yy::AbstractArray{Y})::NamedTuple where U where Y
    ww = Array{Float64}(undef, size(yy))
    save = compare!(ww, C, u, yy)
    return merge((ww=ww,), save)
end

function weight(w::LikelihoodFreeWeight{M,U,Y}, m::M, u::U)::NamedTuple where M where U where Y
    sim = simulate(w.F, m, u)
    out = compare(w.C, u, sim[:y])
    return merge(out, sim)
end
function weight!(
    out::AbstractArray{Float64}, 
    w::LikelihoodFreeWeight{M,U,Y}, 
    mm::AbstractArray{M}, 
    u::U,
)::NamedTuple where M where U where Y

    yy = Array{Y}(undef, size(mm))
    save1 = simulate!(yy, w.F, mm, u)
    save2 = compare!(out, w.C, u, yy)
    return merge((yy=yy,), save1, save2)
end
function weight(w::LikelihoodFreeWeight{M,U,Y}, mm::AbstractArray{M}, u::U)::NamedTuple where M where U where Y
    out = Array{Float64}(undef, size(mm))
    save = weight!(out, w, mm, u)
    return merge((ww = out,), save)
end

function rejection_sample(u::U, q::AbstractGenerator{M}, F::AbstractSimulator{M,U,Y}, C::AbstractComparison{U,Y}, N) where M where U where Y
    return rejection_sample(u, q, LikelihoodFreeWeight(F, C), N)
end

export output_type, data_type
output_type(::AbstractSimulator{M,U,Y}) where M where U where Y = Y
output_type(::AbstractComparison{U,Y}) where U where Y = Y
output_type(::LikelihoodFreeWeight{M,U,Y}) where M where U where Y = Y
data_type(::AbstractSimulator{M,U,Y}) where M where U where Y = data_type(U)
data_type(::AbstractComparison{U,Y}) where U where Y = data_type(U)
data_type(::LikelihoodFreeWeight{M,U,Y}) where M where U where Y = data_type(U)

end