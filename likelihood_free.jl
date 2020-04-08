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
export LikelihoodFreeWeight, AbstractSimulator, AbstractComparison
abstract type AbstractSimulator{M<:AbstractModel, U<:AbstractExperiment, Y} end
abstract type AbstractComparison{U<:AbstractExperiment, Y} end

struct LikelihoodFreeWeight{M, U, Y, TF<:AbstractSimulator{M,U,Y}, TC<:AbstractComparison{U,Y}} <: AbstractWeight{M,U}
    F::TF
    C::TC
    K::Int64
end

export simulate, compare
## A one-dimensional array of Y corresponds to a single parameter
function simulate(F::AbstractSimulator{M,U,Y}, m::M, u::U, K::Int64=1) where M where U where Y
    y = Array{Y,1}(undef, K)
    save = simulate!(y, F, m, u)
    return merge((y=y,), save)
end
function simulate!(y::AbstractArray{Y,1}, F::AbstractSimulator{M,U,Y}, m::M, u::U) where M where U where Y
    for i in eachindex(y)
        y[i] = F(m, u)
    end
    return NamedTuple()
end

## A two-dimensional array of Y (even if the second is a singleton) corresponds to many parameters
function simulate(F::AbstractSimulator{M,U,Y}, mm::AbstractArray{M,1}, u::U, K::Int64=1) where M where U where Y
    yy = Array{Y,2}(undef, length(mm), K)
    save = simulate!(yy, F, mm, u)
    return merge((yy=yy,), save)
end
function simulate!(yy::AbstractArray{Y,2}, F::AbstractSimulator{M,U,Y}, mm::AbstractArray{M,1}, u::U) where M where U where Y
    for i in eachindex(mm)
        simulate!(view(yy, i, :), F, mm[i], u)
    end
    return NamedTuple()
end

# 1D array of Y corresponds to a single parameter sample (i.e a single weight)
function compare(C::AbstractComparison{U,Y}, u::U, y::AbstractArray{Y,1})::NamedTuple where U where Y
    return (w = C(u, y),)
end
# 2D array of Y corresponds to multiple parameter samples (i.e. multiple weights)
function compare(C::AbstractComparison{U,Y}, u::U, yy::AbstractArray{Y,2})::NamedTuple where U where Y
    ww = Array{Float64,1}(undef, size(yy,1))
    save = compare!(ww, C, u, yy)
    return merge((ww=ww,), save)
end
function compare!(ww::AbstractArray{Float64,1}, C::AbstractComparison{U,Y}, u::U, yy::AbstractArray{Y,2})::NamedTuple where U where Y
    for i in eachindex(ww)
        ww[i] = compare(C, u, view(yy,i,:))[:w]
    end
    return NamedTuple()
end

# Likleihood free weight combines simulation with comparison
function weight(w::LikelihoodFreeWeight{M,U,Y}, m::M, u::U)::NamedTuple where M where U where Y
    sim = simulate(w.F, m, u, w.K)
    out = compare(w.C, u, sim[:y])
    return merge(out, sim)
end
function weight(w::LikelihoodFreeWeight{M,U,Y}, mm::AbstractArray{M,1}, u::U)::NamedTuple where M where U where Y
    out = Array{Float64,1}(undef, length(mm))
    save = weight!(out, w, mm, u)
    return merge((ww = out,), save)
end

function weight!(out::AbstractArray{Float64,1}, w::LikelihoodFreeWeight{M,U,Y}, mm::AbstractArray{M}, u::U)::NamedTuple where M where U where Y
    yy = Array{Y,2}(undef, length(mm), w.K)
    save1 = simulate!(yy, w.F, mm, u)
    save2 = compare!(out, w.C, u, yy)
    return merge((yy=yy,), save1, save2)
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