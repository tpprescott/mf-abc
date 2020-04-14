module LikelihoodFree

export AbstractModel, AbstractGenerator, AbstractWeight
# The model contains the parameters that will be inferred
AbstractModel = NamedTuple
# The generator will generate the model according to some distribution (rejection sampling or importance sampling)
abstract type AbstractGenerator{M<:AbstractModel} end
# The weighter is essentially a likelihood estimate where Y is the data's type
abstract type AbstractWeight{M<:AbstractModel} end

include("generate_parameters.jl")
include("sampling_procedures.jl")

# Need to apply a Monte Carlo weight to the generated parameters 
export weight
function weight(w::AbstractWeight{M}, m::M, y_obs)::NamedTuple{(:w,), Tuple{Float64}} where M where T
    return (w = w(m, y_obs),)
end
function weight(w::AbstractWeight{M}, mm::AbstractArray{M,1}, y_obs)::NamedTuple where M
    ww = Array{Float64,1}(undef, length(mm))
    save = weight!(ww, w, mm, y_obs)
    return merge((ww=ww,), save)
end
function weight!(ww::AbstractArray{Float64,1}, w::AbstractWeight{M}, mm::AbstractArray{M,1}, y_obs)::NamedTuple where M
    for i in eachindex(mm)
        ww[i] = weight(w, mm[i], y_obs)[:w]
    end
    return NamedTuple()
end


######### Now go likelihood-free: a subtype of AbstractWeight that will be dependent on a simulation
# Here defined for a single fidelity
export LikelihoodFreeWeight, AbstractSimulator, AbstractComparison
abstract type AbstractSimulator{M<:AbstractModel, T} end
abstract type AbstractComparison{T} end

struct LikelihoodFreeWeight{M, T, TF<:AbstractSimulator{M, T}, TC<:AbstractComparison{T}} <: AbstractWeight{M}
    F::TF       # Simulator
    C::TC       # Comparison
    K::Int64    # Number of simulation replicates
end
export output_dimension
output_dimension(::AbstractSimulator) = error("Need to specify the output dimension of the simulator")

# Likelihood free weight combines simulation with comparison
# Simulation array is always a 4D tensor: DimSummaryStatistics x NumParameters x SimulationReplicates x NumDataPoints 

# Single parameter means a single weight output
function weight(w::LikelihoodFreeWeight{M,T}, m::M, y_obs)::NamedTuple where M where T
    yy = Array{T, 4}(undef, size(y_obs,1), 1, w.K, size(y_obs,2))
    save_sim = simulate!(yy, w.F, m)
    out = compare(w.C, yy, y_obs)
    return merge(out, (y=yy,), save_sim)
end

# Many parameters means many weights output
function weight!(ww::AbstractArray{Float64, 1}, w::LikelihoodFreeWeight{M,T}, mm::AbstractArray{M,1}, y_obs) where M where T
    yy = Array{T, 4}(undef, size(y_obs,1), length(mm), w.K, size(y_obs, 2))
    save_sim = simulate!(yy, w.F, mm)
    save_compare = compare!(ww, w.C, yy, y_obs)
    return merge((yy=yy,), save_sim, save_compare)
end

export simulate, compare, simulate!, compare!

# NB simulation is into a preallocated array: need to define (F::AbstractSimulator)(y, m) as an inplace mutation of y::Array{T,1}
function simulate!(yy::AbstractArray{T,4}, F::AbstractSimulator{M,T}, m::M) where T where M
    size(yy, 2)==1 || error("Wrong sized preallocated array: only one parameter to be simulated")
    yy_collect_replicates = reshape(yy, size(yy,1), :)
    out = simulate!(yy_collect_replicates, F, m)
    return out
end
function simulate!(yy::AbstractArray{T,4}, F::AbstractSimulator{M,T}, mm::AbstractArray{M,1}) where T where M
    numParameters = length(mm)
    size(yy, 2)==numParameters || error("Wrong sized pre-allocated array: $numParameters parameters to be simulated")

    yy_collect_replicates = reshape(yy, size(yy,1), size(yy,2), :)
    for nParameter in axes(yy_collect_replicates,2)
        simulate!(view(yy_collect_replicates, :, nParameter, :), F, mm[nParameter])
    end
    return NamedTuple()
end
function simulate!(y::AbstractArray{T,2}, F::AbstractSimulator{M,T}, m::M) where T where M
    for nReplicate in axes(y,2)
        F(view(y, :, nReplicate), m)
    end
    return NamedTuple()
end

function simulate(F::AbstractSimulator{M,T}, m::M, K::Int64=1) where T where M
    dim = output_dimension(F)
    y = Array{T,2}(undef, dim, K)
    saved = simulate!(y, F, m)
    return merge((y=y,), saved)
end
function simulate(F::AbstractSimulator{M,T}, mm::AbstractArray{M,1}, K::Int64=1) where T where M
    dim = output_dimension(F)
    yy = Array{T,4}(undef, dim, length(mm), K, 1)
    saved = simulate!(yy, F, mm)
    return merge((yy=yy,), saved)
end


function compare(C::AbstractComparison{T}, yy::AbstractArray{T,2}, y_obs) where T
    return (w = C(yy, y_obs),)
end

function compare(C::AbstractComparison{T}, yy::AbstractArray{T,4}, y_obs) where T
    size(yy,4)==size(y_obs,2) || error("Mismatch in dimensions between number of data points and number of simulations")
    if size(yy,2) == 1 # Single weight required
        if size(y_obs,2) == 1
            out = compare(C, view(yy, :, 1, :, 1), y_obs)
            return out
        else
            w = 1.0
            for nDataPoint in axes(y_obs,2)
                if !iszero(w)
                    w *= compare(C, view(yy, :, 1, :, nDataPoint), view(y_obs, :, nDataPoint))[:w]
                end
            end
            return (w=w,)
        end
    else
        ww = Array{Float64,1}(undef, size(yy,2))
        save = compare!(ww, C, yy, y_obs)
        return merge((ww=ww,), save)
    end
end

function compare!(ww::AbstractArray{Float64,1}, C::AbstractComparison{T}, yy::AbstractArray{T,4}, y_obs) where T
    size(yy,4)==size(y_obs,2) || error("Mismatch in dimensions between number of data points and number of simulations")
    for nParameter in eachindex(ww)
        ww[nParameter] = 1.0
        for nDataPoint in axes(yy,4)
            if !iszero(ww[nParameter])
                ww[nParameter] *= compare(C, view(yy, :, nParameter, :, nDataPoint), view(y_obs, :, nDataPoint))[:w]
            end
        end
    end
    return NamedTuple()
end

end