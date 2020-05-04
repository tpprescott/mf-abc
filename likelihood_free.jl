module LikelihoodFree

using IndexedTables, Distributed, ProgressMeter
export select

export AbstractModel, AbstractGenerator, AbstractLikelihoodFunction
# The model contains the parameters that will be inferred
AbstractModel = NamedTuple
# The generator will generate the model according to some distribution (rejection sampling or importance sampling)
abstract type AbstractGenerator{M<:AbstractModel} end

abstract type AbstractLikelihoodFunction end
LikelihoodObservationPair{TL, TX} = Tuple{TL, TX} where TX where TL<:AbstractLikelihoodFunction
LikelihoodObservationSet{N, TLL, TXX} = Tuple{TLL, TXX} where TXX<:NTuple{N, TX where TX} where TLL<:NTuple{N, TL where TL<:AbstractLikelihoodFunction} where N

export likelihood, loglikelihood

# Need to implement L(θ, args...) to return a conditioned likelihood function based on the template L, that is conditioned on θ and any other arguments
function _condition(L::TL, θ::AbstractModel, L_past...; kwargs...)::TL where TL<:AbstractLikelihoodFunction
    return L(θ, L_past...; kwargs...)
end
# Need to implement L(y_obs; loglikelihood::Bool=false) to return a weight (or log weight if required)
function _likelihood(L::AbstractLikelihoodFunction, y_obs; kwargs...)::Float64
    return L(y_obs; loglikelihood=false, kwargs...)
end
function _loglikelihood(L::AbstractLikelihoodFunction, y_obs; kwargs...)::Float64
    return L(y_obs; loglikelihood=true, kwargs...)
end

function likelihood(
    (L, y_obs)::LikelihoodObservationPair{TL, TX},
    θ::AbstractModel,
    L_past...;
    kwargs...
)::NamedTuple{(:w, :L), Tuple{Float64, TL}} where TL<:AbstractLikelihoodFunction where TX

    L_θ = _condition(L, θ, L_past...; kwargs...)
    w = _likelihood(L_θ, y_obs; kwargs...)
    return (w = w, L = L_θ)
end

function loglikelihood(
    (L, y_obs)::LikelihoodObservationPair{TL, TX}, 
    θ::AbstractModel, 
    L_past...; 
    kwargs...
)::NamedTuple{(:logw, :L), Tuple{Float64, TL}} where TL<:AbstractLikelihoodFunction where TX

    L_θ = _condition(L, θ, L_past...; kwargs...)
    w = _loglikelihood(L_θ, y_obs; kwargs...)
    return (logw = w, L = L_θ)
end

function likelihood(
    (L, y_obs)::LikelihoodObservationSet{N, TLL, TYY}, 
    θ::AbstractModel,
    L_past...; 
    kwargs...
)::NamedTuple{(:w, :ww, :L), Tuple{Float64, NTuple{N, Float64}, TLL}} where TYY where TLL<:NTuple{N, AbstractLikelihoodFunction} where N

    L_θ_set = _condition.(L, Ref(θ), L_past...; kwargs...)
    ww = _likelihood.(L_θ_set, y_obs; kwargs...)
    return (w = prod(ww), ww = ww, L = L_θ_set)
end

function loglikelihood(
    (L, y_obs)::LikelihoodObservationSet{N, TLL, TYY}, 
    θ::AbstractModel,
    L_past...; 
    kwargs...
)::NamedTuple{(:logw, :logww, :L), Tuple{Float64, NTuple{N, Float64}, TLL}} where TYY where TLL<:NTuple{N, AbstractLikelihoodFunction} where N

    L_θ_set = _condition.(L, Ref(θ), L_past...; kwargs...)
    logww = _loglikelihood.(L_θ_set, y_obs; kwargs...)
    return (logw = sum(logww), logww = logww, L = L_θ_set)
end


include("generate_parameters.jl")
include("sampling_procedures.jl")

# Need to apply a Monte Carlo weight to the generated parameters 
# Fallback: if only defining a weight, implement (w::AbstractWeight)(yobs; kwargs...)

######### Now go likelihood-free: a subtype of AbstractWeight that will be dependent on a simulation
# Here defined for a single fidelity
export LikelihoodFreeLikelihoodFunction, AbstractSimulator
abstract type AbstractSimulator end
abstract type LikelihoodFreeLikelihoodFunction{F<:AbstractSimulator} <: AbstractLikelihoodFunction end

# Likelihood free likelihood functions rely on simulations
# Default function to return the simulations queries the y field from L - import and redefine if necessary, else use y for simulation output
export get_simulations
function get_simulations(L::LikelihoodFreeLikelihoodFunction{F})::Array{eltype(F), 1} where F<:AbstractSimulator
    return L.y
end

# Likelihood free weight combines simulation with comparison
export simulate

# Fallback: implement (f::F)(; parameters..., noise..., other_kwargs...) where F<:AbstractSimulator
# Need to record the output as eltype(F) where F<:AbstractSimulator
function simulate(f::F, θ::AbstractModel; numReplicates::Int64, kwargs...)::Array{eltype(F),1} where F<:AbstractSimulator
    I_F = Iterators.repeated(f)
    I_θ = Iterators.repeated(θ)
    I_kw = Iterators.repeated(kwargs)
    I = Iterators.take(zip(I_F, I_θ, I_kw), numReplicates)

    R = @showprogress pmap(_simulate, I)
    return R
end

using Random
function simulate(
    f::F, 
    θ::AbstractModel, 
    y0::Array{T, 1};
    numReplicates::Int64,
    numIndependent::Int64=numReplicates,
    kwargs...
) where F<:AbstractSimulator where T<:NamedTuple
    
    K = length(y0)
    numReplicates = max(numReplicates, numIndependent)
    numCoupled = min(numReplicates-numIndependent, K)
    numIndependent = numReplicates-numCoupled
    
    I_F = Iterators.repeated(f)
    I_θ = Iterators.repeated(θ)
    I_kw = Iterators.repeated(kwargs)

    I_independent = Iterators.take(zip(I_F, I_θ, I_kw), numIndependent)
    shuffle!(y0)
    I_coupled = Iterators.take(zip(I_F, I_θ, I_kw, y0), numCoupled)

    R = Array{eltype(F), 1}(undef, numReplicates)
    R[1:numIndependent] = @showprogress "$θ \n $numIndependent independent: " pmap(_simulate, I_independent)
    R[numIndependent+1:end] = @showprogress " $numCoupled coupled: " pmap(_simulate, I_coupled)
    
    return R
end
simulate(f::F, θ::AbstractModel, L0::LikelihoodFreeLikelihoodFunction{F}; kwargs...) where F = simulate(f, θ, get_simulations(L0); kwargs...)

function _simulate((f, θ, kwargs)::Tuple{F, AbstractModel, Any})::eltype(F) where F<:AbstractSimulator
    return f(; θ..., kwargs...)
end
function _simulate((f, θ, kwargs, y)::Tuple{F, AbstractModel, Any, NamedTuple})::eltype(F) where F<:AbstractSimulator
    return f(; θ..., y..., kwargs...)
end

include("abc.jl")
include("synthetic_bayes.jl")

end
