module LikelihoodFree

using IndexedTables
export select

export AbstractModel, AbstractGenerator, AbstractLikelihoodFunction
# The model contains the parameters that will be inferred
AbstractModel = NamedTuple
# The generator will generate the model according to some distribution (rejection sampling or importance sampling)
abstract type AbstractGenerator{M<:AbstractModel} end
# The weight is essentially a likelihood estimate
# Need to implement L(θ, args...) to return a likelihood function, based on the template L, that is conditioned on theta and any other arguments
# Need to implement L(y_obs; log::Bool=false) to return a weight (or log weight if required)
abstract type AbstractLikelihoodFunction end
LikelihoodObservationPair = Tuple{AbstractLikelihoodFunction, X} where X
LikelihoodObservationSet = Tuple{NTuple{N, AbstractLikelihoodFunction}, NTuple{N, X where X}} where N

export likelihood

function _condition(L::AbstractLikelihoodFunction, θ::AbstractModel, L_past::AbstractLikelihoodFunction...; kwargs...)::AbstractLikelihoodFunction
    return L(θ, L_past...; kwargs...)
end
function _likelihood(L::AbstractLikelihoodFunction, y_obs; loglikelihood::Bool=false, kwargs...)::Float64
    return L(y_obs; loglikelihood=loglikelihood, kwargs...)
end

function likelihood(L::AbstractLikelihoodFunction, y_obs, θ::AbstractModel, L_past::AbstractLikelihoodFunction...; loglikelihood::Bool=false, kwargs...)
    L_θ = _condition(L, θ, L_past...; kwargs...)
    w = _likelihood(L_θ, y_obs; loglikelihood=loglikelihood, kwargs...)
    return loglikelihood ? (logw = w, L = L_θ) : (w = w, L = L_θ)
end
likelihood(lh_obs::LikelihoodObservationPair, θ::AbstractModel, L_past::AbstractLikelihoodFunction...; kwargs...) = likelihood(lh_obs..., θ, L_past...; kwargs...)
function likelihood(lh_obs_set::LikelihoodObservationSet, θ::AbstractModel; loglikelihood::Bool=false, kwargs...) 
    y_obs_set = lh_obs_set[2]
    
    L_θ_set = _condition.(lh_obs_set[1], Ref(θ); kwargs...)
    w = _likelihood.(L_θ_set, y_obs_set; loglikelihood=loglikelihood, kwargs...)
    return loglikelihood ? (logw = sum(w), w_components = w, L = L_θ_set) : (w = prod(w), w_components = w, L = L_θ_set)
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

export output_dimension
output_dimension(::AbstractSimulator) = error("Need to specify the output dimension of the simulator: import the function output_dimension")

# Likelihood free weight combines simulation with comparison
export simulate

# Fallback: implement (F::AbstractSimulator)(; parameters..., noise..., other_kwargs...)
function simulate(F::AbstractSimulator, numReplicates::Int64=1; θ::AbstractModel, kwargs...)
    t = table(F.(1:numReplicates; θ..., kwargs...))
    return columns(t)
end
function (F::AbstractSimulator)(i::Int64; kwargs...)
    return F(; kwargs...)
end

include("abc.jl")
include("synthetic_bayes.jl")

end