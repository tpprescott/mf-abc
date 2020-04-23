module LikelihoodFree

using IndexedTables
export select

export AbstractModel, AbstractGenerator, AbstractWeight
# The model contains the parameters that will be inferred
AbstractModel = NamedTuple
# The generator will generate the model according to some distribution (rejection sampling or importance sampling)
abstract type AbstractGenerator{M<:AbstractModel} end
# The weight is essentially a likelihood estimate
abstract type AbstractWeight end

LHFun = Tuple{AbstractWeight, X} where X
LHSet = NTuple{N, LHFun} where N

export MonteCarloProposal
struct MonteCarloProposal{Θ<:AbstractModel, Π<:AbstractGenerator{Θ}, Q<:AbstractGenerator{Θ}, W<:LHSet}
    prior::Π
    q::Q
    lh_set::W
end
MonteCarloProposal(prior::AbstractGenerator{Θ}, q::AbstractGenerator{Θ}, lh_fun::LHFun...) where Θ = MonteCarloProposal(prior, q, lh_fun)
MonteCarloProposal(prior::AbstractGenerator{Θ}, q::AbstractGenerator{Θ}, w::AbstractWeight, y_obs) where Θ = MonteCarloProposal(prior, q, (w, y_obs))
MonteCarloProposal(prior::AbstractGenerator{Θ}, args...) where Θ = MonteCarloProposal(prior, prior, args...)
function MonteCarloProposal(prior::AbstractGenerator{Θ}, q::AbstractGenerator{Θ}, args...) where Θ end

function (Σ::MonteCarloProposal)(i=1; kwargs...)
    a::NamedTuple{(:θ, :logq, :logp)} = rand(Σ.q; prior=Σ.prior, kwargs...)
    b = weight(Σ.lh_set; a..., kwargs...)
    logw = 0.0
    for b_i in b
        logw += b_i[:logw]
    end
    return merge(a, (logw = logw, weight_components = b))
end

include("generate_parameters.jl")
include("sampling_procedures.jl")

# Need to apply a Monte Carlo weight to the generated parameters 
export weight
# Fallback: if only defining a weight, implement (w::AbstractWeight)(yobs; kwargs...)
function weight(w::AbstractWeight, yobs; kwargs...)::NamedTuple
    return w(yobs; kwargs...)
end
weight(lh_fun::LHFun; kwargs...) = (weight(lh_fun...; kwargs...),)
weight(lh_fun::LHFun, lh_set_remainder::LHFun...; kwargs...) = (weight(lh_fun...; kwargs...), weight(lh_set_remainder...; kwargs...)...)
weight(lh_set::LHSet; kwargs...) = weight(lh_set...; kwargs...)

######### Now go likelihood-free: a subtype of AbstractWeight that will be dependent on a simulation
# Here defined for a single fidelity
export LikelihoodFreeWeight, AbstractSimulator, AbstractComparison
abstract type AbstractSimulator end
abstract type AbstractComparison end

struct LikelihoodFreeWeight{Simulator<:AbstractSimulator, Comparison<:AbstractComparison} <: AbstractWeight
    F::Simulator    # Simulator
    C::Comparison   # Comparison
    K::Int64        # Number of simulation replicates
end
export output_dimension
output_dimension(::AbstractSimulator) = error("Need to specify the output dimension of the simulator: import the function output_dimension")

# Likelihood free weight combines simulation with comparison
export simulate, compare

function (w::LikelihoodFreeWeight)(y_obs; kwargs...)
    simulation_out::NamedTuple = simulate(w.F, w.K*length(y_obs); kwargs...)
    compare_out::NamedTuple = compare(w.C, y_obs; simulation_out..., kwargs...)
    return merge(compare_out, simulation_out)
end

# Fallback: implement (F::AbstractSimulator)(; parameters..., noise..., other_kwargs...)
function simulate(F::AbstractSimulator, numReplicates::Int64=1; θ::AbstractModel, kwargs...)
    t = table(F.(1:numReplicates; θ..., kwargs...))
    return columns(t)
end
function (F::AbstractSimulator)(i::Int64; kwargs...)
    return F(; kwargs...)
end

# Fallback: implement (C::AbstractComparison)(y_obs, y; kwargs...)
function compare(C::AbstractComparison, y_obs::Array{Array{T,1},1}; y::Array{Array{T,1},1}, kwargs...) where T
    return C(y_obs, y; kwargs...)
end

end