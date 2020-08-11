module LikelihoodFree

using IndexedTables, Distributed, ProgressMeter, JLD, InvertedIndices
import Distributions.loglikelihood

export select, merge, ndims, domain, domains
import Base.merge, Base.merge_types, Base.merge_names

export AbstractModel, AbstractGenerator, AbstractLikelihoodFunction
# The model contains the parameters that will be inferred
AbstractModel = NamedTuple
function merge(a::Type{A}, b::Type{B}) where A<:AbstractModel where B<:AbstractModel
    names = merge_names(fieldnames(a), fieldnames(b))
    types = merge_types(names, a, b)
    return NamedTuple{names, types}
end
merge(a,b,c...) = merge(merge(a,b), c...)
merge(a) = a
function marginalise(a::AbstractModel, ::Type{Θ})::Θ where Θ<:AbstractModel
    return Θ(get.(Ref(a), fieldnames(Θ), nothing))
end
import Base.ndims
ndims(::Type{T}) where T<:AbstractModel = length(T.names)
ndims(::T) where T<:AbstractModel = ndims(T)

# The generator will generate the model according to some distribution (rejection sampling or importance sampling)
abstract type AbstractGenerator{Θ<:AbstractModel} end
domain(::Type{T}) where T<:AbstractGenerator{Θ} where Θ = Θ
domain(q::AbstractGenerator) = domain(typeof(q))
domains(q) = domain(q)
ndims(::Type{T}) where T<:AbstractGenerator = ndims(domain(T))
ndims(q::AbstractGenerator) = ndims(domain(q))

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
    test_idx::AbstractArray{Int64,1} = Array{Int64,1}(),
    kwargs...
)::NamedTuple{(:w, :ww, :test, :L), Tuple{Float64, Tuple{Float64}, Tuple{Float64}, TL}} where TL<:AbstractLikelihoodFunction where TX

    L_θ = _condition(L, θ, L_past...; kwargs...)
    w = _likelihood(L_θ, y_obs[InvertedIndices.Not(test_idx)]; kwargs...)
    test = _likelihood(L_θ, y_obs[test_idx]; kwargs...)
    return (w = w, ww = (w,), test = (test,), L = L_θ)
end

function loglikelihood(
    (L, y_obs)::LikelihoodObservationPair{TL, TX}, 
    θ::AbstractModel, 
    L_past...; 
    test_idx::AbstractArray{Int64,1} = Array{Int64,1}(),
    kwargs...
)::NamedTuple{(:logw, :logww, :logtest, :L), Tuple{Float64, Tuple{Float64}, Tuple{Float64}, TL}} where TL<:AbstractLikelihoodFunction where TX

    L_θ = _condition(L, θ, L_past...; kwargs...)
    logw = _loglikelihood(L_θ, y_obs[InvertedIndices.Not(test_idx)]; kwargs...)
    logtest = _loglikelihood(L_θ, y_obs[test_idx]; kwargs...)
    return (logw = logw, logww = (logw,), logtest = (logtest,), L = L_θ)
end

function likelihood(
    (L, y_obs)::LikelihoodObservationSet{N, TLL, TYY}, 
    θ::AbstractModel,
    L_past...; 
    test_idx::NTuple{N, AbstractArray{Int64,1}} = Tuple(fill(Array{Int64,1}(),N)),
    kwargs...
)::NamedTuple{(:w, :ww, :test, :L), Tuple{Float64, NTuple{N, Float64}, NTuple{N, Float64}, TLL}} where TYY where TLL<:NTuple{N, AbstractLikelihoodFunction} where N

    L_θ_set = _condition.(L, Ref(θ), L_past...; kwargs...)
    ww = _likelihood.(L_θ_set, getindex.(y_obs, InvertedIndices.Not.(test_idx)); kwargs...)
    test = _likelihood.(L_θ_set, getindex.(y_obs, test_idx); kwargs...)
    return (w = prod(ww), ww = ww, test = test, L = L_θ_set)
end

function loglikelihood(
    (L, y_obs)::LikelihoodObservationSet{N, TLL, TYY}, 
    θ::AbstractModel,
    L_past...; 
    test_idx::NTuple{N, AbstractArray{Int64,1}} = Tuple(fill(Array{Int64,1}(), N)),
    kwargs...
)::NamedTuple{(:logw, :logww, :logtest, :L), Tuple{Float64, NTuple{N, Float64}, NTuple{N, Float64}, TLL}} where TYY where TLL<:NTuple{N, AbstractLikelihoodFunction} where N

    L_θ_set = _condition.(L, Ref(θ), L_past...; kwargs...)
    logww = _loglikelihood.(L_θ_set, getindex.(y_obs, InvertedIndices.Not.(test_idx)); kwargs...)
    logtest = _loglikelihood.(L_θ_set, getindex.(y_obs, test_idx); kwargs...)
    return (logw = sum(logww), logww = logww, logtest = logtest, L = L_θ_set)
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
function simulate(
    f::F, 
    θ::AbstractModel; 
    numReplicates::Int64, 
    kwargs...,
)::Array{eltype(F),1} where F<:AbstractSimulator

    R = f(numReplicates; θ..., kwargs...)
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
)::Array{eltype(F),1} where F<:AbstractSimulator where T<:NamedTuple
    
    K = length(y0)
    numReplicates = max(numReplicates, numIndependent)
    numCoupled = min(numReplicates-numIndependent, K)
    numIndependent = numReplicates-numCoupled

    iszero(numCoupled) && (return simulate(f, θ; numReplicates=numReplicates, kwargs...))
    
    shuffle!(y0)
    

    ind_flags_1 = fill(false, numCoupled)
    ind_flags_2 = fill(true, numIndependent)
    ind_flags = vcat(ind_flags_1, ind_flags_2)

    R = f(numReplicates; columns(Columns(y0))..., θ..., ind_flags=ind_flags, kwargs...)
    return R
end
simulate(f::F, θ::AbstractModel, L0::LikelihoodFreeLikelihoodFunction{F}; kwargs...) where F = simulate(f, θ, get_simulations(L0); kwargs...)

include("abc.jl")
include("synthetic_bayes.jl")


export save_sample, load_sample
function save_sample(fn::String, t::Array{<:IndexedTable,1})
    θ = make_array.(select.(t, :θ))
    w = select.(t, :weight)
    logww = make_array.(select.(t, :logww))
    logtest = make_array.(select.(t, :logtest))
    logp = select.(t, :logp)
    logq = select.(t, :logq)
    save(fn, 
        "θ", θ, 
        "w", w, 
        "logww", logww, 
        "logp", logp, 
        "logq", logq, 
        "logtest", logtest,
    )
    println("Success! Saved to $(fn)")
    return nothing
end

function make_unarray(x::Array{X, 1}, ::Type{Θ}) where X where Θ<:AbstractModel
    return [Θ(x_i) for x_i in x]
end
function make_unarray(x::Array{X, 2}, ::Type{Θ}) where X where Θ<:AbstractModel
    return [Θ(selectdim(x, 2, n)) for n in 1:size(x, 2)]
end
function make_unarray(x::Array{X, 1}) where X
    return [Tuple(x_i) for x_i in x]
end
function make_unarray(x::Array{X, 2}) where X
    return [Tuple(selectdim(x, 2, n)) for n in 1:size(x, 2)]
end

function load_sample(fn::String, ::Type{Θ}) where Θ<:AbstractModel
    data = load(fn)
    N = length(data["θ"])
    t = map(
        i-> table((
                θ = make_unarray(data["θ"][i], Θ),
                logp = data["logp"][i],
                logq = data["logq"][i],
                logww = make_unarray(data["logww"][i]),
                logtest = make_unarray(data["logtest"][i]),
                weight = data["w"][i],
        )), 
        1:N)
    println("Success! Loaded $(fn)")
    return t
end

export ESS
ESS(w) = sum(w)^2/sum(w.^2)
ESS(t::IndexedTable) = ESS(select(t, :weight))


end
