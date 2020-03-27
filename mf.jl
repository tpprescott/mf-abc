module MFABC
using ..LikelihoodFree
using ..ABC

# Multifidelity: for coupling, we require a new abstract type of simulator that also includes a noise process
include("mf_coupledsimulator.jl")

# Multifidelity algorithm requires a concrete type of output and a concrete type of simulator and a concrete type of weight
using SparseArrays
using LightGraphs: SimpleDiGraph, attracting_components

export MFLikelihoodFreeWeight, MFABCWeight
using Distributions: Categorical
export Categorical

struct MFLikelihoodFreeWeight{M, U, Nv, WW<:NTuple{Nv, LikelihoodFreeWeight{M,U}}} <: AbstractWeight{M, U}
    nodes::WW
    F_init::Categorical
end
MFABCWeight{M,U,Nv,WW<:NTuple{Nv, ABCWeight{M,U}}} = MFLikelihoodFreeWeight{M,U,Nv,WW}

import .LikelihoodFree.weight
# Generic - not necessarily MFABC - not sure how to choose next_v
function weight(ww::MFLikelihoodFreeWeight{M, U}, m::M, u::U) where M where U
    v = rand(ww.F_init)
    cont = true
    eta = one(Float64)
    wt = zero(Float64)
    while cont
        wt += (weight(ww.nodes[v], m, u) - wt)/eta
        cont, v, eta = next_v(v, m, u)
    end
    return wt
end

next_v(v,m,u) = false, v+1, 1.0

end

# mutable struct ZeroSharedNoise <: NoiseInput end
#=
mutable struct MFABCOutput{Nv, TY<:NTuple{Nv, SimOutput}, TN<:NTuple{Nv, NoiseInput}} <: AbstractSummaryStatisticSpace
    y::TY # Output tuple (for all nodes)
    n::TN # Noise tuple (for all nodes)
    seq::Vector{Int64}
    eta::Vector{Float64}
    function MFABCOutput{Nv, TY, TN}() where Nv where TY where TN
        y = TY()
        n = TN()
        seq = Vector{Int64}()
        eta = Vector{Float64}()
        new{Nv, TY, TN}(y, n, seq, eta)
    end
end
function (::Type{T})()::T where T<:NTuple{Nv, SimOutput} where Nv
     return Tuple(Y() for Y in T.parameters)
end
function (::Type{T})()::T where T<:NTuple{Nv, NoiseInput} where Nv
    return Tuple(N() for N in T.parameters)
end

export NoiseInput, Coupling, MFABCSimulatorNode
# Node is a simulator that also has a specific noise that we keep track of
abstract type MFABCSimulatorNode{M, U, Y, W<:NoiseInput} <: Simulator{M,U,Y} end
noise_type(::MFABCSimulatorNode{M,U,Y,W}) where M where U where Y where W = W

export AbstractContinuationProbability
abstract type AbstractContinuationProbability{U<:Experiment, Y<:SimOutput} end
include("continuation_probability.jl")

export ContinuationDistribution
struct ContinuationDistribution{T<:AbstractContinuationProbability}
    # Code will be slow if there are a mixture of different continuation probability types in the graph
    g::SparseMatrixCSC{T, Int64}
    function ContinuationDistribution(g::SparseMatrixCSC{T,Int64}) where T
        return new{T}(g)
    end
    function ContinuationDistribution(src_arr::AbstractVector{Int64}, dst_arr::AbstractVector{Int64}, eta_i::AbstractVector{T}) where T
        n = max(maximum(src_arr), maximum(dst_arr))
        return new{T}(sparse(dst_arr, src_arr, eta_i,n,n))
    end
    function ContinuationDistribution(eta::T) where T
        return new{T}(sparse([2],[1],[eta],2,2))
    end
end

using Distributions:isprobvec
struct MFABCSimulator{M, U, Y, Nv, TF<:NTuple{Nv, MFABCSimulatorNode{M,U}}, TH<:ContinuationDistribution} <: Simulator{M, U, Y}
    F::TF
    H::TH
    CDF_init::Vector{Float64}

    function MFABCSimulator(
        HiFi::MFABCSimulatorNode{M,U},
        LoFi::NTuple{N, MFABCSimulatorNode{M,U}}, 
        H::TH, 
        F_init::Vector{Float64},
        ) where M where U where N where TH<:ContinuationDistribution
        
        # Check sizes
        Nv = N+1
        H.g.n == Nv || error("Incorrect graph size")
        len_F_init = length(F_init)
        len_F_init <= N || error("Too many initial probabilities")
        append!(F_init, zeros(Nv-len_F_init))
        F_init[end] = 1-sum(F_init)
        isprobvec(F_init) || error("Invalid initial probabilities")
        cumsum!(F_init, F_init)
        
        # Check that node Nv (HiFi) is the only absorbing node
        dst, src, _ = findnz(H.g)
        graph = SimpleDiGraph(sparse(src, dst, ones(size(src)), Nv, Nv))
        attracting_components(graph)==[[Nv]] || error("Invalid graph: need HiFi to be the only attracting node")

        # Sort out the types
        OutputSpace = Tuple{output_type.(LoFi)..., output_type(HiFi)}
        NoiseSpace = Tuple{noise_type.(LoFi)..., noise_type(HiFi)}
        Y = MFABCOutput{Nv, OutputSpace, NoiseSpace}
        sim_list = (LoFi..., HiFi)
        TF = typeof(sim_list)

        # Return
        return new{M, U, Y, Nv, TF, TH}(sim_list, H, F_init)
    end
end
function MFABCSimulator(HiFi::MFABCSimulatorNode{M,U}, LoFi::NTuple{N,MFABCSimulatorNode{M,U}}, H::TH; F_init::Vector{Float64}=[1.0]) where M where U where N where TH
    return MFABCSimulator(HiFi, LoFi, H, F_init)
end
function MFABCSimulator(HiFi::MFABCSimulatorNode{M,U}, LoFi::MFABCSimulatorNode{M,U}, eta::T; F_init::Vector{Float64}=[1.0]) where M where U where T
    return MFABCSimulator(HiFi, (LoFi,), ContinuationDistribution(eta); F_init=F_init)
end
function MFABCSimulator(HiFi::MFABCSimulatorNode{M,U}, LoFi::NTuple{N, MFABCSimulatorNode{M,U}}, eta::NTuple{N, AbstractContinuationProbability}; type::Symbol, F_init::AbstractVector{Float64}=[1.0]) where M where U where N
    in(type, (:path, :tree)) || error("Need either :path or :tree")
    eta_vec = [eta_i for eta_i in eta]
    if type==:path
        src = collect(1:N)
        dst = src .+ 1
    elseif type==:tree
        src = collect(1:N)
        dst = fill(N+1, N)
    end
    H = ContinuationDistribution(src, dst, eta_vec)
    return MFABCSimulator(HiFi, LoFi, H; F_init = F_init)
end

#=
include("mfabc_simulate.jl")
include("mfabc_weight.jl")

import .LikelihoodFree.sample
export sample

function sample(
    q::Generator{M},
    F::MFABCSimulator{M,U,Y,Nv},
    u::U,
    d::D,
    eps::Float64,
    N::Int64;
    kwargs...
)::MCOutput{M} where U where Y where M where D where Nv
    
    return sample(q, F, u, d, fill(eps, Nv), N; kwargs...)
end
function sample(
    q::Generator{M},
    F::MFABCSimulator{M,U,Y},
    u::U,
    d::D,
    eps::Vector{Float64},
    N::Int64;
    kwargs...
)::MCOutput{M} where U where Y where M where D
    
    w = MFABCWeight(eps, d)
    return sample(q, F, u, w, N; kwargs...)
end
=#
end

=#