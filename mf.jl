module MFABC
using ..LikelihoodFree
using ..ABC

# Multifidelity algorithm requires a concrete type of output and a concrete type of simulator and a concrete type of weight
using SparseArrays
using LightGraphs: SimpleDiGraph, attracting_components


export AbstractContinuationProbability
abstract type AbstractContinuationProbability{M,U,Y} end
include("continuation_probability.jl")

export ContinuationDistribution
struct ContinuationDistribution{T<:AbstractContinuationProbability}
    # Code will be slow if there are a mixture of different continuation probability types in the graph
    g::SparseMatrixCSC{T, Int64}
    function ContinuationDistribution(g::SparseMatrixCSC{T,Int64}) where T<:AbstractContinuationProbability
        return new{T}(g)
    end
    function ContinuationDistribution(
        src_arr::AbstractVector{Int64}, 
        dst_arr::AbstractVector{Int64}, 
        eta_i::AbstractVector{T},
    ) where T<:AbstractContinuationProbability
        n = max(maximum(src_arr), maximum(dst_arr))
        return new{T}(sparse(dst_arr, src_arr, eta_i,n,n))
    end
    function ContinuationDistribution(eta::T) where T<:AbstractContinuationProbability
        return new{T}(sparse([2],[1],[eta],2,2))
    end
end

# Multifidelity: for coupling, we require a new abstract type of simulator that also includes a noise process
include("mf_coupledsimulator.jl")


export MFABCWeight
using Distributions: Categorical
export Categorical

struct MFABCWeight{M, U, Nv, WW<:NTuple{Nv, ABCWeight{M,U}}, EE<:ContinuationDistribution} <: AbstractWeight{M, U}
    nodes::WW
    edges::EE
    F_init::Categorical
    function MFABCWeight(nodes::WW, edges::EE, F_init::Categorical) where WW <: NTuple{Nv, ABCWeight{M,U}} where Nv where M where U where EE<:ContinuationDistribution
                
        # Check graph is the right size
        edges.g.n==edges.g.m==Nv || error("Wrong sized graph: size $Nv nodes vs $(edges.g.m) by $(edges.g.n) edges")

        # Check initial distribution is right size
        length(F_init.p)==Nv || error("Stupid initial distribution")

        # Check the final node is the only absorbing node (all paths will lead to the high fidelity simulation)
        dst, src, _ = findnz(edges.g)
        graph = SimpleDiGraph(sparse(src, dst, ones(size(src)), Nv, Nv))
        attracting_components(graph)==[[Nv]] || error("Invalid graph: need HiFi to be the only attracting node")
        
        # Carry on
        return new{M, U, Nv, WW, EE}(nodes, edges, F_init)
    end
end

function MFABCWeight(
    HiFi::ABCWeight{M,U}, 
    LoFi::NTuple{N, ABCWeight{M,U}}, 
    eta::NTuple{N, AbstractContinuationProbability}, 
    F_init::Categorical; 
    type::Symbol
) where M where U where N
    nodes = (LoFi..., HiFi)
    eta_vec = [eta_i for eta_i in eta]
    src = collect(1:N)
    if type==:path
        dst = src .+ 1
    elseif type==:tree
        dst = fill(N+1, N)
    else
        error("Specify a graph type (:path or :tree) or another connecting graph")
    end
    edges = ContinuationDistribution(src, dst, eta_vec)
    return MFABCWeight(nodes, edges, F_init)
end

function MFABCWeight(
    HiFi::ABCWeight{M,U},
    LoFi::ABCWeight{M,U},
    eta::AbstractContinuationProbability,
) where M where U
    return MFABCWeight((LoFi, HiFi), ContinuationDistribution(eta), Categorical([1.0, 0.0]))
end

import Base.Iterators

mutable struct MFABCIterator{M, U, Nv, MF_W <: MFABCWeight{M,U,Nv}}
    mf_w::MF_W
    m::M
    u::U
end
IteratorSize(::Type{I}) where I <: MFABCIterator = SizeUnknown()
eltype(::Type{I}) where I <: MFABCIterator = NamedTuple

# Sort out how to sort out the sequence of iterations
include("next_simulation.jl")

function weight(mf_w::MFABCWeight{M, U}, m::M, u::U) where M where U
    iter = MFABCIterator(mf_w, m, u)
    return weight(iter)
end
function weight(iter::MFABCIterator{M, U, Nv}) where M where U where Nv
    wt = zero(Float64)
    d = Array{Float64,1}()
    v = Array{Int64,1}()
    eta = Array{Float64, 1}()
    for node in iter
        wt += (node[:w] - wt)/node[:eta]
        append!(d, node[:d])
        append!(v, node[:v])
        append!(eta, node[:eta])
    end
    return (w = wt, v=v, d=d, eta=eta)
end

function weight!(ww::AbstractArray{Float64}, mf_w::MFABCWeight{M, U}, mm::Array{M}, u::U) where M where U
    iter = MFABCIterator(mf_w, mm[1], u)
    dd = Array{Array{Float64,1}}(undef, size(mm))
    seq = Array{Array{Int64,1}}(undef, size(mm))
    eta = Array{Array{Float64,1}}(undef, size(mm))
    for i in eachindex(mm)
        iter.m = mm[i]
        out = weight(iter)
        ww[i] = out[:w]
        dd[i] = out[:d]
        seq[i] = out[:v]
        eta[i] = out[:eta]
    end
    return (dd=dd, seq=seq, eta=eta)
end

function weight(mf_w::MFABCWeight{M, U}, mm::Array{M}, u::U) where M where U
    iter = MFABCIterator(mf_w, mm[1], u)
    ww = Array{Float64}(undef, size(mm))
    save = weight!(ww, mf_w, mm, u)
    return merge((ww=ww,), save)
end



end
