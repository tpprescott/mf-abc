import StatsBase
import Plots
import Base.push!, Base.append!, Base.==


mutable struct Maxogram{T<:Real,N,E} <: StatsBase.AbstractHistogram{T,N,E}
    edges::E
    weights::Array{T,N}
    closed::Symbol
    isdensity::Bool
    function Maxogram{T,N,E}(edges::NTuple{N,AbstractArray}, weights::Array{T,N},
                              closed::Symbol, isdensity::Bool=false) where {T,N,E}
        closed == :right || closed == :left || error("closed must :left or :right")
        isdensity && !(T <: AbstractFloat) && error("Density maxogram must have float-type weights")
        StatsBase._edges_nbins(edges) == size(weights) || error("Maxogram edge vectors must be 1 longer than corresponding weight dimensions")
        new{T,N,E}(edges,weights,closed,isdensity)
    end
end

Maxogram(edges::NTuple{N,AbstractVector}, weights::AbstractArray{T,N},
          closed::Symbol=:left, isdensity::Bool=false) where {T,N} =
    Maxogram{T,N,typeof(edges)}(edges,weights,closed,isdensity)

Maxogram(edges::NTuple{N,AbstractVector}, ::Type{T}, closed::Symbol=:left,
          isdensity::Bool=false) where {T,N} =
    Maxogram(edges,zeros(T, StatsBase._edges_nbins(edges)...),closed,isdensity)

Maxogram(edges::NTuple{N,AbstractVector}, closed::Symbol=:left,
          isdensity::Bool=false) where {N} =
    Maxogram(edges,Int,closed,isdensity)

(==)(h1::Maxogram,h2::Maxogram) = (==)(h1.edges,h2.edges) && (==)(h1.weights,h2.weights) && (==)(h1.closed,h2.closed) && (==)(h1.isdensity,h2.isdensity)

binindex(h::Maxogram{T,N}, xs::NTuple{N,Real}) where {T,N} =
    map((edge, x) -> StatsBase._edge_binindex(edge, h.closed, x), h.edges, xs)

binvolume(h::Maxogram{T,N}, binidx::NTuple{N,Integer}) where {T,N} =
    binvolume(StatsBase._promote_edge_types(h.edges), h, binidx)

binvolume(::Type{V}, h::Maxogram{T,N}, binidx::NTuple{N,Integer}) where {V,T,N} =
    prod(map((edge, i) -> StatsBase._edge_binvolume(V, edge, i), h.edges, binidx))

Maxogram(edge::AbstractVector, weights::AbstractVector{T}, closed::Symbol=:left, isdensity::Bool=false) where {T} =
    Maxogram((edge,), weights, closed, isdensity)

Maxogram(edge::AbstractVector, ::Type{T}, closed::Symbol=:left, isdensity::Bool=false) where {T} =
    Maxogram((edge,), T, closed, isdensity)

Maxogram(edge::AbstractVector, closed::Symbol=:left, isdensity::Bool=false) =
    Maxogram((edge,), closed, isdensity)

fit(::Type{Maxogram{T}},v::AbstractVector, edg::AbstractVector; closed::Symbol=:left) where {T} =
    fit(Maxogram{T},(v,), (edg,), closed=closed)
fit(::Type{Maxogram{T}},v::AbstractVector; closed::Symbol=:left, nbins=StatsBase.sturges(length(v))) where {T} =
    fit(Maxogram{T},(v,); closed=closed, nbins=nbins)
fit(::Type{Maxogram{T}},v::AbstractVector, wv::StatsBase.AbstractWeights, edg::AbstractVector; closed::Symbol=:left) where {T} =
    fit(Maxogram{T},(v,), wv, (edg,), closed=closed)
fit(::Type{Maxogram{T}},v::AbstractVector, wv::StatsBase.AbstractWeights; closed::Symbol=:left, nbins=sturges(length(v))) where {T} =
    fit(Maxogram{T}, (v,), wv; closed=closed, nbins=nbins)

fit(::Type{Maxogram}, v::AbstractVector, wv::StatsBase.AbstractWeights{W}, args...; kwargs...) where {W} = fit(Maxogram{W}, v, wv, args...; kwargs...)

function push!(h::Maxogram{T,N},xs::NTuple{N,Real},w::Real) where {T,N}
    h.isdensity && error("Density Maxogram doesn't make sense!")
    idx = binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        @inbounds h.weights[idx...] = max(h.weights[idx...], w)
    end
    h
end

fit(::Type{Maxogram{T}}, vs::NTuple{N,AbstractVector}, edges::NTuple{N,AbstractVector}; closed::Symbol=:left) where {T,N} =
    append!(Maxogram(edges, T, closed, false), vs)

fit(::Type{Maxogram{T}}, vs::NTuple{N,AbstractVector}; closed::Symbol=:left, nbins=StatsBase.sturges(length(vs[1]))) where {T,N} =
    fit(Maxogram{T}, vs, StatsBase.histrange(vs, StatsBase._nbins_tuple(vs, nbins),closed); closed=closed)

fit(::Type{Maxogram{T}}, vs::NTuple{N,AbstractVector}, wv::StatsBase.AbstractWeights{W}, edges::NTuple{N,AbstractVector}; closed::Symbol=:left) where {T,N,W} =
    append!(Maxogram(edges, T, closed, false), vs, wv)

fit(::Type{Maxogram{T}}, vs::NTuple{N,AbstractVector}, wv::StatsBase.AbstractWeights; closed::Symbol=:left, nbins=StatsBase.sturges(length(vs[1]))) where {T,N} =
    fit(Maxogram{T}, vs, wv, StatsBase.histrange(vs, StatsBase._nbins_tuple(vs, nbins),closed); closed=closed)

fit(::Type{Maxogram}, args...; kwargs...) = fit(Maxogram{Int}, args...; kwargs...)
fit(::Type{Maxogram}, vs::NTuple{N,AbstractVector}, wv::StatsBase.AbstractWeights{W}, args...; kwargs...) where {N,W} = fit(Maxogram{W}, vs, wv, args...; kwargs...)

function _make_max(
    vs::NTuple{N,AbstractVector},
    binning;
    normed = false,
    weights = nothing,
) where {N}
    localvs = Plots._filternans(vs)
    edges = Plots._hist_edges(localvs, binning)
    h = identity(
        weights === nothing ?
            fit(Maxogram, localvs, edges, closed = :left) :
            fit(
            Maxogram,
            localvs,
            StatsBase.Weights(weights),
            edges,
            closed = :left,
        ),
    )
    # normalize!(h, mode = _hist_norm_mode(normed))
end


Plots.@recipe function f(::Type{Val{:maxogram}}, x, y, z)
    seriestype := length(y) > 1e6 ? :stepmax : :barmax
    ()
end
Plots.@deps maxogram barmax

Plots.@recipe function f(::Type{Val{:barmax}}, x, y, z)
    h = _make_max(
        (y,),
        plotattributes[:bins],
        normed = plotattributes[:normalize],
        weights = plotattributes[:weights],
    )
    x := h.edges[1]
    y := h.weights
    seriestype := :barbins
    ()
end
Plots.@deps barmax barbins

Plots.@recipe function f(::Type{Val{:stepmax}}, x, y, z)
    h = _make_max(
        (y,),
        plotattributes[:bins],
        normed = plotattributes[:normalize],
        weights = plotattributes[:weights],
    )
    x := h.edges[1]
    y := h.weights
    seriestype := :stepbins
    ()
end
Plots.@deps stepmax stepbins

Plots.@recipe function f(::Type{Val{:scattermax}}, x, y, z)
    h = _make_max(
        (y,),
        plotattributes[:bins],
        normed = plotattributes[:normalize],
        weights = plotattributes[:weights],
    )
    x := h.edges[1]
    y := h.weights
    seriestype := :scatterbins
    ()
end
Plots.@deps scattermax scatterbins


Plots.@recipe function f(h::Maxogram{T,1,E}) where {T,E}
    seriestype --> :barbins

    st_map = Dict(
        :bar => :barbins,
        :scatter => :scatterbins,
        :step => :stepbins,
        :steppost => :stepbins, # :step can be mapped to :steppost in pre-processing
    )
    seriestype :=
        get(st_map, plotattributes[:seriestype], plotattributes[:seriestype])

    if plotattributes[:seriestype] == :scatterbins
        # Workaround, error bars currently not set correctly by scatterbins
        edge, weights, xscale, yscale, baseline =
            Plots._preprocess_binlike(plotattributes, h.edges[1], h.weights)
        xerror --> diff(h.edges[1]) / 2
        seriestype := :scatter
        (Plots._bin_centers(edge), weights)
    else
        (h.edges[1], h.weights)
    end
end


Plots.@recipe function f(hv::AbstractVector{H}) where {H<:Maxogram}
    for h in hv
        Plots.@series begin
            h
        end
    end
end


# ---------------------------------------------------------------------------
# Maxogram 2D


Plots.@recipe function f(::Type{Val{:maxogram2d}}, x, y, z)
    h = _make_max(
        (x, y),
        plotattributes[:bins],
        normed = plotattributes[:normalize],
        weights = plotattributes[:weights],
    )
    x := h.edges[1]
    y := h.edges[2]
    z := Plots.Surface(h.weights)
    seriestype := :bins2d
    ()
end
Plots.@deps maxogram2d bins2d


Plots.@recipe function f(h::Maxogram{T,2,E}) where {T,E}
    seriestype --> :bins2d
    (h.edges[1], h.edges[2], Plots.Surface(h.weights))
end

Plots.@shorthands maxogram