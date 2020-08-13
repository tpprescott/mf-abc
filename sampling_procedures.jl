include("mcmc.jl")
include("is.jl")
include("smc.jl")

include("sample_visualisation.jl")

export posweight, new_col, new_par, sample_mean
posweight(row) = row.weight>0
function new_col(name::Symbol, func, t::IndexedTable)
    ff = @showprogress pmap(func, select(t,:θ))
    return transform(t, name => ff)
end

function new_par(name::Symbol, func, t::IndexedTable)
    θ = select(t, :θ)
    ff = @showprogress pmap(func, select(t,:θ))
    xθ = [merge(θ_i, NamedTuple{(name,)}((ff_i,))) for (ff_i, θ_i) in zip(ff, θ)]
    return transform(t, :θ => xθ)
end

function sample_mean(name::Symbol, t::IndexedTable)
    w = select(t, :weight)
    f = select(t, name)
    return sum(w.*f)/sum(w)
end
function sample_mean(func, t::IndexedTable)
    t_extended = new_col(:new_col, func, t)
    return sample_mean(:new_col, t_extended)
end