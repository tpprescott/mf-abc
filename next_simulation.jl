function Base.iterate(iter::MFABCIterator{M,U}) where M where U
    
    v = rand(iter.mf_w.F_init)
    eta = 1.0
    
    node = merge(weight(iter.mf_w.nodes[v], iter.m, iter.u), (eta=eta, v=v)) # Named tuple containing w, d, y, (n if included (i.e. noise process))
    cont, eta, v = next_simulation(iter, v, node)
    return node, (cont, eta, v, (node,))
end

function Base.iterate(iter::MFABCIterator{M,U}, (cont, eta, v, path)) where M where U
    cont || (return nothing)
    
    node = merge(weight(iter.mf_w.nodes[v], iter.m, iter.u, path...), (eta=eta, v=v))
    cont, eta, v = next_simulation(iter, v, node, path...)
    return node, (cont, eta, v, (node, path...))
end

import .LikelihoodFree.weight
import .LikelihoodFree.weight!
import .LikelihoodFree.simulate
import .LikelihoodFree.simulate!
function weight(w::LikelihoodFreeWeight{M,U}, m::M, u::U, path...) where M where U
    sim = simulate(w.F, m, u, path...)
    out = compare(w.C, u, sim[:y])
    return merge(out, sim)
end
simulate(w::AbstractSimulator{M,U,Y}, m::M, u::U, path...) where M where U where Y = simulate(w,m,u)

out_idx(mat::SparseMatrixCSC, v::Int64) = mat.colptr[v]:mat.colptr[v+1]-1
function step_dist(H::ContinuationDistribution, m::M, u::U, v::Int64) where M where U
    idx = out_idx(H.g, v)
    eta_v_ = view(H.g.nzval, idx)
    j = view(H.g.rowval, idx)
    return eta_v_, j
end

function next_simulation(iter::MFABCIterator{M,U}, v::Int64, path...) where M where U
    eta_v_, j = step_dist(iter.mf_w.edges, iter.m, iter.u, v)
    ispos(length(j)) || (return false, zero(Float64), zero(Int64))

    eta = map(eta_vj -> continuation_probability(eta_vj, iter.m, iter.u, path...), eta_v_)
    cont_prob = sum(eta)
    isp(cont_prob) || error("Invalid continuation probability of $cont_prob")
    
    cont_prob = eta[end]
    if rand()<cont_prob
        eta ./= cont_prob
        v = j[rand(Categorical(eta))]
        return true, cont_prob, v
    else
        return false, cont_prob, 0
    end
end


#=

function next_simulation(x::AbstractVector{T}, y::AbstractVector{Int64}, m::Model, u::U, output::MFABCOutput)::Tuple{Bool, Int64, Float64} where T<:AbstractContinuationProbability{U} where U
    ispos(length(x)) || (return (false, zero(Int64), zero(Float64)))
    eta = [continuation_probability(x_i, m, u, output) for x_i in x]
    cumsum!(eta,eta)
    isp(eta[end]) || error("Invalid continuation probability of $(eta[end])")
    r = rand()
    f(x) = <(r,x)
    if f(eta[end])
        return true, y[findfirst(f, eta)], eta[end]
    else 
        return false, zero(Int64), eta[end]
    end
end
function next_simulation(H::ContinuationDistribution{T}, m::Model, u::U, output::MFABCOutput) where T<:AbstractContinuationProbability{U} where U
    return next_simulation(step_dist(H, m, u, output)..., m, u, output)
end

=#