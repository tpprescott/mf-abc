

out_idx(mat::SparseMatrixCSC, v::Int64) = mat.colptr[v]:mat.colptr[v+1]-1
function step_dist(H::ContinuationDistribution{T}, m::Model, u::U, output::MFABCOutput)::Tuple{AbstractVector{T}, AbstractVector{Int64}} where T<:AbstractContinuationProbability{U} where U
    # Find the neighborhood of v 
    # TODO exclude those already simulated
    v = output.seq[end]
    idx = out_idx(H.g, v)
    eta_vj = view(H.g.nzval, idx)
    j = view(H.g.rowval, idx)
    return eta_vj, j
end

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

