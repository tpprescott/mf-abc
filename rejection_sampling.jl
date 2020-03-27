# Build a sample
function _batch!(
    mm::AbstractArray{M},
    ww::AbstractArray{Float64},
    u::U,
    q::AbstractGenerator{M},
    w::AbstractWeight{M,U};
    saved...,
)::Nothing where M where U
    rand!(mm, q; saved...)
    weight!(ww, w, mm, u; saved...)
    return nothing
end

#=
function _batch(
    u::U,
    q::AbstractGenerator{M},
    w::AbstractWeight{M,U},
    N::Int64,
    saved...,
)::Tuple{AbstractArray{M}, AbstractArray{Float64}} where M where U

    mm = rand(q, N)
    ww = weight(w, mm, u, saved...)
    return mm, ww
end
=#

export sample
function sample(
    u::U,
    q::AbstractGenerator{M},
    w::AbstractWeight{M, U},
    N::Int64;
    saved...,
) where M where U
    mm = Array{M}(undef, N)
    ww = Array{Float64}(undef, N)
    _batch!(mm, ww, u, q, w; saved...)
    return collect(zip(mm, ww)), saved
end
