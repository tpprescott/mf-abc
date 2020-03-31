function _batch!(
    mm::AbstractArray{M},
    ww::AbstractArray{Float64},
    u::U,
    q::AbstractGenerator{M},
    w::AbstractWeight{M,U},
)::NamedTuple where M where U

    save1 = rand!(mm, q)
    save2 = weight!(ww, w, mm, u)
    return merge(save1, save2)
end

export rejection_sample, importance_sample

function rejection_sample(
    u::U,
    q::AbstractGenerator{M},
    w::AbstractWeight{M, U},
    N::Int64,
) where M where U
    mm = Array{M}(undef, N)
    ww = Array{Float64}(undef, N)
    save = _batch!(mm, ww, u, q, w)
    output = merge((mm=mm, ww=ww), save)
    return output
end

function importance_sample(
    u::U,
    prior::AbstractGenerator{M},
    proposal::AbstractGenerator{M},
    w::AbstractWeight{M, U},
    N::Int64,
) where M where U
    mm = Array{M}(undef, N)
    ww = Array{Float64}(undef, N)
    save = _batch!(mm, ww, u, proposal, w)

    pp = (:pp in keys(save)) ? save[:pp] : unnormalised_likelihood(prior, mm)[:pp]
    qq = (:qq in keys(save)) ? save[:qq] : unnormalised_likelihood(proposal, mm)[:pp]
    ww .*= pp./qq
    return merge((mm=mm, ww=ww, pp=pp, qq=qq), save)
end