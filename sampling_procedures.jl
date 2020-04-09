function _batch!(
    mm::AbstractArray{M},
    ww::AbstractArray{Float64},
    y_obs::Y,
    q::AbstractGenerator{M},
    w::AbstractWeight{M},
)::NamedTuple where M where Y

    save1 = rand!(mm, q)
    save2 = weight!(ww, w, mm, y_obs)
    return merge(save1, save2)
end

export rejection_sample, importance_sample

function rejection_sample(
    y_obs::Y,
    q::AbstractGenerator{M},
    w::AbstractWeight{M},
    N::Int64,
) where M where Y
    mm = Array{M}(undef, N)
    ww = Array{Float64}(undef, N)
    save = _batch!(mm, ww, y_obs, q, w)
    output = merge((mm=mm, ww=ww), save)
    return output
end

function importance_sample(
    y_obs::Y,
    prior::AbstractGenerator{M},
    proposal::AbstractGenerator{M},
    w::AbstractWeight{M},
    N::Int64,
) where M where Y
    mm = Array{M}(undef, N)
    ww = Array{Float64}(undef, N)
    save = _batch!(mm, ww, y_obs, proposal, w)

    pp = (:pp in keys(save)) ? save[:pp] : unnormalised_likelihood(prior, mm)[:pp]
    qq = (:qq in keys(save)) ? save[:qq] : unnormalised_likelihood(proposal, mm)[:pp]
    ww .*= pp./qq
    return merge((mm=mm, ww=ww, pp=pp, qq=qq), save)
end