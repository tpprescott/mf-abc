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

export sample
function sample(
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

function sample(
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
#Base.iterate(q::AbstractGenerator{M}, count=1) where M = (rand(q), count+1)
#Iterators.IteratorSize(::Type{T}) where T<:AbstractGenerator{M} where M = Iterators.IsInfinite()
#Iterators.eltype(::Type{T}) where T<:AbstractGenerator{M} where M = M

using Distributions
import Distributions.rand, Distributions.rand!

export DistributionGenerator
struct DistributionGenerator{M, D<:Distribution} <: AbstractGenerator{M}
    d::D
    function DistributionGenerator(::Type{M}, x::D) where M where D<:Distribution
        N = length(fieldnames(M))
        length(x)==N || error("$N names for $(length(x)) variables")
        return new{M,D}(x)
    end
end

function (q::DistributionGenerator{M,D})()::M where M where D
    return M(rand(q.d))
end
function rand(q::DistributionGenerator{M, D}) where M<:AbstractModel where D
    return (m=q(),)
end

function rand!(mm::AbstractArray{M}, q::DistributionGenerator{M,D}) where M<:AbstractModel where D
    xx = rand(q.d, length(mm))
    if length(q.d)==1
        for i in eachindex(mm)
            mm[i] = M(xx[i])
        end
    else
        for i in eachindex(mm)
            mm[i] = M(view(xx,:,i))
        end
    end
    return NamedTuple()
end

function make_array(mm::Array{M}) where M<:AbstractModel
    dim = length(fieldnames(M))
    if dim==1
        T = fieldtypes(M)[1]
        out = Vector{T}(undef, length(mm))
        for i in eachindex(mm)
            out[i] = mm[i][1]
        end
    else
        T = promote_type(fieldtypes(M)...)
        out = Array{T,2}(undef, dim, length(mm))
        vals = values.(mm)
        for i in eachindex(mm)
            for j in 1:dim
                out[j,i] = vals[i][j]
            end
        end
    end
    return out
end

function unnormalised_likelihood(q::DistributionGenerator{M,D}, m::M) where M<:AbstractModel where D
    dim = length(q.d)
    p = (dim==1 ? pdf(q.d, values(m)[1]) : pdf(q.d, [values(m)...]))
    return (p=p,)
end
function unnormalised_likelihood!(pp::AbstractArray{Float64,1}, q::DistributionGenerator{M, D}, mm::AbstractArray{M}) where M where D
    dim = length(q.d)
    mm_array = make_array(mm)
    pp[:] .= (dim==1 ? pdf.(q.d, mm_array) : pdf(q.d, mm_array))
    return NamedTuple()
end
function unnormalised_likelihood(q::DistributionGenerator{M,D}, mm::AbstractArray{M}) where M<:AbstractModel where D
    pp = Array{Float64,1}(undef, size(mm))
    save = unnormalised_likelihood!(pp, q, mm)
    return merge((pp=pp,), save)
end

function weighted_sum(F, ww::AbstractArray{Float64}, xx; normalise=false)
    G(w,x) = w*F(x)
    out = mapreduce(G, +, ww, xx)
    if normalise
        out ./= sum(ww)
    end
    return out
end

ispos(x)=x>zero(x)
isneg(x)=ispos(-x)
isnz(x)=!iszero(x)
pospart(x)=(ispos(x) ? x : zero(x))
negpart(x)=pospart(-x)

using StatsBase: std, AnalyticWeights
using LinearAlgebra: normalize!

export SequentialImportanceDistribution
struct SequentialImportanceDistribution{M,D<:AbstractGenerator{M},P<:AbstractGenerator{M}} <: AbstractGenerator{M}
    select_particle::Categorical
    perturb_particle::Array{D,1}
    prior::P
    delta::Float64
    ζ_A::Float64
    abs_ww_A::Array{Float64,1}
    abs_ww_R::Array{Float64,1}
    function SequentialImportanceDistribution(ww::Array{Float64,1}, mm::Array{M,1}, prior::P, delta=0.0) where P<:AbstractGenerator{M} where M<:AbstractModel
        dim = length(fieldnames(M))

        weights = copy(ww)
        normalize!(weights,1)

        AR = isnz.(weights)
        abs_ww_A = pospart.(weights[AR])
        abs_ww_R = negpart.(weights[AR])
        ζ_A = sum(abs_ww_A)

        select_particle = Categorical(abs_ww_A./ζ_A)

        mm_arr = make_array(mm[AR])
        if dim == 1
            bandwidth = 2*std(mm_arr, AnalyticWeights(abs_ww_A), corrected=true)
            perturb_particle = [DistributionGenerator(M, Normal(mm_i, bandwidth)) for mm_i in mm_arr]
        else
            bandwidth = 2*std(mm_arr, AnalyticWeights(abs_ww_A), 2, corrected=true)
            perturb_particle = [DistributionGenerator(M, MvNormal(view(mm_arr,:,i), view(bandwidth,:,1))) for i in 1:size(mm_arr, 2)]
        end
        D = eltype(perturb_particle)
        return new{M, D, P}(select_particle, perturb_particle, prior, delta, ζ_A, abs_ww_A, abs_ww_R)     
    end
end


function rand(q::SequentialImportanceDistribution{M,D,P}) where M where D where P
    # Complicated case only if some weights are negative
    if any(ispos, q.abs_ww_R)
        # Select from prior or...
        if rand() < q.delta/(q.delta + (1-q.delta)*q.ζ_A)
            m_star = rand(q.prior)[:m]
            p = unnormalised_likelihood(q.prior, m_star)[:p]
        # ...from positive mixture
        else
            j = rand(q.select_particle)
            m_star = rand(q.perturb_particle[j])[:m]
            p = unnormalised_likelihood(q.prior, m_star)[:p]
            ispos(p) || (return rand(q))
        end
        # Rejection step to ensure selection from max(0, F-G) 
        KK_1(K_i) = unnormalised_likelihood(K_i, m_star)[:p]
        F = q.delta*p + (1-q.delta)*weighted_sum(KK_1, q.abs_ww_A, q.perturb_particle)
        G = (1 - q.delta)*weighted_sum(KK_1, q.abs_ww_R, q.perturb_particle)
        β = G/F
        γ = q.delta*p/F
        if rand() < max(γ, 1-β)
            return (m = m_star, p = p, q = q.delta*p + (1-q.delta)*max(0, F-G))
        else
            return rand(q)
        end
    else    
        # Select a particle and perturb it
        j = rand(q.select_particle)
        m_star = rand(q.perturb_particle[j])[:m]

        # Check prior likelihood is positive
        p = unnormalised_likelihood(q.prior, m_star)[:p]
        ispos(p) || (return rand(q))

        # Function maps a perturbation kernel to its contribution to the likelihood
        KK_2(K_i) = unnormalised_likelihood(K_i, m_star)[:p]
        
        # Weighted sum of perturbation kernel likelihoods gives the (unnormalised) likelihood under the importance distribution
        lh = weighted_sum(KK_2, q.abs_ww_A, q.perturb_particle)
        return (m = m_star, p = p, q = lh)
    end
end

function rand!(mm::AbstractArray{M}, q::SequentialImportanceDistribution{M,D,P}) where M<:AbstractModel where D where P
    pp = Array{Float64}(undef, size(mm))
    qq = Array{Float64}(undef, size(mm))
    for i in eachindex(mm)
        out = rand(q)
        mm[i] = out[:m]
        pp[i] = out[:p]
        qq[i] = out[:q]
    end
    return (pp=pp, qq=qq)
end

function rand(q::SequentialImportanceDistribution{M,D,P}, N::Int64) where M<:AbstractModel where D where P
    mm = Array{M,1}(undef, N)
    save = rand!(mm, q)
    return merge((mm=mm,), save)
end