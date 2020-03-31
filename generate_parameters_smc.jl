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

# TODO - should really implement unnormalised_likelihood for this distribution too