function weighted_sum(F, ww::AbstractArray{Float64,1}, xx; normalise=false)
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

using StatsBase, Statistics

export SequentialImportanceDistribution
struct SequentialImportanceDistribution{Θ, W<:Weights, Π<:AbstractGenerator{Θ}} <: AbstractGenerator{Θ}
    w_A::W
    w_R::W
    K_θ::Array{PerturbationKernel{Θ}, 1}
    prior::Π
    δ::Float64
    
    function SequentialImportanceDistribution(
        θ::AbstractArray{Θ, 1}, 
        importance_weights::AbstractArray{Float64,1}, 
        prior::Π,
        δ=0.0,
    ) where Π <: AbstractGenerator{Θ} where Θ <: AbstractModel
    
        length(θ)==length(importance_weights) || error("Mismatch between parameter samples and importance weights")
        dim = length(θ[1])

        importance_weights ./= sum(importance_weights)
        AR = isnz.(importance_weights)
        w_A = Weights(pospart.(importance_weights[AR]))
        w_R = Weights(negpart.(importance_weights[AR]))

        covariance_matrix = dim==1 ? cov(make_array(θ[AR]), w_A, corrected=false) : cov(make_array(θ[AR]), w_A, 2, corrected=false)
        K_θ = broadcast(PerturbationKernel, θ[AR], Ref(covariance_matrix))

        return new{Θ, typeof(w_A), Π}(w_A, w_R, K_θ, prior, δ)     
    end
end

function (q::SequentialImportanceDistribution{Θ, W, Π})(; kwargs...) where Θ where W where Π
    # Complicated case only if some weights are negative
    if ispos(q.w_R.sum)
        # Select from prior or...
        if rand() < q.δ/(q.δ + (1-q.δ)*q.w_A.sum)
            proposal = rand(q.prior)
        # ...from positive mixture
        else
            proposal = rand(sample(q.K_θ, q.w_A), prior=q.prior)
            isfinite(proposal.logp) || (return rand(q))
        end
        p = exp(proposal.logp)

        # Rejection step to ensure selection from max(0, F-G) 
        q_i = broadcast(exp ∘ logpdf, q.K_θ, Ref(proposal.θ))
        F = q.δ*p + (1-q.δ)*mean(q_i, q.w_A)*(q.w_A.sum)
        G = (1-q.δ)*mean(q_i, q.w_R)*(q.w_R.sum)
        β = G/F
        γ = q.δ*p/F
        if rand() < max(γ, 1-β)
            return merge(proposal, (logq = log(q.δ*p + (1-q.δ)*max(0, F-G)),) )
        else
            return rand(q)
        end
    else    
        # Select a particle and perturb it
        proposal  = rand(sample(q.K_θ, q.w_A), prior=q.prior)

        # Check prior likelihood is positive
        isfinite(proposal.logp) || (return rand(q))

        # Weighted sum of perturbation kernel likelihoods gives the (unnormalised) likelihood under the importance distribution
        q_i = broadcast(exp ∘ logpdf, q.K_θ, Ref(proposal.θ))
        lh = mean(q_i, q.w_A)*q.w_A.sum
        return merge(proposal, (logq = log(lh),))
    end
end

# TODO - should really implement logpdf for this distribution too