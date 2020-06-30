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
    defence::Π
    δ::Float64
    
    function SequentialImportanceDistribution(
        θ::AbstractArray{Θ, 1}, 
        importance_weights::AbstractArray{Float64,1}, 
        defence::Π,
        δ=0.0,
    ) where Π <: AbstractGenerator{Θ} where Θ <: AbstractModel
    
        length(θ)==length(importance_weights) || error("Mismatch between parameter samples and importance weights")
        dim = length(θ[1])

        iw = copy(importance_weights)
        iw ./= sum(iw)
        AR = isnz.(iw)
        w_A = Weights(pospart.(iw[AR]))
        w_R = Weights(negpart.(iw[AR]))

        covariance_matrix = dim==1 ? cov(make_array(θ[AR]), w_A, corrected=false) : cov(make_array(θ[AR]), w_A, 2, corrected=false)
        K_θ = broadcast(PerturbationKernel, θ[AR], Ref(covariance_matrix))

        return new{Θ, typeof(w_A), Π}(w_A, w_R, K_θ, defence, δ)     
    end
end

δ_effective(q::SequentialImportanceDistribution) = ispos(q.w_R.sum) ? q.δ : zero(q.δ)
choose_defence(q::SequentialImportanceDistribution) = ispos(q.w_R.sum) ? q.δ/(q.δ + (1-q.δ)*q.w_A.sum) : 0.0
function FG(θ, q::SequentialImportanceDistribution)
    p = exp(logpdf(q.defence, θ))
    q_i = broadcast(exp ∘ logpdf, q.K_θ, Ref(θ))
    δ = δ_effective(q)

    F = δ*p + (1-δ)*mean(q_i, q.w_A)*(q.w_A.sum)
    G = iszero(q.w_R.sum) ? 0.0 : (1-δ)*mean(q_i, q.w_R)*(q.w_R.sum)
    return F, G, p, δ
end

function (q::SequentialImportanceDistribution{Θ, W, Π})(; kwargs...) where Θ where W where Π
    # Complicated case only if some weights are negative
    u = choose_defence(q)
    # Select from defence component or from perturbations around a random (positive-weighted) parameter value
    θ, = rand()<u ? q.defence(; kwargs...) : sample(q.K_θ, q.w_A)(; kwargs...)
    # Truncate to the support of the defence component
    isfinite(logpdf(q.defence, θ)) || (return q(; kwargs...))

    # Rejection step to ensure selection from max(0, F-G) 
    F, G, p, δ = FG(θ, q)
    β = G/F
    γ = δ*p/F

    return rand()<max(γ, 1-β) ? (θ, log(max(δ*p, F-G))) : q(; kwargs...)
end

function logpdf(q::SequentialImportanceDistribution{Θ}, θ::Θ) where Θ
    F, G, p, δ = FG(θ, q)
    return iszero(p) ? -Inf : log(max(δ*p, F-G))
end