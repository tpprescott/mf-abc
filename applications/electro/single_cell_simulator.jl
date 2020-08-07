using DifferentialEquations
using LinearAlgebra

export AbstractEMField

# For every concrete Field<:AbstractEMField we need to be able to call
# (emf::Field)(t::Float64)::Complex{Float64}
abstract type AbstractEMField end
function drift(emf::AbstractEMField)
    f = function (du, u, p, t)
        du[1] = -∇W(u[1]; p...)
        du[1] += p[:polarity_bias] * emf(t)
        du[1] *= p[:σ]^2
        du[1] /= 2
        return nothing
    end
    return f
end 

export NoEF, ConstantEF, StepEF

struct NoEF <: AbstractEMField end
const NOFIELD = NoEF()
function drift(emf::NoEF)
    f = function (du, u, p, t)
        du[1] = -∇W(u[1]; p...)
        du[1] *= p[:σ]^2
        du[1] /= 2
        return nothing
    end
    return f
end
(::NoEF)(t) = complex(0.0)

struct ConstantEF <: AbstractEMField
    x::Complex{Float64}
end
(emf::ConstantEF)(t) = emf.x

struct StepEF <: AbstractEMField
    x0::Complex{Float64}
    x1::Complex{Float64}
    t_step::Float64
end
(emf::StepEF)(t) = t<emf.t_step ? emf.x0 : emf.x1

export v_EM, v_cell, velocity

function v_EM(t; 
    v::Float64, 
    EMF::AbstractEMField=NOFIELD, 
    position_bias::Float64=0.0, 
    kwargs...)
    
    return v * position_bias * EMF(t)
end

alignment(z1::Complex, z2::Complex) = Float64((dot(z1,z2)+dot(z2,z1))/2)
function v_cell(t, p::Complex{Float64}; 
    v::Float64, 
    EMF::AbstractEMField=NOFIELD, 
    speed_change::Float64=0.0,
    alignment_bias::Float64=0.0,
    kwargs...)

    if iszero(p)
        return complex(0.0)
    else
        out = v*p
        u = EMF(t)
        if !iszero(u)
            phat = p/abs(p)
            out *= (1 + speed_change*abs(u) + alignment_bias*alignment(u, phat)) 
        end 
        return out
    end
end

function velocity(t, p::Complex{Float64}; kwargs...) 
    return v_EM(t; kwargs...) + v_cell(t, p; kwargs...)
end

export SingleCellSimulator
# abstract type SingleCellSimulator{M, T, F<:EMField} <: AbstractSimulator{M, T} end

struct SingleCellSimulator{EMF<:AbstractEMField} <: AbstractSimulator
    emf::EMF
    σ_init::Float64
    tspan::Tuple{Float64, Float64}
    saveat::Array{Float64,1}

    function SingleCellSimulator(;
        emf::F=NOFIELD,
        σ_init::Float64=0.0, 
        tspan::Tuple{Float64,Float64}=(0.0, 180.0), 
        saveat::Array{Float64,1}=collect(0.0:5.0:180.0),
        ) where F<:AbstractEMField
        
        return new{F}(emf, σ_init, tspan, saveat)
    end
end

function ∇W(p::Complex{Float64}; β::Float64, λ::Float64, kwargs...)::Complex{Float64}
    p2 = abs2(p)
    return β*(p2-1)*(p2-(λ-1))*p
end
function drift_NoEF!(du, u, p, t)
    du[1] = -∇W(u[1]; p...)
    return nothing
end
function noise!(du, u, p, t)
    du[1] = p[:σ]
    return nothing
end

function W(z::Complex{Float64}; β::Float64, λ::Float64, kwargs...)::Float64
    r2 = abs2(z)
    return β*((r2^3)/6 - λ*(r2^2)/4 + (λ-1)*(r2)/2)
end
W(x, y; kwargs...) = W(complex(x,y); kwargs...)

function _map_barriers_to_coefficients(EB_on::Float64, EB_off::Float64)::NTuple{2,Float64}
    λ = get_λ(EB_on, EB_off)
    β = get_β(EB_on, λ)
    return β, λ
end
function get_λ(EB_on, EB_off)
    EB_on == EB_off && return 4/3
    R = EB_off / (EB_on - EB_off)
    D = sqrt(abs(R^2 + R^3))
    if EB_on > EB_off
        w = (R + D)^(1/3)
        delta = w - R/w
    else
        sol_vec = @. (atan(D,R) + 2*π*[0,1,2])/3
        broadcast!(cos, sol_vec, sol_vec)
        sol_vec .*= 2*sqrt(-R)
        delta = 0.0
        while !(2/3<delta<1)
            delta=pop!(sol_vec)
        end
    end
    return 2 - delta
end
function get_β(EB_on, λ)
    return -12.0 * EB_on / (((λ-1)^2)*(λ-4))
end
function initial_conditions(sigma)
    return [sigma*complex(randn(), randn())]
end

function (F::SingleCellSimulator)(; 
    polarised_speed::Float64, σ::Float64, EB_on::Float64, EB_off::Float64,
    speed_change::Float64=0.0, polarity_bias::Float64=0.0, position_bias::Float64=0.0, alignment_bias::Float64=0.0,
    u0::Array{Complex{Float64},1}=initial_conditions(F.σ_init), W=nothing,
    output_trajectory=false, kwargs...)

    # Simulate the polarity SDE
    β, λ = _map_barriers_to_coefficients(EB_on, EB_off)
    parm_p = (
        σ=σ,
        β=β, 
        λ=λ, 
        polarity_bias=polarity_bias,
    )
    
    independentFlag = W === nothing
    couple = independentFlag ? nothing : NoiseWrapper(W)
    prob_p = SDEProblem(drift(F.emf), noise!, u0, F.tspan, parm_p, noise=couple)
    sol_p = solve(prob_p, save_noise = independentFlag)

    # Integrate to get position
    function v!(dx, x, parm, t)
        dx[1] = velocity(t, sol_p(t); parm...)
        return nothing
    end
    parm_x = (
        v = polarised_speed,
        EMF = F.emf,
        speed_change = speed_change,
        alignment_bias = alignment_bias,
        position_bias = position_bias,
    )

    prob_x = ODEProblem(v!, [0.0], F.tspan, parm_x)
    if output_trajectory
        sol_x = solve(prob_x)
        return sol_p, sol_x
    else
        sol_x = solve(prob_x, saveat=F.saveat)
        summary = get_displacements(sol_x.u)
        independentFlag && (W=sol_p.W)
        return (y=summary, u0=u0, W=W)
    end
end
import Base.eltype
eltype(::Type{T}) where T<:SingleCellSimulator = NamedTuple{(:y, :u0, :W), Tuple{Array{Float64,1}, Array{Complex{Float64}, 1}, NoiseProcess}}


function polarity(z::Complex{Float64};
    polarised_speed::Float64,
    EMF::AbstractEMField=NOFIELD,
    speed_change::Float64,
    alignment_bias::Float64,
    kwargs...)

    z_EM = v_EM(0; v=polarised_speed, EMF=EMF, kwargs...)
    z_cell = z - z_EM

    iszero(z_cell) && (return complex(0.0))
    phat = z_cell/abs(z_cell)
    u = EMF(0)
    z_cell /= (1 + speed_change*abs(u) + alignment_bias*alignment(u,phat))

    return z_cell

end

export stationary_velocity
function stationary_velocity(vel::Complex{Float64};
    polarised_speed::Float64,
    EB_on::Float64,
    EB_off::Float64,
    EMF::AbstractEMField=NOFIELD,
    polarity_bias::Float64=0.0,
    speed_change::Float64=0.0,
    alignment_bias::Float64=0.0,
    kwargs...
    )

    p = polarity(vel; 
    polarised_speed=polarised_speed, EMF=EMF, speed_change=speed_change, alignment_bias=alignment_bias, kwargs...)

    β, λ = _map_barriers_to_coefficients(EB_on, EB_off)
    ϕ::Float64 = exp(-W(p; β=β, λ=λ) + polarity_bias*alignment(p,EMF(0)))
    return ϕ
end
