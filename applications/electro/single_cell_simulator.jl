using DifferentialEquations
using LinearAlgebra

export AbstractEMField

# For every concrete Field<:AbstractEMField we need to be able to call
# (emf::Field)(t::Float64)::Complex{Float64}
abstract type AbstractEMField end
export NoEF, ConstantEF, StepEF

struct NoEF <: AbstractEMField end
const NOFIELD = NoEF()
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

############### POLARITY SDE DEFINITIONS

function ∇W(pol::Complex{Float64}; β::Float64, λ::Float64, kwargs...)::Complex{Float64}
    p2 = abs2(pol)
    return β*(p2-1)*(p2-(λ-1))*pol
end
function W(pol::Complex{Float64}; β::Float64, λ::Float64, kwargs...)::Float64
    p2 = abs2(pol)
    return β*((p2^3)/6 - λ*(p2^2)/4 + (λ-1)*(p2)/2)
end
W(x, y; kwargs...) = W(complex(x,y); kwargs...)


function drift(EMF::AbstractEMField)
    f = function (pol, p, t)
        dpol = -∇W(pol; p...)
        dpol += p.γ4 * EMF(t)
        dpol *= p.D
        return dpol
    end
    return f
end
function drift(EMF::NoEF)
    f = function (pol, p, t)
        dpol = -∇W(pol; p...)
        dpol *= p.D
        return dpol
    end
    return f
end
noise(u, p, t) = p.σ

################# POLARITY-VELOCITY MAPS

export v_EM, v_cell, velocity

function v_EM(EMF::AbstractEMField)
    f = function (pos, p, t)
        return p.v * p.γ1 * EMF(t)
    end
    return f
end

alignment(z1::Complex, z2::Complex) = Float64((dot(z1,z2)+dot(z2,z1))/2)
function v_cell(EMF::AbstractEMField)
    f = function (pos, p, t)
        u = EMF(t)
        g = function (pol::Complex{Float64})
            if iszero(pol)
                out = pol
            else
                out = p.v * pol
                if !iszero(u)
                    polhat = pol/abs(pol)
                    out *= (1 + p.γ2*abs(u) + p.γ3*alignment(u, polhat))
                end
            end
            return out
        end
        return g
    end
    return f
end
function v_cell(EMF::AbstractEMField, pol::Complex{Float64})
    if iszero(pol)
        f = function (pos, p, t)
            return zero(Complex{Float64})
        end
    else
        f = function (pos, p, t)
            u = EMF(t)
            out = p.v * pol
            if !iszero(u)
                polhat = pol/abs(pol)
                out *= (1 + p.γ2*abs(u) + p.γ3*alignment(u, polhat)) 
            end 
            return out
        end
    end
    return f
end
function v_cell(EMF::AbstractEMField, pol::RODESolution)
    f = function (pos, p, t)
        pol_t = pol(t)
        if iszero(pol_t)
            out = zero(Complex{Float64})
        else
            u = EMF(t)
            out = p.v * pol_t
            if !iszero(u)
                polhat = pol_t/abs(pol_t)
                out *= (1 + p.γ2*abs(u) + p.γ3*alignment(u, polhat)) 
            end
        end
        return out
    end
    return f
end

function velocity(EMF::AbstractEMField)
    _v_EM = v_EM(EMF)
    _v_cell = v_cell(EMF)
    f = function (pos, p, t)
        offset_EM = _v_EM(pos, p, t)
        fun_cell = _v_cell(pos, p, t)
        g(pol::Complex{Float64}) = fun_cell(pol) + offset_EM
        return g
    end
    return f
end
function velocity(EMF::AbstractEMField, pol)
    _v_EM = v_EM(EMF)
    _v_cell = v_cell(EMF, pol)
    f(pos, p, t) = _v_EM(pos, p, t) + _v_cell(pos, p, t)
    return f
end

function polarity(EMF::AbstractEMField)
    _v_EM = v_EM(EMF)
    f = function (pos, p, t)
        offset_EM = _v_EM(pos, p, t)
        u = EMF(t)
        g = function (vel::Complex{Float64})
                out = vel - offset_EM
                if iszero(out)
                    return out
                elseif !iszero(u)
                    polhat = out/abs(out)
                    out /= (1 + p.γ2*abs(u) + p.γ3*alignment(u, polhat))
                end
                out /= p.v
                return out
            end
        return g
    end
    return f
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
function initial_conditions(sigma)::Complex{Float64}
    return sigma*complex(randn(), randn())
end

function (F::SingleCellSimulator)(; 
    v::Float64, EB_on::Float64, EB_off::Float64, D::Float64,
    γ1::Float64=0.0, γ2::Float64=0.0, γ3::Float64=0.0, γ4::Float64=0.0, 
    u0::Complex{Float64}=initial_conditions(F.σ_init), 
    W=nothing,
    output_trajectory=false, 
    kwargs...)

    # Simulate the polarity SDE
    β, λ = _map_barriers_to_coefficients(EB_on, EB_off)
    parm_p = (
        D=D,
        σ=sqrt(2*D),
        β=β, 
        λ=λ, 
        γ4=γ4,
    )
    
    independentFlag = (W === nothing)
    couple = independentFlag ? nothing : NoiseWrapper(W)
    prob_p = SDEProblem(drift(F.emf), noise, u0, F.tspan, parm_p, noise=couple)
    sol_p = solve(prob_p, save_noise = independentFlag)

    # Integrate to get position
    parm_x = (
        v=v,
        γ1=γ1,
        γ2=γ2,
        γ3=γ3,
    )

    _velocity = velocity(F.emf, sol_p)
    prob_x = ODEProblem(_velocity, complex(0.0), F.tspan, parm_x)
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
eltype(::Type{T}) where T<:SingleCellSimulator = NamedTuple{(:y, :u0, :W), Tuple{Array{Float64,1}, Complex{Float64}, NoiseProcess}}

export stationarydist
function stationarydist(EMF::AbstractEMField)
    get_polarity = polarity(EMF)
    f = function (pos, p, t)
        fun_polarity = get_polarity(pos, p, t)
        β, λ = _map_barriers_to_coefficients(p.EB_on, p.EB_off)
        fun_alignment(pol) = p.γ4 * alignment(pol, EMF(t))
        g = function (vel::Complex{Float64})
            pol = fun_polarity(vel)
            ϕ = exp(-W(pol; β=β, λ=λ) + fun_alignment(pol))
            return ϕ
        end
        return g
    end
    return f
end