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


function pol_drift(EMF::AbstractEMField)
    f = function (pol, p, t)
        dpol = -∇W(pol; p...)
        dpol += p.γ4 * EMF(t)
        dpol *= p.D
        return dpol
    end
    return f
end
function pol_drift(EMF::NoEF)
    f = function (pol, p, t)
        dpol = -∇W(pol; p...)
        dpol *= p.D
        return dpol
    end
    return f
end
pol_noise(u, p, t) = p.σ

################# POLARITY-VELOCITY MAPS

export v_EM, v_cell, velocity

function v_EM(EMF::AbstractEMField)
    f = function (par, t)
        return par.v * par.γ1 * EMF(t)
    end
    return f
end

alignment(z1::Complex, z2::Complex) = Float64((dot(z1,z2)+dot(z2,z1))/2)
function v_cell(EMF::AbstractEMField)
    f = function (pol, par, t)
        u = EMF(t)
        if iszero(pol)
            out = pol
        else
            out = par.v * pol
            if !iszero(u)
                polhat = pol/abs(pol)
                out *= (1 + par.γ2*abs(u) + par.γ3*alignment(u, polhat))
            end
        end
        return out
    end
    return f
end
function v_cell(EMF::AbstractEMField, pol::Complex{Float64})
    ff = v_cell(EMF)
    f = function (par, t)
        return ff(pol, par, t)
    end
    return f
end

function velocity(EMF::AbstractEMField)
    _v_EM = v_EM(EMF)
    _v_cell = v_cell(EMF)
    f = function (pol, par, t)
        component_EM = _v_EM(par, t)
        component_cell = _v_cell(pol, par, t)
        return component_EM + component_cell
    end
    return f
end

function polarity(EMF::AbstractEMField)
    _v_EM = v_EM(EMF)
    f = function (par, t)
        offset_EM = _v_EM(par, t)
        u = EMF(t)
        g = function (vel::Complex{Float64})
            out = vel - offset_EM
            if iszero(out)
                return out
            elseif !iszero(u)
                polhat = out/abs(out)
                out /= (1 + par.γ2*abs(u) + par.γ3*alignment(u, polhat))
            end
            out /= par.v
            return out
            end
        return g
    end
    return f
end


#############################################################################
# SIMULATOR
#############################################################################

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

function _map_barriers_to_coefficients(ΔW_on::Float64, ΔW_off::Float64)::NTuple{2,Float64}
    λ = get_λ(ΔW_on, ΔW_off)
    β = get_β(ΔW_on, λ)
    return β, λ
end
function get_λ(ΔW_on, ΔW_off)
    ΔW_on == ΔW_off && return 4/3
    R = ΔW_off / (ΔW_on - ΔW_off)
    D = sqrt(abs(R^2 + R^3))
    if ΔW_on > ΔW_off
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
function get_β(ΔW_on, λ)
    return -12.0 * ΔW_on / (((λ-1)^2)*(λ-4))
end

gaussian_z(sigma)::Complex{Float64} = sigma*randn(ComplexF64)
const NOISEFORM = [complex(1.0), complex(0.0)]
const B_t = WienerProcess(0.0, [complex(0.0)])

function couple_noise(U0, ind_flags, u0, W)
    fun = function(prob, i, repeat)
        if ind_flags[i]
            remake(prob, u0=U0[i])
        else
            remake(prob, u0=u0[i], noise=NoiseWrapper(W[i]))
        end
    end
    return fun
end

function output_func(tvec, W_previous, ind_flags, output_trajectory)
    if output_trajectory
        fun = ((sol, i) -> (sol, false))
    else
        fun = function (sol, i)
            posvec = broadcast(t->sol(t)[2], tvec)
            summary = get_displacements(posvec)
            W = ind_flags[i] ? sol.W : W_previous[i]
            u0 = sol.prob.u0
            return (y = summary, u0=u0, W=W), false
        end
    end
    return fun
end


function SDEdrift(emf::AbstractEMField)
    dpol = pol_drift(emf)
    dpos = velocity(emf)
    f! = function (du, u, (parm_p, parm_x), t)
        du[1] = dpol(u[1], parm_p, t)
        du[2] = dpos(u[1], parm_x, t)
        return nothing
    end
    return f!
end
function g!(du, u, (parm_p, parm_x), t)
    du[1] = parm_p.σ
    du[2] = 0.0
end

function (F::SingleCellSimulator)(n::Int64 = 1;
    # Common parameters
    v::Float64, ΔW_on::Float64, ΔW_off::Float64, D::Float64,
    # EMF parameters
    γ1::Float64=0.0, γ2::Float64=0.0, γ3::Float64=0.0, γ4::Float64=0.0, 
    # Specify initial conditions
    U0 = [[gaussian_z(F.σ_init), 0] for i in 1:n],
    # Coupling
    ind_flags::Array{Bool, 1}=fill(true, n), 
    u0=fill(0.0, n), 
    W=fill(B_t, n),
    # What to save
    output_trajectory=false, 
    kwargs...)

    β, λ = _map_barriers_to_coefficients(ΔW_on, ΔW_off)
    parm_p = (
        D=D,
        σ=sqrt(2*D),
        β=β, 
        λ=λ, 
        γ4=γ4,
    )
    parm_x = (
        v=v,
        γ1=γ1,
        γ2=γ2,
        γ3=γ3,
    )
    
    f! = SDEdrift(F.emf)

    prob_nominal = SDEProblem(
        f!, 
        g!,
        fill(complex(0.0), 2),
        F.tspan,
        (parm_p, parm_x),
        noise_rate_prototype = NOISEFORM,
    )
    prob = EnsembleProblem(
        prob_nominal, 
        prob_func = couple_noise(U0, ind_flags, u0, W),
        output_func = output_func(F.saveat, W, ind_flags, output_trajectory),
    )

    sol = solve(prob, trajectories=n, save_noise=true)
    return sol.u
end

import Base.eltype
eltype(::Type{T}) where T<:SingleCellSimulator = NamedTuple{(:y, :u0, :W), Tuple{Array{Float64,1}, Array{Complex{Float64},1}, NoiseProcess}}

export stationarydist
function stationarydist(EMF::AbstractEMField)
    get_polarity = polarity(EMF)
    f = function (p, t)
        fun_polarity = get_polarity(p, t)
        β, λ = _map_barriers_to_coefficients(p.ΔW_on, p.ΔW_off)
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