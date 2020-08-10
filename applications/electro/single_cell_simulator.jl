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
    f = function (pos, par, t)
        return par.v * par.γ1 * EMF(t)
    end
    return f
end

alignment(z1::Complex, z2::Complex) = Float64((dot(z1,z2)+dot(z2,z1))/2)
function v_cell(EMF::AbstractEMField)
    f = function (pos, (pol, par), t)
        u = EMF(t)
        pol_t = pol(t)
        if iszero(pol_t)
            out = pol_t
        else
            out = par.v * pol_t
            if !iszero(u)
                polhat = pol_t/abs(pol_t)
                out *= (1 + par.γ2*abs(u) + par.γ3*alignment(u, polhat))
            end
        end
        return out
    end
    return f
end
function v_cell(EMF::AbstractEMField, pol::Complex{Float64})
    ff = v_cell(EMF)
    pol_fun = (t -> pol)
    f = function (pos, par, t)
        return ff(pos, (pol_fun, par), t)
    end
    return f
end

function velocity(EMF::AbstractEMField)
    _v_EM = v_EM(EMF)
    _v_cell = v_cell(EMF)
    f = function (pos, (pol, par), t)
        component_EM = _v_EM(pos, par, t)
        component_cell = _v_cell(pos, (pol, par), t)
        return component_EM + component_cell
    end
    return f
end

function polarity(EMF::AbstractEMField)
    _v_EM = v_EM(EMF)
    f = function (pos, par, t)
        offset_EM = _v_EM(pos, par, t)
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


#############################################################################
# SIMULATOR
#############################################################################
function couple_noise(ind_flags, u0, W; ic_sigma=0.1)
    fun = function(prob, i, repeat)
        if ind_flags[i]
            remake(prob, u0=initial_conditions(ic_sigma))
        else
            remake(prob, u0=u0[i], noise= NoiseWrapper(W[i]))
        end
    end
    return fun
end
function prob_fun_x(sol_p)
    fun = function(prob, i, repeat)
        remake(prob, p=(sol_p[i], prob.p[2]))
    end
    return fun
end

function output_fun_x(tvec, W_previous, ind_flags, output_trajectory)
    if output_trajectory
        fun = ((sol, i) -> (sol, false))
    else
        fun = function (sol, i)
            summary = get_displacements(sol.(tvec))
            W = ind_flags[i] ? sol.prob.p[1].W : W_previous[i]
            u0 = sol.prob.p[1].prob.u0
            return (y = summary, u0=u0, W=W), false
        end
    end
    return fun
end

const B_t = WienerProcess(0.0, complex(0.0))

function (F::SingleCellSimulator)(n::Int64 = 1;
    v::Float64, EB_on::Float64, EB_off::Float64, D::Float64,
    γ1::Float64=0.0, γ2::Float64=0.0, γ3::Float64=0.0, γ4::Float64=0.0, 
        ind_flags::Array{Bool, 1} = fill(true, n), 
        u0 = fill(complex(0.0), n), 
        W = fill(B_t, n),
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
    
    prob_p_nominal = SDEProblem(
        drift(F.emf), 
        noise, 
        complex(0.0), 
        F.tspan, 
        parm_p,
    )
    prob_p = EnsembleProblem(
        prob_p_nominal, 
        prob_func = couple_noise(ind_flags, u0, W),
    )

    sol_p = solve(prob_p, trajectories=n, save_noise=true)

    # Integrate to get position
    parm_x = (
        v=v,
        γ1=γ1,
        γ2=γ2,
        γ3=γ3,
    )

    prob_x_nominal = ODEProblem(
        velocity(F.emf),
        complex(0.0), 
        F.tspan, 
        (t->complex(0.0), parm_x),
    )

    prob_x = EnsembleProblem(prob_x_nominal,
        prob_func = prob_fun_x(sol_p),
        output_func = output_fun_x(F.saveat, W, ind_flags, output_trajectory),
    )

    sol = solve(prob_x, saveat=F.saveat, trajectories=n)
    return sol.u
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