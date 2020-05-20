using DifferentialEquations
using LinearAlgebra
alignment(z1::Complex, z2::Complex) = (dot(z1,z2)+dot(z2,z1))/2

export AbstractEMField, NoEF

# For every concrete Field<:AbstractEMField we need to be able to call
# (emf::Field)(t::Float64)::Complex{Float64}
abstract type AbstractEMField end
function drift(emf::AbstractEMField)
    f = function (du, u, p, t)
        drift_NoEF!(du, u, p, t)
        input = emf(t)
        if !iszero(input)
            du[1] *= (1.0 + abs(input)*p[:γ_speed])
            du[1] += iszero(u[2]) ? 0.0 : p[:γ_alignment]*(alignment(u[2],input)/abs(u[2]))*u[2]
            du[1] += p[:γ_position] * input
            du[2] += p[:γ_polarity] * input
        end
    end
    return f
end 

export NoEF, ConstantEF, StepEF

struct NoEF <: AbstractEMField end
const NOFIELD = NoEF()
drift(::NoEF) = drift_NoEF!
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

export SingleCellSimulator
# abstract type SingleCellSimulator{M, T, F<:EMField} <: AbstractSimulator{M, T} end

struct SingleCellSimulator{EMF<:AbstractEMField} <: AbstractSimulator
    emf::EMF
    σ_init::Float64
    tspan::Tuple{Float64, Float64}
    saveat::Array{Float64,1}
    # Summary functions
    displacements::Bool
    angles::Bool

    function SingleCellSimulator(;
        emf::F=NOFIELD,
        displacements::Bool=true,
        angles::Bool=false,
        σ_init::Float64=0.0, 
        tspan::Tuple{Float64,Float64}=(0.0, 180.0), 
        saveat::Array{Float64,1}=collect(0.0:5.0:180.0),
        ) where F<:AbstractEMField
        
        return new{F}(emf, σ_init, tspan, saveat, displacements, angles)
    end
end

function ∇W(β::Float64, λ::Float64, x::Complex{Float64})::Complex{Float64}
    return β*(abs2(x)-1)*(abs2(x)-λ+1)*x
end
function drift_NoEF!(du, u, p, t)
    du[1] = p[:v] * u[2]
    du[2] = -∇W(p[:β], p[:λ], u[2])
    return nothing
end
function noise!(du, u, p, t)
    du[2] = p[:σ]
    return nothing
end

function _map_barriers_to_coefficients(EB_on::Float64, EB_off::Float64, σ::Float64)::NTuple{2,Float64}
    λ = get_λ(EB_on, EB_off)
    β = get_β(EB_on, σ, λ)
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
function get_β(EB_on, σ, λ)
    return -6.0 * (σ^2) * EB_on / (((λ-1)^2)*(λ-4))
end
function initial_conditions(sigma)
    return [complex(0), sigma*complex(randn(), randn())]
end

const noise_shape = [complex(0.0), complex(1.0)]
function (F::SingleCellSimulator)(; 
    polarised_speed::Float64, σ::Float64, EB_on::Float64, EB_off::Float64,
    EF_speed_change::Float64=0.0, EF_polarity_bias::Float64=0.0, EF_position_bias::Float64=0.0, EF_alignment_bias::Float64=0.0,
    u0::Array{Complex{Float64},1}=initial_conditions(F.σ_init), W=nothing,
    kwargs...)

    β, λ = _map_barriers_to_coefficients(EB_on, EB_off, σ)
    p = (
        v=polarised_speed, 
        σ=σ, 
        β=β, 
        λ=λ, 
        γ_polarity = 0.5*EF_polarity_bias*σ^2, # Dimensionalise the parameter
        γ_speed = EF_speed_change, # Keep parameter nondimensional
        γ_position = EF_position_bias*polarised_speed, # Dimensionalise the parameter
        γ_alignment = EF_alignment_bias*polarised_speed, # Dimensionalise the parameter
    )
    
    independentFlag = W === nothing
    couple = independentFlag ? nothing : NoiseWrapper(W)
    prob = SDEProblem(drift(F.emf), noise!, u0, F.tspan, p, noise_rate_prototype=noise_shape, noise=couple)
    sol = solve(prob, saveat=F.saveat, save_idxs=1, save_noise = independentFlag)

    summary = Array{Float64,1}()
    sizehint!(summary, 6)
    F.displacements && append!(summary, get_displacements(sol.u))
    F.angles && append!(summary, get_angles(sol.u))
    isempty(summary) && error("Nothing returned by simulation")

    independentFlag && (W=sol.W)
    return (y=summary, u0=u0, W=W)
end
import Base.eltype
eltype(::Type{T}) where T<:SingleCellSimulator = NamedTuple{(:y, :u0, :W), Tuple{Array{Float64,1}, Array{Complex{Float64}, 1}, NoiseProcess}}
