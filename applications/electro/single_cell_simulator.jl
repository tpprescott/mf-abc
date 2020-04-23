using DifferentialEquations

export AbstractEMField, NoEF

# For every concrete Field<:EMField we need to be able to call
# (emf::Field)(t::Float64)::Complex{Float64}
abstract type AbstractEMField end
function (emf::AbstractEMField)(du, u, p, t)
    drift_NoEF!(du, u, p, t)
    du[2] += p[:γ]*emf(t)
    return nothing
end

export NoEF, ConstantEF, StepEF

struct NoEF <: AbstractEMField end
const NOFIELD = NoEF()
(::NoEF)(t) = complex(0.0)
(::NoEF)(du, u, p, t) = drift_NoEF!(du, u, p, t)

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

const noise_shape = [complex(0.0), complex(1.0)]
struct SingleCellSimulator <: AbstractSimulator
    prob::SDEProblem
    σ_init::Float64
    tspan::Tuple{Float64, Float64}
    saveat::Array{Float64,1}
    # Summary functions
    displacements::Bool
    angles::Bool

    function SingleCellSimulator(;
        displacements::Bool=true,
        angles::Bool=false,
        σ_init::Float64=0.0, 
        tspan::Tuple{Float64,Float64}=(0.0, 180.0), 
        saveat::Array{Float64,1}=collect(0.0:5.0:180.0),
        emf::F=NOFIELD,
        ) where F<:AbstractEMField

        prob = SDEProblem(
            emf,
            noise!, 
            [complex(0.0), complex(0.0)], 
            tspan, 
            Dict(:v=>1.0, :σ=>1.0, :λ=>1.3, :β=>10.0, :γ=>0.0),
            noise_rate_prototype=noise_shape,
            )
        
        return new(prob, σ_init, tspan, saveat, displacements, angles)
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
function initial_conditions!(x0::Array{Complex{Float64}, 1}, sigma)
    x0[2] = sigma*complex(randn(), randn())
    return nothing
end

function (F::SingleCellSimulator)(; polarised_speed::Float64, σ::Float64, EB_on::Float64, EB_off::Float64, EF_bias::Float64=0.0, kwargs...)
    F.prob.p[:v] = polarised_speed
    F.prob.p[:σ] = σ
    F.prob.p[:β], F.prob.p[:λ] = _map_barriers_to_coefficients(EB_on, EB_off, σ)
    F.prob.p[:γ] = 0.5*EF_bias*σ^2
    F.prob.u0[2] = F.σ_init*complex(randn(), randn())

    sol = solve(F.prob, saveat=F.saveat, save_idxs=1, save_noise=true)
    summary = Array{Float64,1}()
    F.displacements && append!(summary, get_displacements(sol.u))
    F.angles && append!(summary, get_angles(sol.u))
    isempty(summary) && error("Nothing returned by simulation")
    return (y=summary, u0=copy(sol.prob.u0), W=NoiseWrapper(sol.W))
end