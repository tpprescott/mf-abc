using DifferentialEquations: SDEProblem, solve

export AbstractEMField, NoEF

# For every concrete Field<:EMField we need to be able to call
# (emf::Field)(t::Float64)::Complex{Float64}
abstract type AbstractEMField end
function (emf::AbstractEMField)(du, u, p, t)
    drift_NoEF!(du, u, p, t)
    du[2] += p[:γ]*emf(t)
    return nothing
end

struct NoEF <: AbstractEMField end
const NOFIELD = NoEF()
(::NoEF)(t) = complex(0.0)
(::NoEF)(du, u, p, t) = drift_NoEF!(du, u, p, t)

export SingleCellSimulator
# abstract type SingleCellSimulator{M, T, F<:EMField} <: AbstractSimulator{M, T} end

const noise_shape = [complex(0.0), complex(1.0)]
struct SingleCellSimulator{M<:SingleCellModel} <: AbstractSimulator{M, Float64}
    prob::SDEProblem
    σ_init::Float64
    tspan::Tuple{Float64, Float64}
    saveat::Array{Float64,1}
    function SingleCellSimulator(;
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
        
        M = typeof(emf)==NoEF ? SingleCellModel_NoEF : SingleCellModel_EF
        return new{M}(prob, σ_init, tspan, saveat)
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
function initial_conditions!(x0::Array{Complex{Float64}, 1}, sigma::T) where T<:Real
    x0[2] = sigma*complex(randn(), randn())
    return nothing
end

function (F::SingleCellSimulator)(y::AbstractArray{Float64, 1}, m::SingleCellModel_NoEF)
    F.prob.p[:v] = m[:polarised_speed]
    F.prob.p[:σ] = m[:σ]
    F.prob.p[:β], F.prob.p[:λ] = _map_barriers_to_coefficients(m[:EB_on], m[:EB_off], m[:σ])
    initial_conditions!(F.prob.u0, F.σ_init)

    sol = solve(F.prob, saveat=F.saveat, save_idxs=1)
    get_displacements!(y, sol.u)
end
function (F::SingleCellSimulator{SingleCellModel_EF})(y::AbstractArray{Float64, 1}, m::SingleCellModel_EF)
    F.prob.p[:v] = m[:polarised_speed]
    F.prob.p[:σ] = m[:σ]
    F.prob.p[:β], F.prob.p[:λ] = _map_barriers_to_coefficients(m[:EB_on], m[:EB_off], m[:σ])
    F.prob.p[:γ] = 0.5*m[:EF_bias]*m[:σ]^2
    initial_conditions!(F.prob.u0, F.σ_init)

    sol = solve(F.prob, saveat=F.saveat, save_idxs=1)
    get_displacements!(view(y, 1:3), sol.u)
    get_angles!(view(y, 4:6), sol.u)
end
import .LikelihoodFree.output_dimension
output_dimension(::SingleCellSimulator{SingleCellModel_NoEF}) = 3
output_dimension(::SingleCellSimulator{SingleCellModel_EF}) = 6