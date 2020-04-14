using DifferentialEquations: SDEProblem, solve

using ..LikelihoodFree

export SingleCellSimulator
abstract type SingleCellSimulator{M, T} <: AbstractSimulator{M, T} end

export SC_NoEF_Displacements
struct SC_NoEF_Displacements <: SingleCellSimulator{SingleCellModel_NoEF, Float64}
    prob::SDEProblem
    σ_init::Float64
    tspan::Tuple{Float64, Float64}
    saveat::Array{Float64,1}
    function SC_NoEF_Displacements(; σ_init::Float64=0.0, tspan::Tuple{Float64,Float64}=(0.0, 180.0), saveat::Array{Float64,1}=collect(0.0:5.0:180.0))
        prob = SDEProblem(drift_NoEF!, noise!, [complex(0.0), complex(0.0)], tspan, Dict(:v=>1.0, :σ=>1.0, :λ=>1.3, :β=>10.0), noise_rate_prototype=[0.0,1.0])
        return new(prob, σ_init, tspan, saveat)
    end
end
import .LikelihoodFree.output_dimension
output_dimension(::SC_NoEF_Displacements) = 3

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

function (F::SC_NoEF_Displacements)(y::AbstractArray{Float64, 1}, m::SingleCellModel_NoEF)
    F.prob.p[:v] = m[:polarised_speed]
    F.prob.p[:σ] = m[:σ]
    F.prob.p[:β], F.prob.p[:λ] = _map_barriers_to_coefficients(m[:EB_on], m[:EB_off], m[:σ])
    initial_conditions!(F.prob.u0, F.σ_init)

    sol = solve(F.prob, saveat=F.saveat, save_idxs=1)
    get_displacements!(y, sol.u)
end