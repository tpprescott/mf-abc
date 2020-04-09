# Single Cell Simulator
module SingleCell

using Polynomials
using Roots: find_zero
using DifferentialEquations: SDEProblem, solve, SOSRA

using ..LikelihoodFree
import ..NoEF_Experiment
import ..SingleCellModel_NoEF

const f = Poly([-4.0, 3.0])
const g = poly([1.0, 1.0, 4.0])

export SingleCellSimulator_NoEF
abstract type SingleCellSimulator_NoEF{Y} <: AbstractSimulator{SingleCellModel_NoEF, NoEF_Experiment, Y} end


function _map_barriers_to_coefficients(EB_on::Float64, EB_off::Float64)::NTuple{2,Float64}
    h = f + (1-EB_off/EB_on)*g
    λ = find_zero(h, (1,2))
    β = -12.0 * EB_on / g(λ)
    return β, λ
end
function ∇W(β::Float64, λ::Float64, x::Complex{Float64})::Complex{Float64}
    return β*(abs2(x)-1)*(abs2(x)-λ+1)*x
end

function drift!(du, u, p, t)
    du[1] = p[:v] * u[2]
    du[2] = -∇W(p[:β], p[:λ], u[2])
    return nothing
end
function noise!(du, u, p, t)
    du[2] = p[:σ]
    return nothing
end
function initial_conditions(sigma::T) where T<:Real
    return [complex(0.), sigma*complex(randn(), randn())]
end

struct SC_NoEF_Displacements <: SingleCellSimulator_NoEF{Array{Float64,1}}
    prob::SDEProblem
    function SC_NoEF_Displacements(u::NoEF_Experiment)
        prob = SDEProblem(drift!, noise!, initial_conditions(u.σ_init), (0.0, u.maxT), Dict(:v=>1.0, :σ=>1.0, :λ=>1.3, :β=>10.0), noise_rate_prototype=[0.0,1.0])
        return new(prob)
    end
end

function (F::SingleCellSimulator_NoEF{Y})(m, u) where Y
    F.prob.p[:v] = m[:polarised_speed]
    F.prob.p[:σ] = m[:σ]
    F.prob.p[:β], F.prob.p[:λ] = _map_barriers_to_coefficients(0.5*m[:EB_on]*m[:σ]^2, 0.5*m[:EB_off]*m[:σ]^2)
    F.prob.x0 .= initial_conditions(u.σ_init)
    nCell = size(F.prob.u0,2)
    sol = solve(F.prob, SOSRA(), saveat=u.t_obs, save_idxs = 1:2:2*nCell)
    
    x = complex.(zeros(length(u.t_obs), nCell))
    for i in eachindex(sol)
        x[i,:] .= sol[i]
    end
    return Y(x, sol.t)
end

end