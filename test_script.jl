struct Tom <: AbstractModel
    x::Float64
end

struct TomQ <: AbstractGenerator{Tom} end
(::TomQ)() = Tom(rand())
q = TomQ()

TomU = ExperimentalData{Float64}
u = TomU(0.4)

struct TomY <: AbstractSummaryStatisticSpace
    y::Float64
end

struct TomF <: AbstractSimulator{Tom, TomU, TomY} end
(::TomF)(m::Tom, u::TomU)::TomY = TomY(m.x + randn())
F = TomF()

struct TomD <: AbstractDistance{TomU, TomY} end
(::TomD)(u::TomU, y::TomY) = abs(u.y_obs - y.y)
d = TomD()

c = ABCComparison(0.1, d)
w = LikelihoodFreeWeight(F,c)

m = rand(q)
mm = rand(q, 20)