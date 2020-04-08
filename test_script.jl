Tom = NamedTuple{(:x,), Tuple{Float64,}}
Tom2 = NamedTuple{(:a, :b), Tuple{Float64, Float64}}

using Distributions: Uniform, MvNormal, Normal

q = DistributionGenerator(Tom, Uniform(0,1))
q2 = DistributionGenerator(Tom2, MvNormal(2, 1.0))

TomU = ExperimentalData{Float64}
u = TomU(0.4)

struct TomF <: AbstractSimulator{Tom, TomU, Float64} end
(::TomF)(m::Tom, u::TomU)::Float64 = m.x + randn()
F = TomF()
struct TomF2 <: AbstractSimulator{Tom2, TomU, Float64} end
(::TomF2)(m::Tom2, u::TomU)::Float64 = Float64(m.a + m.b*randn())
F2 = TomF2()


struct TomD <: AbstractDistance{TomU, Float64} end
(::TomD)(u::TomU, y::Float64) = abs(u.y_obs - y)
d = TomD()

c = ABCComparison(d, 0.1)
w = ABCWeight(F, d, 0.1) # Same thing as LikelihoodFreeWeight(F,c)
w2 = ABCWeight(F2, d, 0.1)

### MF

F_1 = TomF(); F_2 = TomF();
W1 = ABCWeight(F_1, d, 0.1); W2 = ABCWeight(F_2, d, 0.1);
η = EarlyAcceptReject{Tom}(0.3, 0.5, d, 0.1)

mf_w = MFABCWeight(W1, W2, η)

## Sequential importance sampling

using Distributions: Normal, MvNormal, Uniform

out = rejection_sample(u, q, w,400)
X = SequentialImportanceDistribution(out[:ww], out[:mm], q)

out2 = rejection_sample(u, q2, w2, 400)
X2 = SequentialImportanceDistribution(out2[:ww], out2[:mm], q2)