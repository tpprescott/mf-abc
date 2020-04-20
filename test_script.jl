TomM = NamedTuple{(:x,), Tuple{Float64,}}
Tom2 = NamedTuple{(:a, :b), Tuple{Float64, Float64}}

using Distributions: Uniform, MvNormal, Normal

q = DistributionGenerator(TomM, Uniform(0,1))
q2 = DistributionGenerator(Tom2, MvNormal(2, 1.0))

y_obs = Array{Float64,1}([0.4])

struct TomF <: AbstractSimulator{Float64} end
function (::TomF)(y::AbstractArray{Float64,1}; x::Float64, pars...)::NamedTuple 
    n = randn()
    y[1] = x + n
    return (n=n, )
end
F = TomF()

#struct TomF2 <: AbstractSimulator{Tom2, TomU, Float64} end
#(::TomF2)(m::Tom2, u::TomU)::Float64 = Float64(m.a + m.b*randn())
#F2 = TomF2()

struct TomD <: AbstractDistance{Float64} end
(::TomD)(u, y) = sum(abs, u - y)
d = TomD()

c_abc = ABCComparison{Float64, TomD}(d, 0.1)
w_abc = ABCWeight(F, d, 0.1) # Same thing as LikelihoodFreeWeight(F,c_abc)

c_sb = SyntheticLikelihood{Float64}()
w_sb = LikelihoodFreeWeight(F, c_sb, 500)

# w2 = LikelihoodFreeWeight(F2, c)

### MF
break

F_1 = TomF(); F_2 = TomF();
W1 = LikelihoodFreeWeight(F_1, c); W2 = LikelihoodFreeWeight(F_2, c);
η = EarlyAcceptReject{Tom}(0.3, 0.5, d, 0.1)

mf_w = MFABCWeight(W1, W2, η)

## Sequential importance sampling

using Distributions: Normal, MvNormal, Uniform

out = rejection_sample(u, q, w,400)
X = SequentialImportanceDistribution(out[:ww], out[:mm], q)

out2 = rejection_sample(u, q2, w2, 400)
X2 = SequentialImportanceDistribution(out2[:ww], out2[:mm], q2)