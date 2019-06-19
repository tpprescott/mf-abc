# mf-abc
Multifidelity ABC: https://arxiv.org/abs/1811.09550

Load functions by running `include("MultiFidelityABC.jl")`. This introduces three modules:
1. `Repressilator`
1. `Viral`
1. `MultiFidelityABC`

The first two modules are implementations of the two examples found in the paper, and the code can be found in `Repressilator.jl` and `Viral.jl` respectively.

The third module is the main part of the code, and split into three:
1. `simmethods.jl` implements the simulation approaches (Gillespie algorithm, tau leap, and hybrid deterministic/stochastic simulation. This is done by creating three types `gm`, `tlm`, `hm` respectively and writing down the `simulate` and `complete` functions for each. Note that `simulate` simulates the model from nothing, while `complete` takes a coupling output from a low-fidelity simulation (i.e. tau leap or hybrid) and generates a Gillespie completion.
1. `mfabc.jl` implements multifidelity ABC algorithms.
1. `produce_figs.jl` implements the figures used in the paper.

We have saved the simulations used in the paper in the `repressilator` and `viral` folders. Running the script `figure_script.jl` will reproduce the figures in the paper based on these saved simulations.

**Note:** We have recently discovered that Git LFS charges for storage. We have also put the [repressilator](https://cloud.maths.ox.ac.uk/index.php/s/nwM5DfwqG5wYrb9 "Repressilator Simulations")  and [viral](https://cloud.maths.ox.ac.uk/index.php/s/Qx5Wo4rYW4bTtKr "Viral Dynamics Simulations") simulation data into separate cloud storage.

## mfabc.jl
### Types

```
struct MFABC
    parameter_sampler::Function 
    lofi::Function              
    hifi::Function              
end
```
* Parameter sampler generates a parameter
* lofi is a map from the parameter to a (low fidelity) output and a coupling
* hifi is a map from the parameter and the coupling to a (high fidelity) output

```
struct Particle{N}
    k::Parameters
    dist::NTuple{N,Float64}
    cost::NTuple{N,Float64}
end
```
* Parameterised by integer `N`, intended to take values 1 or 2
* `k` is the parameter proposal for the particle
* `dist` is a vector of distances between 1 or 2 simulations and the data
* `cost` is a vector of the costs of 1 or 2 simulations

```
struct MFABCParticle
    p::Particle
    eta::Float64
    w::Float64
end
```
* MFABC algorithm applied to a particle
* Records the continuation probability used in generating the particle (whether to use one or two simulations)
* Records the acceptance decision as a weight

```
BenchmarkCloud = Array{Particle{2}, 1}
MFABCCloud = Array{MFABCParticle, 1}
Cloud = Union{BenchmarkCloud, MFABCCloud}
```
* Benchmarking requires both simulations
* General MFABC cloud is a vector of MFABC particles.

### Functions
```
get_eta(p_tp::Float64, p_fp::Float64, p_fn::Float64, ct::Float64, c_p::Float64, c_n::Float64; method)
get_eta(s::Union{Array{MFABCParticle,1}, Array{Particle{2},1}}, epsilons::Tuple{Float64,Float64}; method, lower_eta, kwargs...)
get_eta(bm::Array{Particle{2},1}, epsilons::Tuple{Float64,Float64}, F::Array{Function,1})
```
Find the optimal values of ![](https://latex.codecogs.com/svg.latex?\inline&space;\eta_1) and ![](https://latex.codecogs.com/svg.latex?\inline&space;\eta_2).

1. Find values given known true/false positive/negative rates and computational costs. `method` is a string that can take values `"mf"` (multifidelity: default), `"er"` (early rejection),`"ed"` (early decision),`"abc"` (rejection sampling).
1. Second method finds these rates and costs for a benchmark set, given values of ABC thresholds. We can optionally set lower bounds on the continuation probabilities, and the other keyword argument is `F`, to weight the true/false positives/negatives by their effect on the estimate of a specified function F of the unknown parameters. 
1. The final signature implements seeking continuation probabilities for an array of functions (specific to task in paper).

```
phi(eta, p_tp::Float64, p_fp::Float64, p_fn::Float64, ct::Float64, c_p::Float64, c_n::Float64)
phi(eta, s::Union{Array{MFABCParticle,1}, Array{Particle{2},1}}, epsilons::Tuple{Float64,Float64}; kwargs...)
```
Return the value of the function being minimised by the optimal continuation probabilities, given the true/false positive/negative rates or the set of completed simulations (and threshold values, and potentially function F of unknown parameters)

```
MakeMFABCCloud(mfabc::MFABC, epsilons::Tuple{Float64,Float64}, etas::Tuple{Float64,Float64}; kwargs...)
MakeMFABCCloud(mfabc::MFABC, s::Array{Particle{2},1}, epsilons::Tuple{Float64,Float64}; method, kwargs...)
MakeMFABCCloud(mfabc::MFABC, epsilons::Tuple{Float64,Float64}; burnin, method, kwargs...)
MakeMFABCCloud(s::Array{Particle{2},1}, epsilons::Tuple{Float64,Float64}, etas::Tuple{Float64,Float64})
MakeMFABCCloud(s::Array{Particle{2},1}, epsilons::Tuple{Float64,Float64}, etas::Tuple{Float64,Float64}, budget::Float64)
MakeMFABCCloud(s::Array{Particle{2},1}, epsilons::Tuple{Float64,Float64}; method, kwargs...)
MakeMFABCCloud(indir::String)
```
Take an MFABC problem (parameter sampler, low fidelity model, and high fidelity model) and produce a cloud of MFABC particles.
1. Set thresholds, fix continuation probabilities, possible keyword arguments are (one has to be specified, `N` takes precedence):
   1. `N` - number of parameters to sample from `parameter_sampler`
   1. `budget` - total computational cost allowed
   1. Note that this function uses the parallel loop `pmap` if Julia has been started with multiple workers, to exploit the _embarassingly parallel_ nature of rejection sampling.
1. Use the benchmark set `s` to generate continuation probabilities, optimised using `method` keyword, and then generate new samples using the MFABC problem (adaptively evolving continuation probabilities as more simulations completed).
1. Generate a benchmark set of size `burnin`, to put into preceding method
1. Convert a benchmark set into an MFABC set using the fixed values of `etas`
1. As previously, but truncate once the set reaches a specified computational budget
1. Convert the benchmark set into an MFABC set using optimised values of `etas` based on `method` from `"mf"` (multifidelity: default), `"er"` (early rejection),`"ed"` (early decision),`"abc"` (rejection sampling).
1. Read a previously completed MFABC cloud from data: `ìndir` specifies the directory containing the files required.

```
MakeBenchmarkCloud(mfabc::MFABC, N::Int64)
MakeBenchmarkCloud(mfabc::MFABC, N::Int64, fn::String)
MakeBenchmarkCloud(indir::String)
```
1. Generate a benchmark cloud (i.e. both simulations completed) of `N` particles
   * Note that this function uses the parallel loop `pmap` if Julia has been started with multiple workers, to exploit the _embarassingly parallel_ nature of rejection sampling.
1. As above, but write to a folder
1. Read a folder into a benchmark cloud

```
cost(p::Particle)
cost(p::Particle, i::Integer)
cost(pp::MFABCParticle)
cost(pp::MFABCParticle, i)
cost(c::Union{Array{MFABCParticle,1}, Array{Particle{2},1}})
cost(c::Union{Array{MFABCParticle,1}, Array{Particle{2},1}}, i)
```
* Return the computational cost of a particle.
* Return the cost of a cloud.
* `i=1` specifies total time spent simulating the low fidelity model and `ì=2` specifies total time spent on the high fidelity model: otherwise these are summed.

```
length(p::Particle) = length(p.cost)
length(pp::MFABCParticle) = length(pp.p)
length(c::Cloud, i::Integer) = length(filter(p->(length(p)==i), c))
```
1. Length of a particle is the number of simulations (low fidelity only returns `1`, both simulations returns `2`).
1. Length of an MFABC particle is the length of its particle.
1. Length of any cloud is the number of particles it contains.
   * Specifying `i` counts how many particles in the cloud are of a specific length
   
## produce_figs.jl

### Functions

```
estimate_mu(s::Array{MFABCParticle,1}, parameterFun::Function)
```
Produce an estimate of a function of the unknown parameters, given the MFABC cloud.

```
ESS(s::Array{MFABCParticle,1})
```
Return the effective sample size of the MFABC cloud.

### Figures
The remaining exported functions are specific to producing the figures in the paper, and are less relevant for practically working with the output of a multifidelity ABC algorithm.
