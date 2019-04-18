using Distributed
@everywhere include("repressilator.jl")
@everywhere include("viral.jl")

r_mf = repressilator_mfabc_problem(repressilator_prior)
v_mf = viral_mfabc_problem(viral_prior)

# Test
r_out = get_benchmark(r_mf)
v_out = get_benchmark(v_mf)