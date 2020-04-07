include("MultiFidelityABC.jl")
mkpath("poster_figures")

using StatsPlots, Random

println("#### Repressilator")
println("# Loading data")
bm = MakeBenchmarkCloud("repressilator/output")
epsilons = (50.0,50.0)
sample_size = 10^4

println("# MF Distances")
fig1a = view_distances(bm[1:sample_size], epsilons)
savefig(plot(fig1a, size=(512,550)), "poster_figures/distances.pdf")

println("# MF Tuning/Results")
fig2a = compare_efficiencies(bm, sample_size, epsilons, output="theory")
fig2b, table1, latex_table1 = compare_efficiencies(bm, sample_size, epsilons, output="plot")
savefig(plot(fig2a, fig2b, layout=2, size=(1100,550)), "poster_figures/tuned.pdf")

println("# ABC and Estimators")
figA = view_distances(bm[1:sample_size], epsilons, 2, L"n")

function get_n(bm, i::Int64, epsilons::NTuple{2,Float64})
    return [p.k[2] for p in bm if p.dist[i]<epsilons[i]]
end

s = get_n(bm[1:sample_size], 2, epsilons)
mean(x) = sum(x)/length(x)

figB = plot(s, seriestype=:histogram,
    title="Empirical posterior and mean", 
    xlabel=L"n",
    ylabel="",
    yticks=[],
    legend=:none,
    normalize=:pdf,
    bins=20)
figB = vline!([mean(s)], linewidth=3, linestyle=:dash, color=[:black], label="")

# Get the times and the estimates
bm_p = Iterators.partition(bm, sample_size)

N = length(bm_p)
v1 = Array{Float64, 1}(undef, N) 
v2 = Array{Float64, 1}(undef, N)
v3 = Array{Float64, 1}(undef, N)
t1 = Array{Float64, 1}(undef, N) 
t2 = Array{Float64, 1}(undef, N)
t3 = Array{Float64, 1}(undef, N)

function get_n(mfcloud)
    return sum([p.w * p.p.k[2] for p in mfcloud])/sum([p.w for p in mfcloud])
end

for (i, bm_subsample) in Iterators.enumerate(bm_p)
    r = collect(bm_subsample)

    m1 = MakeMFABCCloud(r, epsilons, (1.0, 1.0))
    m2 = MakeMFABCCloud(r, epsilons, (0.0, 0.0))
    m3 = MakeMFABCCloud(r, epsilons, (0.1, 0.1)) 

    v1[i] = get_n(m1)
    v2[i] = get_n(m2)
    v3[i] = get_n(m3)
    t1[i] = cost(m1)
    t2[i] = cost(m2)
    t3[i] = cost(m3)
end

figC = plot(; xlabel=L"n", ylabel="Simulation time (s)", title="Empirical posterior means", legend=:none)
figC = scatter!(v1, t1, label="High fidelity")

figD = plot(; xlabel=L"n", ylabel="Simulation time (s)", title="Comparing posterior means", legend=:right)
figD = scatter!(v1, t1, label="High fidelity")
figD = scatter!(v2, t2, label="Low fidelity")
figD = scatter!(v3, t3, label="Multifidelity")

savefig(plot(figA, size=(550, 354)), "poster_figures/simulationdistances.pdf")
savefig(plot(figB, size=(550, 354)), "poster_figures/histogram.pdf")
savefig(plot(figC, size=(550, 354)), "poster_figures/highfidelity_estimates.pdf")
savefig(plot(figD, size=(550, 354)), "poster_figures/multifidelity_estimates.pdf")