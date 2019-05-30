include("MultiFidelityABC.jl")
mkpath("figures")

using StatsPlots

println("#### Repressilator")
println("# Loading data")
bm = MakeBenchmarkCloud("repressilator/output")
epsilons = (50.0,50.0)
sample_size = 10^4

println("# Fig 1")
fig1a = view_distances(bm[1:sample_size], epsilons)
fig1b = view_distances(bm[1:sample_size], epsilons, 2, L"n")
savefig(plot(fig1a, fig1b, layout=2, size=(1000,400)), "figures/fig1.pdf")

println("# Fig 2 and Table 1")
fig2a = compare_efficiencies(bm, sample_size, epsilons, output="theory")
fig2b, table1, latex_table1 = compare_efficiencies(bm, sample_size, epsilons, output="plot")
savefig(plot(fig2a, fig2b, layout=2, size=(1000,400)), "figures/fig2.pdf")

println("# Table 2 and 3")
variances, phis, etas = observed_variances(bm, sample_size, epsilons, Repressilator.F, [100.0, 200.0, 300.0])

error("OK UP TO HERE!")

println("#### Viral")
println("# Load data")
bm = MakeBenchmarkCloud("viral/output")
mf_location = "viral/output/mf/"
mf_cloud_idx = readdir(mf_location)
mf_set = [MakeMFABCCloud(mf_location*cloud_id) for cloud_id in mf_cloud_idx]
epsilons = (0.4,0.4)
sample_size = 10^3

println("# Fig 3")
eps(view_distances(bm[1:10000], epsilons), "figures/fig3")

println("# Fig 4")
eps(plot_eta_estimates(mf_set, bm, epsilons; method="mf", lower_eta=0.1), "figures/fig4")

println("# Fig 5")
bm_set = [cld[1:sample_size] for cld in mf_set]
inc_set = [cld[sample_size+1:end] for cld in mf_set]
eps(plot_apost_efficiencies(inc_set, bm_set), "figures/fig5")