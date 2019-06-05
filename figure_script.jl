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
savefig(plot(fig1a, fig1b, layout=2, size=(900,360)), "figures/fig1.pdf")

println("# Fig 2 and Table 1")
fig2a = compare_efficiencies(bm, sample_size, epsilons, output="theory")
fig2b, table1, latex_table1 = compare_efficiencies(bm, sample_size, epsilons, output="plot")
savefig(plot(fig2a, fig2b, layout=2, size=(1100,440)), "figures/fig2.pdf")

println("# Table 2 and 3")
variances, phis, etas = observed_variances(bm, sample_size, epsilons, Repressilator.F, [100.0, 200.0, 300.0])

println("#### Viral")
println("# Load data")
bm = MakeBenchmarkCloud("viral/output")
mf_location = "viral/output/mf/"
mf_cloud_idx = readdir(mf_location)
mf_set = [MakeMFABCCloud(mf_location*cloud_id) for cloud_id in mf_cloud_idx]
epsilons = (0.25,0.25)
eta_0 = 0.01
sample_size = 10^3

println("# Fig 3")
savefig(view_distances(bm[1:10000], epsilons, epsilons.*2), "figures/fig3.pdf")

println("# Fig 4")
fig4 = plot_eta_estimates(mf_set, bm, epsilons; method="mf", lower_eta=eta_0)
plot!(xlim=(0,0.4),ylim=(0,0.4))
savefig(fig4, "figures/fig4.pdf")

println("# Fig 5")
bm_set = [cld[1:sample_size] for cld in mf_set]
inc_set = [cld[sample_size+1:end] for cld in mf_set]
savefig(plot_apost_efficiencies(inc_set, bm_set), "figures/fig5.pdf")
