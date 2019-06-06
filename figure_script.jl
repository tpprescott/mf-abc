include("MultiFidelityABC.jl")
mkpath("figures")

using StatsPlots, Random

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
eta_tab, phi_tab = get_eta(bm, epsilons, Repressilator.F)
Random.seed!(123)
var_tab = variance_table(bm, 10^3, epsilons, eta_tab, Repressilator.F, 30.0)
Random.seed!()

println("#### Viral")
println("# Load data")

bm = MakeBenchmarkCloud("viral/output")
mf_smallBI_location = "./viral/output/mf_smallBI/"
mf_largeBI_location = "./viral/output/mf_largeBI/"

epsilons = (0.25,0.25)
eta_0 = 0.01
smallBI_size = 10^3
largeBI_size = length(bm)

function divide_cloud(c::MFABCCloud, s::Integer; stage::String)
    if stage=="bm"
        return c[1:s]
    elseif stage=="inc"
        return c[s+1:end]
    end
    error("What stage? bm or inc")
end

bm_set = Array{MFABCCloud,1}()
mf_set = Array{MFABCCloud,1}()
inc_smallBI_set = Array{MFABCCloud,1}()
inc_largeBI_set = Array{MFABCCloud,1}()

for cloud_location in mf_smallBI_location.*readdir(mf_smallBI_location)
    c = MakeMFABCCloud(cloud_location)
    push!(mf_set, c)
    push!(bm_set, divide_cloud(c, smallBI_size, stage="bm"))
    push!(inc_smallBI_set, divide_cloud(c, smallBI_size, stage="inc"))
end
for cloud_location in mf_largeBI_location.*readdir(mf_largeBI_location)
    c = MakeMFABCCloud(cloud_location)
    push!(inc_largeBI_set, divide_cloud(c, largeBI_size, stage="inc"))
end

println("# Fig 3")
savefig(view_distances(bm[1:10000], epsilons, epsilons.*2), "figures/fig3.pdf")

println("# Fig 4")
fig4 = plot_eta_estimates(mf_set, bm, epsilons; method="mf", lower_eta=eta_0)
plot!(xlim=(0,0.4),ylim=(0,0.4))
savefig(fig4, "figures/fig4.pdf")

println("# Fig 5")
savefig(plot_apost_efficiencies(("After large burn-in","After small burn-in","During burn-in"),inc_largeBI_set,inc_smallBI_set,bm_set), "figures/fig5.pdf")
