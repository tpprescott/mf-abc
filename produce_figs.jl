using DelimitedFiles
using StatsBase, Statistics
using Random

export estimate_mu, ESS, estimator_variance
export compare_efficiencies, view_distances, observed_variances

######### Helper functions

function ESS(weights::Array{Float64,1})
    return sum(weights)^2/sum(weights.^2)
end
function ESS(s::Cloud{MFABCParticle})
    return ESS([p.w for p in s])
end

function ESS_vs_times(weights::Array{Float64,1}, costs::Array{Float64,1})
    return ESS(weights), sum(costs)
end
function ESS_vs_times(s::Cloud{MFABCParticle})
    return ESS_vs_times([p.w for p in s], [p.c for p in s])
end

function efficiency(s::Cloud{MFABCParticle})
    /(ESS_vs_times(s)...)
end

function estimate_mu(weights::Array{Float64,1}, ksample::Array{<:Parameters,1}, parameterFun::Function)
    F = [parameterFun(k) for k in ksample]
    return sum(weights.*F)/sum(weights)
end
function estimate_mu(s::Cloud{MFABCParticle}, parameterFun::Function)
    return estimate_mu([p.w for p in s], [p.k for p in s], parameterFun)
end

function estimator_variance(sset::Array{Cloud{MFABCParticle}, 1}, parameterFun::Function)
    return var(map(s->estimate_mu(s, parameterFun), sset))
end

####################################

using Plots, StatsPlots, KernelDensity, LaTeXStrings, Printf
plotlyjs()

#=eta_mf_Fi, phi_mf_Fi = zip(map(Fi->get_eta(bm, epsilons, Fi, method="mf"), F)...)
eta_er_Fi, phi_er_Fi = zip(map(Fi->get_eta(bm, epsilons, Fi, method="er"), F)...)
eta_ed_Fi, phi_ed_Fi = zip(map(Fi->get_eta(bm, epsilons, Fi, method="ed"), F)...) =#

function compare_efficiencies(bm::Cloud{BenchmarkParticle}, sim_sets::Array{Cloud{BenchmarkParticle},1}, epsilons::Tuple{Float64, Float64};
    output::String="plot")

    eta_mf, phi_mf = get_eta(bm, epsilons, method="mf")
    eta_er, phi_er = get_eta(bm, epsilons, method="er")
    eta_ed, phi_ed = get_eta(bm, epsilons, method="ed")
    eta_abc, phi_abc = get_eta(bm, epsilons, method="abc")
    # Get some other continuation probabilities
    eta_pp = 0.5.*eta_mf .+ 0.5.*(1,1)
    phi_pp = phi(eta_pp,bm,epsilons)
    eta_pm = 0.5.*eta_mf .+ 0.5.*(1,0)
    phi_pm = phi(eta_pm,bm,epsilons)
    eta_mp = 0.5.*eta_mf .+ 0.5.*(0,1)
    phi_mp = phi(eta_mp,bm,epsilons)
    eta_mm = 0.5.*eta_mf .+ 0.5.*(0,0)
    phi_mm = phi(eta_mm,bm,epsilons)

    str_vec = ["Early Accept/Reject", "Early Rejection", "Early Decision", "Rejection Sampling", "+/+", "-/+", "+/-", "-/-"]
    eta_vec = [eta_mf, eta_er, eta_ed, eta_abc, eta_pp, eta_mp, eta_pm, eta_mm]
    phi_vec = [phi_mf, phi_er, phi_ed, phi_abc, phi_pp, phi_mp, phi_pm, phi_mm]
    I = sortperm(phi_vec)


    Random.seed!(888)
    clouds = [[MFABCCloud(s, epsilons, etas) for s in sim_sets] for etas in eta_vec]
    Random.seed!()
    efficiencies = [efficiency.(clds) for clds in clouds]
    
    if output=="plot"
        fig = plot(color_palette=:darkrainbow,
            thickness_scaling=1,
            xlabel="Effective samples per second",
            xtickfontsize=10,
            yticks=[],
            xgrid=false,
            legend=(0.7,0.25),
            legendfontsize=11,
            title="Distribution of Efficiency Across Multiple Realisations",
            titlefontsize=12)
        for i in I
            if i<=4
                plot!(kde(efficiencies[i]),
                label=str_vec[i],
                linewidth=2.5
                )
            else
                plot!(kde(efficiencies[i]),
                label=str_vec[i],
                linewidth=1.5,
                linestyle=:dot
                )
            end
        end
        
        return fig

    elseif output=="table"
        pc_bigger_than(eff1,eff2) = count(F1>F2 for F1 in eff1 for F2 in eff2)/(length(eff1)*length(eff2))
        num_rows = length(efficiencies)-1
        tab = Array{Tuple{String, String, Float64},2}(undef,num_rows,num_rows)
        latextab = raw"\mathbb{P}\left(\mathrm{row exceeds column}\right)"*prod(map(s->raw" & \text{"*s*"}", str_vec[I[2:end]]))*raw"\\ \hline "
        for r in 1:num_rows
            latextab *= raw"\\ \text{" * str_vec[I[r]] * "}" * " & "^(r-1)
            for c in r:num_rows
                p=pc_bigger_than(efficiencies[I[r]],efficiencies[I[c+1]])
                latextab *= @sprintf(" & %.2f",p)
                tab[r,c] = (str_vec[I[r]], str_vec[I[c+1]], p)
            end
        end
        open("./efficiency_table.txt", "w") do f
            write(f, latextab)
        end
        return tab, latextab
    
    elseif output=="theory"
        
        plot(color_palette=:darkrainbow, legend=:none, grid=:none,
            xlabel=L"\eta_1",
            ylabel=L"\eta_2",
            labelfontsize=10,
            title="Continuation Probabilities and Efficiency Landscape",
            titlefontsize=11)
        for i in I
            scatter!(eta_vec[i], series_annotations=text.([" "*str_vec[i]],:left,8))
        end

        grd = 0:0.01:1
        plot!(collect(grd), x->x, color=[:black], linestyle=:dash, label=[])
        plot!(ones(size(grd)),collect(grd), color=[:black], linestyle=:dash, label=[])

        contour!(collect(grd),collect(grd),(x,y)->1/phi((x,y), bm, epsilons),
        aspect_ratio=:equal,
        levels = (1/phi_mf).*[0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.6],
        color=[:grey],
        linestyle=:dot,
        label=[],
        colorbar=:none)
        
    
    else
        return collect(zip((str_vec[I], eta_vec[I], phi_vec[I], efficiencies[I])...))
    end
end

function view_distances(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64, Float64})

    match = [Tuple(p.dist) for p in s if !xor((p.dist .< epsilons)...)]
    fp = [Tuple(p.dist) for p in s if ((p.dist[1] < epsilons[1]) & (p.dist[2] >= epsilons[2])) ]
    fn = [Tuple(p.dist) for p in s if ((p.dist[1] >= epsilons[1]) & (p.dist[2] < epsilons[2])) ]
    # Compare low and high fidelity
    plot(; title="Distance from data: low and high fidelity simulations", titlefontsize=10, 
    aspect_ratio=:equal, grid=:none, legend=(0.3,0.9),
    xlabel=L"\tilde{d}(\tilde{y},\tilde{y}_{obs})", ylabel=L"d(y,y_{obs})", labelfontsize=8)
    scatter!(match, markersize=2, markerstrokewidth=0, label="Matching estimator values")
    scatter!(fp, markersize=3, label="False positive")
    scatter!(fn, markersize=3, label="False negative")
    vline!([epsilons[1]], linestyle=:dash, color=[:black], label="")
    hline!([epsilons[2]], linestyle=:dash, color=[:black], label="")
end
function view_distances(s::Cloud{BenchmarkParticle}, epsilons::Tuple{Float64, Float64}, par_n::Integer, par_name::AbstractString)

    match = [(p.k[par_n], p.dist[2]) for p in s if !xor((p.dist .< epsilons)...)]
    fp = [(p.k[par_n], p.dist[2]) for p in s if ((p.dist[1] < epsilons[1]) & (p.dist[2] >= epsilons[2])) ]
    fn = [(p.k[par_n], p.dist[2]) for p in s if ((p.dist[1] >= epsilons[1]) & (p.dist[2] < epsilons[2])) ]

    # Compare distance by parameter
    plot(; title="Distance from data: by "*par_name, titlefontsize=10, grid=:none, legend=:none,
    xlabel=par_name, ylabel=L"d(y,y_{obs})", labelfontsize=8)
    scatter!(match, markersize=2, markerstrokewidth=0, label="Matching estimator values")
    scatter!(fp, markersize=3, label="False positive")
    scatter!(fn, markersize=3, label="False negative")
    hline!([epsilons[2]], linestyle=:dash, color=[:black], label="")

end

function observed_variances(bm::Cloud{BenchmarkParticle}, sims_set::Array{Cloud{BenchmarkParticle},1}, epsilons::Tuple{Float64,Float64}, F::Array{Function,1})
    method_list = ["abc", "ed", "er", "mf"]
    vartab = Array{Array{Float64,1},2}(undef,length(F),length(method_list))
    phitab = Array{Float64,2}(undef,length(F),length(method_list))
    Random.seed!(1457)
    for i in 1:length(F)
        for j in 1:length(method_list)
            etas, phitab[i,j] = get_eta(bm, epsilons, F[i], method=method_list[j])
            sset = [MFABCCloud(sims, epsilons, etas) for sims in sims_set]
            vartab[i,j] = [estimator_variance(sset, F[i])]
        end
    end
    Random.seed!()
    return vartab, phitab
end