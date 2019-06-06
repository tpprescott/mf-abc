using DelimitedFiles
using StatsBase, Statistics, StatsPlots
using Random

export estimate_mu, ESS
export compare_efficiencies, view_distances, variance_table, plot_eta_estimates, plot_apost_efficiencies

######### Helper functions

function ESS(weights::Array{Float64,1})
    return sum(weights)^2/sum(weights.^2)
end
function ESS(s::MFABCCloud)
    return ESS([p.w for p in s])
end

function total_cost(s::MFABCCloud)
    return sum([sum(pp.p.cost) for pp in s])
end
function total_cost(s::BenchmarkCloud)
    return sum([sum(p.cost) for p in s])
end

function efficiency(s::MFABCCloud)
    return ESS(s)/total_cost(s)
end

function estimate_mu(s::MFABCCloud, parameterFun::Function)
    F_i = map(parameterFun, [pp.p.k for pp in s])
    w_i = [pp.w for pp in s]
    return sum(w_i .* F_i)/sum(w_i)
end

####################################

using Plots, StatsPlots, KernelDensity, LaTeXStrings, Printf
f(i) = Plots.font("serif",i...)
pyplot(titlefont=f(12), guidefont=f(9), legendfont=f(9), xtickfont=f(8), ytickfont=f(8))


function get_efficiencies(bm::BenchmarkCloud, epsilons::Tuple{Float64, Float64}, size_samples::Int64, eta_vec::Array{Tuple{Float64,Float64},1})
    sample_idx = Iterators.partition(1:length(bm), size_samples)
    Random.seed!(123)
    cloud_set = [MakeMFABCCloud(bm[i], epsilons, etas) for (i,etas) in Iterators.product(sample_idx, eta_vec)]
    Random.seed!()
    return ESS.(cloud_set)./cost.(cloud_set)
end

function compare_efficiencies(bm::BenchmarkCloud, size_samples::Int64, epsilons::Tuple{Float64, Float64};
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
    offset_vec = [(0.03,0.01),(-0.02,0.01),(0.02,0.03),(-0.02,0.00),(0.02,0.02),(0.02,0.02),(0.02,0.02),(0.02,0.02)]
    alpha_vec = [1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 0.3]
    position_vec = [:left,:right,:left,:right,:left,:left,:left,:left]
    I = sortperm(phi_vec)

    if output=="plot"
        efficiencies = get_efficiencies(bm, epsilons, size_samples, eta_vec)
        #=         
        fig = plot(color_palette=:darkrainbow,
            thickness_scaling=1,
            xlabel="Effective samples per second",
            xtickfontsize=10,
            yticks=[],
            xgrid=false,
            #legend=(0.7,0.25),
            legendfontsize=11,
            title="Distribution of Efficiency Across Multiple Realisations",
            titlefontsize=12)
        for i in I
            if i<=4
                plot!(kde(efficiencies[:,i]),
                label=str_vec[i],
                linewidth=2.5
                )
            else
                plot!(kde(efficiencies[:,i]),
                label=str_vec[i],
                linewidth=1.5,
                linestyle=:dot
                )
            end
        end
        =#
        fig = plot(color_palette=:darkrainbow,
        thickness_scaling=1,
        ylabel="Effective samples per second",
        ytickfontsize=10,
        xticks=[],
        ygrid=false,
        #legend=(0.7,0.25),
        legendfontsize=10,
        title="Observed Distribution of Efficiency",
        titlefontsize=12)
        violin!(efficiencies[:,I], label=hcat(str_vec[I]...), alpha=hcat(alpha_vec[I]...))
    
        pc_bigger_than(eff1,eff2) = count(F1>F2 for F1 in eff1 for F2 in eff2)/(length(eff1)*length(eff2))
        num_rows = size(efficiencies,2)-1
        tab = Array{Tuple{String, String, Float64},2}(undef,num_rows,num_rows)
        latextab = raw"\mathbb{P}\left(\mathrm{row exceeds column}\right)"*prod(map(s->raw" & \text{"*s*"}", str_vec[I[2:end]]))*raw"\\ \hline "
        for r in 1:num_rows
            latextab *= raw"\\ \text{" * str_vec[I[r]] * "}" * " & "^(r-1)
            for c in r:num_rows
                p=pc_bigger_than(efficiencies[:,I[r]],efficiencies[:,I[c+1]])
                latextab *= @sprintf(" & %.2f",p)
                tab[r,c] = (str_vec[I[r]], str_vec[I[c+1]], p)
            end
        end
        open("./efficiency_table.txt", "w") do f
            write(f, latextab)
        end
        return fig, tab, latextab
    
    elseif output=="theory"
        
        fig = plot(color_palette=:darkrainbow, legend=:none, grid=:none,
            xlabel=L"\eta_1",
            ylabel=L"\eta_2",
            labelfontsize=10,
            title="Continuation Probabilities and Efficiency",
            titlefontsize=11)
        for i in I
            scatter!(eta_vec[i], markersize=8, markerstrokewidth=0, alpha=alpha_vec[i], annotations=((eta_vec[i].+offset_vec[i])..., text(str_vec[i],f((8,position_vec[i])))))
        end        
        
        grd = 0:0.01:1
        plot!(collect(grd), x->x, color=[:black], linestyle=:dash, label=[])
        plot!(ones(size(grd)),collect(grd), color=[:black], linestyle=:dash, label=[])

        sp = sample_properties(bm, epsilons)
        phi_plot((x,y)) = phi((x,y), sp...)

        contour!(collect(grd),collect(grd),(x,y)->1/phi_plot((x,y)),
        aspect_ratio=:equal,
        levels = (1/phi_mf).*[0.6, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
        c=cgrad(:grays_r),
        linewidth=0.6,
        linestyle=:dot,
        label=[],
        colorbar=:none)

        return fig

    else
        return collect(zip((str_vec[I], eta_vec[I], phi_vec[I], efficiencies[:,I])...))
    end
end

function view_distances(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64})

    match = [p.dist for p in s if !xor((p.dist .< epsilons)...)]
    fp = [p.dist for p in s if ((p.dist[1] < epsilons[1]) & (p.dist[2] >= epsilons[2])) ]
    fn = [p.dist for p in s if ((p.dist[1] >= epsilons[1]) & (p.dist[2] < epsilons[2])) ]
    # Compare low and high fidelity
    plot(; title="Distance from data: multifidelity", titlefontsize=12, 
    aspect_ratio=:equal, grid=:none,
    xlabel=latexstring("\$ \\tilde{d}(\\tilde{y}, \\tilde{y}_{obs}) \$"), ylabel=latexstring("\$ d(y,y_{obs}) \$"), labelfontsize=10)
    scatter!(match, markersize=1.5, markerstrokewidth=0, alpha=0.6, label="Matching estimator values")
    scatter!(fp, markersize=3, markerstrokewidth=0, label="False positive")
    scatter!(fn, markersize=3, markerstrokewidth=0, label="False negative")
    vline!([epsilons[1]], linestyle=:dash, color=[:black], label="")
    hline!([epsilons[2]], linestyle=:dash, color=[:black], label="")
end
function view_distances(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}, par_n::Integer, par_name::AbstractString)

    match = [(p.k[par_n], p.dist[2]) for p in s if !xor((p.dist .< epsilons)...)]
    fp = [(p.k[par_n], p.dist[2]) for p in s if ((p.dist[1] < epsilons[1]) & (p.dist[2] >= epsilons[2])) ]
    fn = [(p.k[par_n], p.dist[2]) for p in s if ((p.dist[1] >= epsilons[1]) & (p.dist[2] < epsilons[2])) ]

    # Compare distance by parameter
    plot(; title="Distance from data: by parameter", titlefontsize=12, grid=:none, legend=:none,
    xlabel=par_name, ylabel=L"d(y,y_{obs})", labelfontsize=10)
    scatter!(match, markersize=1.5, markerstrokewidth=0, alpha=0.6, label="Matching estimator values")
    scatter!(fp, markersize=3, markerstrokewidth=0, label="False positive")
    scatter!(fn, markersize=3, markerstrokewidth=0, label="False negative")
    hline!([epsilons[2]], linestyle=:dash, color=[:black], label="")

end
function view_distances(s::BenchmarkCloud, epsilons::Tuple{Float64, Float64}, inset_limits::Tuple{Float64,Float64})

    match = [p.dist for p in s if !xor((p.dist .< epsilons)...)]
    fp = [p.dist for p in s if ((p.dist[1] < epsilons[1]) & (p.dist[2] >= epsilons[2])) ]
    fn = [p.dist for p in s if ((p.dist[1] >= epsilons[1]) & (p.dist[2] < epsilons[2])) ]
    # Compare low and high fidelity
    plot(; title="Distance from data: multifidelity", titlefontsize=12, 
    aspect_ratio=:equal, grid=:none,
    xlabel=latexstring("\$ \\tilde{d}(\\tilde{y}, \\tilde{y}_{obs}) \$"), ylabel=latexstring("\$ d(y,y_{obs}) \$"), labelfontsize=10)
    
    plot!(; inset_subplots=(1,bbox(0.15,0.15,0.3,0.3,:bottom,:right)))

    scatter!(match, markersize=1.5, markerstrokewidth=0, alpha=0.6, label="Matching estimator values", subplot=1)
    scatter!(fp, markersize=3, markerstrokewidth=0, label="False positive", subplot=1)
    scatter!(fn, markersize=3, markerstrokewidth=0, label="False negative", subplot=1)

    scatter!(match, markersize=2.5, markerstrokewidth=0, alpha=0.6, label="Matching estimator values", subplot=2)
    scatter!(fp, markersize=4, markerstrokewidth=0, label="False positive", subplot=2)
    scatter!(fn, markersize=4, markerstrokewidth=0, label="False negative", subplot=2)

    vline!([epsilons[1]], linestyle=:dash, color=[:black], label="", subplot=1)
    hline!([epsilons[2]], linestyle=:dash, color=[:black], label="", subplot=1)

    vline!([epsilons[1]], linestyle=:dash, color=[:black], label="", subplot=2)
    hline!([epsilons[2]], linestyle=:dash, color=[:black], label="", subplot=2)

    plot!(xlim=(0,inset_limits[1]), ylim=(0,inset_limits[2]), legend=:none, aspect_ratio=:equal, grid=:none, subplot=2, xticks=[epsilons[1]], yticks=[epsilons[2]], framestyle=:box)
end


function get_eta(bm::BenchmarkCloud, epsilons::Tuple{Float64,Float64}, F::Array{Function,1})
    method_list = ["abc", "er", "ed", "mf"]
 #   sample_idx = Iterators.partition(1:length(bm), size_samples)
 #   bm_set = [bm[i] for i in sample_idx]

    eta_phi_fun((f,m)) = get_eta(bm, epsilons, method=m, F=f)
    eta_phi = map(eta_phi_fun, Iterators.product(F, method_list))
    ess_eta = get_eta(bm, epsilons, method="mf")[1]
    ess_phi = [phi(ess_eta, bm, epsilons, F=f) for f in F]

    eta_tab = hcat([x[1] for x in eta_phi], repeat([ess_eta], length(F)))
    phi_tab = hcat([x[2] for x in eta_phi], ess_phi)

    return eta_tab, phi_tab
end

function observed_variance(bm_set::Array{BenchmarkCloud,1}, epsilons::Tuple{Float64,Float64}, eta::Tuple{Float64,Float64}, budget::Float64, F::Function)
    mf_set = [MakeMFABCCloud(bm_i, epsilons, eta, budget) for bm_i in bm_set]
    return var(estimate_mu.(mf_set, F))
end

function variance_table(bm::BenchmarkCloud, sample_size::Integer, epsilons::Tuple{Float64, Float64}, eta_tab::Array{Tuple{Float64, Float64}}, Functions::Array{Function,1}, budget::Float64)
    idx = Iterators.partition(1:length(bm), sample_size)
    f_tab = repeat(Functions,1,size(eta_tab,2))
    if size(f_tab)==size(eta_tab)
        shuffle!(bm)
        bm_set = [bm[i] for i in idx]
        return [observed_variance(bm_set, epsilons, eta, budget, f) for (eta, f) in zip(eta_tab, f_tab)]
    else
        error("eta table doesn't match number of functions")
    end
end
function variance_table(bm::BenchmarkCloud, sample_size::Integer, epsilons::Tuple{Float64, Float64}, Functions::Array{Function,1}, budget::Float64)
    eta_tab = get_eta(bm, epsilons, Functions)[1]
    return variance_table(bm, sample_size, epsilons, eta_tab, Functions, budget)
end

function plot_eta_estimates(cloud_set::Array{<:Cloud,1}, bm::BenchmarkCloud, epsilons::Tuple{Float64, Float64}; method::String="mf", kwargs...)
    eta_real, phi_mf = get_eta(bm, epsilons, method=method)
    eta_estimates = [get_eta(cld, epsilons; method=method, kwargs...)[1] for cld in cloud_set]
    
    fig = plot(legend=:none, grid=:none,
    xlabel=L"\eta_1",
    ylabel=L"\eta_2",
    labelfontsize=10,
    title="Continuation Probabilities and Efficiency",
    titlefontsize=11)
    
    scatter!(eta_estimates, markersize=6, markerstrokewidth=0, xlim=(0,1), ylim=(0,1),label="Estimates")
    scatter!(eta_real, markersize=6, markerstrokewidth=0, label="Benchmark")
    
    grd = 0:0.01:1
    sp = sample_properties(bm, epsilons)
    phi_plot((x,y)) = phi((x,y), sp...)

    contour!(collect(grd),collect(grd),(x,y)->1/phi_plot((x,y)),
    aspect_ratio=:equal,
    levels = (1/phi_mf).*[0.6, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
    c=cgrad(:grays_r),
    linewidth=0.6,
    linestyle=:dot,
    label=[],
    colorbar=:none)
end

function plot_apost_efficiencies(labels::NTuple{N, AbstractString}, cld_sets::Vararg{Array{<:Cloud,1},N}) where N
    
    eff(set) = ESS.(set)./cost.(set)

    fig = plot(color_palette=:darkrainbow,
    thickness_scaling=1,
    ylabel="Effective samples per second",
    ytickfontsize=10,
    xticks=[],
    ygrid=false,
    #legend=(0.7,0.25),
    legendfontsize=11,
    title="Observed Distribution of Efficiency",
    titlefontsize=12)
    
    for (c, l) in zip(cld_sets, labels)
        violin!(eff(c), label=l)
    end
    return fig
end
