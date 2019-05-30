using DelimitedFiles
using StatsBase, Statistics, StatsPlots
using Random

export estimate_mu, ESS
export compare_efficiencies, view_distances, observed_variances, efficiency_histogram, plot_eta_estimates

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
    Random.seed!(123)
    sample_idx = Iterators.partition(1:length(bm), size_samples)
    efficiencies = zeros(length(sample_idx),length(eta_vec))
    for (i,etas) in enumerate(eta_vec)
        cld = MakeMFABCCloud(bm, epsilons, etas)
        for (j,idx) in enumerate(sample_idx)
            efficiencies[j,i] = efficiency(cld[idx])
        end
    end
    Random.seed!()
    return efficiencies
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
    alpha_vec = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
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
        legendfontsize=11,
        title="Distribution of Efficiency Across Multiple Realisations",
        titlefontsize=12)
        boxplot!(efficiencies[:,I], label=hcat(str_vec[I]...), alpha=hcat(alpha_vec[I]...))
    
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
            title="Continuation Probabilities and Efficiency Landscape",
            titlefontsize=11)
        for i in I
            scatter!(eta_vec[i], annotations=((eta_vec[i].+offset_vec[i])..., text(str_vec[i],f((8,position_vec[i])))))
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
        linewidth=1,
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
    plot(; title="Distance from data: low and high fidelity simulations", titlefontsize=12, 
    aspect_ratio=:equal, grid=:none,
    xlabel=latexstring("\$ \\tilde{d}(\\tilde{y}, \\tilde{y}_{obs}) \$"), ylabel=latexstring("\$ d(y,y_{obs}) \$"), labelfontsize=10)
    scatter!(match, markersize=1, markerstrokewidth=0, label="Matching estimator values")
    scatter!(fp, markersize=2, label="False positive")
    scatter!(fn, markersize=2, label="False negative")
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
    scatter!(match, markersize=1, markerstrokewidth=0, label="Matching estimator values")
    scatter!(fp, markersize=2, label="False positive")
    scatter!(fn, markersize=2, label="False negative")
    hline!([epsilons[2]], linestyle=:dash, color=[:black], label="")

end

function observed_variances(bm::BenchmarkCloud, size_samples::Int64, epsilons::Tuple{Float64,Float64}, F::Array{Function,1}, budgets::Array{Float64,1}=[Inf64])
    method_list = ["abc", "er", "ed", "mf"]
    vartab = Array{Array{Float64,1},2}(undef,length(F),length(method_list)+1)
    phitab = Array{Float64,2}(undef,length(F),length(method_list)+1)
    etatab = Array{Tuple{Float64,Float64},2}(undef,length(F),length(method_list)+1)

    ess_eta = get_eta(bm, epsilons, method="mf")[1]
    sample_idx = Iterators.partition(1:length(bm), size_samples)

    function budget_cloud(cloud::MFABCCloud, budget::Float64)::Int64
        c=0
        for (n,pp) in enumerate(cloud)
            c += sum(pp.p.cost)
            if c>budget
                return n-1
            end
        end
        return length(cloud)
    end

    Random.seed!(111)
    mu = zeros(length(sample_idx),length(budgets))
    for (i,fun) in enumerate(F)
        for (j,mth) in enumerate(method_list)
            vartab[i,j] = zeros(length(budgets))
            etas, phitab[i,j] = get_eta(bm, epsilons, method=mth, F=fun)
            for (k,idx) in enumerate(sample_idx)
                cld = MakeMFABCCloud(bm[idx], epsilons, etas)
                for (l,b) in enumerate(budgets)
                    n = budget_cloud(cld, b)
                    mu[k,l] = estimate_mu(cld[1:n], fun)
                end
            end
            vartab[i,j] = vec(var(mu, dims=1))
            etatab[i,j] = etas
        end

        # Additional method: repeat procedure for the eta used to maximise ESS (i.e. independently of F)
        phitab[i,end] = phi(ess_eta, bm, epsilons, F=fun)
        etatab[i,end] = ess_eta
        vartab[i,end] = zeros(length(budgets))
        for (k,idx) in enumerate(sample_idx)
            cld = MakeMFABCCloud(bm[idx], epsilons, ess_eta)
            for (l,b) in enumerate(budgets)
                n = budget_cloud(cld, b)
                mu[k,l] = estimate_mu(cld[1:n], fun)
            end
            vartab[i,end] = vec(var(mu,dims=1))
        end
    end
    Random.seed!()
    return vartab, phitab, etatab
end

function efficiency_histogram(bm::BenchmarkCloud, size_samples::Int64, epsilons::Tuple{Float64, Float64}; method::String="mf")
    etas = get_eta(bm, epsilons, method=method)[1]
    efficiencies = vec(get_efficiencies(bm, epsilons, size_samples, [etas]))

    title_dictionary = Dict("mf"=>"Early Accept/Reject", "ed"=>"Early Decision", "er"=>"Early Rejection")
    fig = plot(xlabel="Effective samples per second",
    xtickfontsize=10,
    yticks=[],
    xgrid=false,
    legend=:none,
    title=title_dictionary[method],
    titlefontsize=12)
    histogram!(efficiencies, bins=20)
    vline!([mean(efficiencies)], color=[:red], label="", linewidth=3.0)
end

function plot_eta_estimates(cloud_set::Array{<:Cloud,1}, bm::BenchmarkCloud, epsilons::Tuple{Float64, Float64}; method::String="mf", kwargs...)
    eta_real, phi_mf = get_eta(bm, epsilons, method=method)
    eta_estimates = [get_eta(cld, epsilons; method=method, kwargs...)[1] for cld in cloud_set]
    scatter(eta_estimates, xlim=(0,1), ylim=(0,1),label="Estimates")
    scatter!(eta_real, label="Benchmark")
    
    grd = 0:0.01:1
    sp = sample_properties(bm, epsilons)
    phi_plot((x,y)) = phi((x,y), sp...)

    contour!(collect(grd),collect(grd),(x,y)->1/phi_plot((x,y)),
    aspect_ratio=:equal,
    levels = (1/phi_mf).*[0.6, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
    c=cgrad(:grays_r),
    linewidth=1,
    linestyle=:dot,
    label=[],
    colorbar=:none)
end

function plot_apost_efficiencies(inc_set::Array{<:Cloud,1}, bm_set::Array{<:Cloud,1})
    eff(set) = ESS.(set)./cost.(set)
    boxplot([eff(bm_set), eff(inc_set)], label=["Benchmark" "Adaptive"], xticks=[], ylabel="ESS per second", 
        title="Distribution of efficiencies across multiple realisations")
end