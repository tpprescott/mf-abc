using DelimitedFiles
using StatsBase, Statistics
using Random

######### Helper functions

function ESS(kweights::Array{Float64,1})
    return sum(kweights)^2/sum(kweights.^2)
end
function ESS_vs_times(kweights::Array{Float64,1}, c_used::Array{Float64,1}, ctilde::Array{Float64,1})
    return ESS(kweights), sum(ctilde)+sum(c_used)
end
function ESS_vs_times(s::Sample, epsilons::Tuple{Float64, Float64}, etas::Tuple{Float64, Float64}=(1.0,1.0))
    kweights, c_used = posthoc(s, epsilons, etas)
    return ESS_vs_times(kweights, c_used, s.ctilde)
end
function efficiency(a,b,c)
    ESS_vs_times(a,b,c)[1]/ESS_vs_times(a,b,c)[2]
end

function estimate_mu(parameterFun::Function, kweights::Array{Float64,1}, ksample::Array{Float64,2})
    F = [parameterFun(ksample[j,:]) for j in 1:size(ksample,1)]
    return sum(kweights.*F)/sum(kweights)
end
function estimate_mu(parameterFun::Function, s::Sample, epsilons::Tuple{Float64,Float64}, etas::Tuple{Float64,Float64}=(1.0,1.0);
    budget::Number=Inf64)

    kweights, c_used = posthoc(s, epsilons, etas)
    used_idx = cumsum(s.ctilde .+ c_used).<budget
    return estimate_mu(parameterFun, kweights[used_idx], s.ksample[used_idx,:])
end

function estimator_variance(parameterFun::Function, sset::Array{Sample,1}, epsilons::Tuple{Float64,Float64}, etas::Tuple{Float64,Float64}=(1.0,1.0);
    budget::Number=Inf64)

    return var(map(s->estimate_mu(parameterFun, s, epsilons, etas, budget=budget), sset))
end

####################################

using Plots, StatsPlots, KernelDensity, LaTeXStrings, Printf
plotlyjs()


# Get eta_1, eta_2 (continuation probabilities) optimized for specified functions
F = [k->Float64(1.9 < k[2] < 2.1),
     k->Float64(1.2 < k[2] < 1.4),
     k->k[2]]

#=eta_mf_Fi, phi_mf_Fi = zip(map(Fi->get_eta(bm, epsilons, Fi, method="mf"), F)...)
eta_er_Fi, phi_er_Fi = zip(map(Fi->get_eta(bm, epsilons, Fi, method="er"), F)...)
eta_ed_Fi, phi_ed_Fi = zip(map(Fi->get_eta(bm, epsilons, Fi, method="ed"), F)...) =#

function compare_efficiencies(bm::Sample, sset::Array{Sample,1}, epsilons::Tuple{Float64, Float64};
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
    efficiencies = [map(s->efficiency(s,epsilons,etas), sset) for etas in eta_vec]
    
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

function view_distances(s::Sample, epsilons::Tuple{Float64, Float64})

    Ot = (s.dtilde .< epsilons[1])
    O = (s.d .< epsilons[2])

    # Compare low and high fidelity
    plot(; title="Distance from data: low and high fidelity simulations", titlefontsize=10, 
    aspect_ratio=:equal, grid=:none, legend=(0.3,0.9),
    xlabel=L"\tilde{d}(\tilde{y},\tilde{y}_{obs})", ylabel=L"d(y,y_{obs})", labelfontsize=8)
    scatter!(s.dtilde[Ot .== O], s.d[Ot .== O], markersize=2, markerstrokewidth=0, label="Matching estimator values")
    scatter!(s.dtilde[Ot .& .~O], s.d[Ot .& .~O], markersize=3, label="False positive")
    scatter!(s.dtilde[.~Ot .& O], s.d[.~Ot .& O], markersize=3, label="False negative")
    vline!([epsilons[1]], linestyle=:dash, color=[:black], label="")
    hline!([epsilons[2]], linestyle=:dash, color=[:black], label="")
end
function view_distances(s::Sample, epsilons::Tuple{Float64, Float64}, par_n::Integer, par_name::AbstractString)

    Ot = (s.dtilde .< epsilons[1])
    O = (s.d .< epsilons[2])

    # Compare distance by parameter
    plot(; title="Distance from data: by "*par_name, titlefontsize=10, grid=:none, legend=:none,
    xlabel=par_name, ylabel=L"d(y,y_{obs})", labelfontsize=8)
    scatter!(s.ksample[Ot .== O, par_n], s.d[Ot .== O], markersize=2, markerstrokewidth=0, label="Matching estimator values")
    scatter!(s.ksample[Ot .& .~O, par_n], s.d[Ot .& .~O], markersize=3, label="False positive")
    scatter!(s.ksample[.~Ot .& O, par_n], s.d[.~Ot .& O], markersize=3, label="False negative")
    hline!([epsilons[2]], linestyle=:dash, color=[:black], label="")

end

function observed_variances(bm::Sample, sset::Array{Sample,1}, epsilons::Tuple{Float64,Float64}, F::Array{Function,1}, budgets::Array{Float64,1}=[Inf64])
    method_list = ["abc", "ed", "er", "mf"]
    vartab = Array{Array{Float64,1},2}(undef,length(F),length(method_list))
    phitab = Array{Float64,2}(undef,length(F),length(method_list))
    for i in 1:length(F)
        for j in 1:length(method_list)
            eta, phitab[i,j] = get_eta(bm, epsilons, F[i], method=method_list[j])
            vartab[i,j] = [estimator_variance(F[i], sset, epsilons, eta, budget=b) for b in budgets]
        end
    end
    return vartab, phitab
end