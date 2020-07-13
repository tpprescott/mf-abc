using StatsPlots, LaTeXStrings
include("maxogram.jl")

############################################
export parameterweights

@userplot ParameterWeights
recipetype(::Val{:parameterweights}, args...) = ParameterWeights(args...)

function update_axes_guides(d::KW, labs, lims, i, j, n)
    # d[:title]  = (i==1 ? _cycle(labs,j) : "")
    # d[:xticks] = (i==n)
    d[:xguide] = (i==n ? StatsPlots._cycle(labs,j) : "")
    d[:xlims] = (i>j ? :auto : StatsPlots._cycle(lims,j))
    # d[:yticks] = (j==1)
    d[:yguide] = (j==1 ? StatsPlots._cycle(labs,i) : "")
    d[:ylims] = (i>=j ? :auto : StatsPlots._cycle(lims,i))
end

@recipe function f(ps::ParameterWeights)

    mat = ps.args[1]
    w = ps.args[2]
    (N, n) = size(mat)

    labs = pop!(plotattributes, :label, [""])
    lims = pop!(plotattributes, :xlims, [""])

    link := :x  # need custom linking for y
    layout := @layout (n,n)
    legend := false
    foreground_color_border := nothing
#    margin := 1mm
    titlefont := font(11)
    # fillcolor --> Plots.fg_color(plotattributes)
    # linecolor --> Plots.fg_color(plotattributes)
    markeralpha := 0.4
    #grad = cgrad(:lajolla)
    indices = reshape(1:n^2, n, n)'
    title = get(plotattributes, :title, "")
    title_location = get(plotattributes, :title_location, :center)
    title := ""

    # histograms on the diagonal
    for i=1:n
        @series begin
            if title != "" && title_location == :left && i == 1
                title := title
            end
            seriestype := :histogram
            linewidth := 0
            subplot := indices[i,i]
            weights := w
            grid := false
            #xformatter --> ((i == n) ? :auto : (x -> ""))
            yformatter := (y -> "")
            update_axes_guides(plotattributes, labs, lims, i, i, n)
            view(mat,:,i)
        end
    end

    # scatters
    for i=1:n
        ylink := setdiff(vec(indices[i,:]), indices[i,i])
        vi = view(mat,:,i)
        for j = 1:n
            j==i && continue
            vj = view(mat,:,j)
            update_axes_guides(plotattributes, labs, lims, i, j, n)
                @series begin
                    subplot := indices[i,j]
                    seriestype := :histogram2d
                    weights := w
                    # markerstrokewidth --> 0
                    # marker_z --> (i>j ? w : logww)
                    seriescolor --> cgrad(:lajolla)
                    # xformatter --> ((i == n) ? :auto : (x -> ""))
                    # yformatter --> ((j == 1) ? :auto : (y -> ""))
                    colorbar --> :none
                    vj, vi
                end
        end
    end
end

function parameterweights(t::IndexedTable; columns, kwargs...)
    @df select(t, :θ) parameterweights(cols(columns), select(t, :weight); kwargs...)
end

############################################
export parameterloglh

@userplot ParameterLogLH
recipetype(::Val{:parameterloglh}, args...) = ParameterLogLH(args...)

@recipe function f(ps::ParameterLogLH)

    mat = ps.args[1]
    loglh = ps.args[2]
    (N, n) = size(mat)

    labs = pop!(plotattributes, :label, [""])
    lims = pop!(plotattributes, :xlims, [""])

    link := :x  # need custom linking for y
    layout := @layout (n,n)
    legend := false
    foreground_color_border := nothing
#    margin := 1mm
    titlefont := font(11)
    # fillcolor --> Plots.fg_color(plotattributes)
    # linecolor --> Plots.fg_color(plotattributes)
    markeralpha := 0.4
    #grad = cgrad(:lajolla)
    indices = reshape(1:n^2, n, n)'
    title = get(plotattributes, :title, "")
    title_location = get(plotattributes, :title_location, :center)
    title := ""

    # histograms on the diagonal
    for i=1:n
        @series begin
            if title != "" && title_location == :left && i == 1
                title := title
            end
            seriestype := :maxogram
            linewidth := 0
            subplot := indices[i,i]
            weights := loglh
            grid := false
            #xformatter --> ((i == n) ? :auto : (x -> ""))
            yformatter := (y -> "")
            update_axes_guides(plotattributes, labs, lims, i, i, n)
            view(mat,:,i)
        end
    end

    # scatters
    for i=1:n
        ylink := setdiff(vec(indices[i,:]), indices[i,i])
        vi = view(mat,:,i)
        for j = 1:n
            j==i && continue
            vj = view(mat,:,j)
            update_axes_guides(plotattributes, labs, lims, i, j, n)
                @series begin
                    subplot := indices[i,j]
                    seriestype := :maxogram2d
                    weights := loglh
                    # markerstrokewidth --> 0
                    # marker_z --> (i>j ? w : logww)
                    seriescolor --> cgrad(:blues)
                    # xformatter --> ((i == n) ? :auto : (x -> ""))
                    # yformatter --> ((j == 1) ? :auto : (y -> ""))
                    colorbar --> :none
                    vj, vi
                end
        end
    end
end

function parameterloglh(t::IndexedTable; columns, kwargs...)
    logw = sum.(select(t, :logww))
    logw .-= maximum(logw)
    broadcast!(exp, logw, logw)
    @df select(t, :θ) parameterloglh(cols(columns), logw; kwargs...)
end