using StatsPlots, LaTeXStrings

export parameterscatter

@userplot ParameterScatter
recipetype(::Val{:parameterscatter}, args...) = ParameterScatter(args...)

function update_axes_guides(d::KW, labs, lims, i, j, n)
    # d[:title]  = (i==1 ? _cycle(labs,j) : "")
    # d[:xticks] = (i==n)
    d[:xguide] = (i==n ? StatsPlots._cycle(labs,j) : "")
    d[:xlims] = StatsPlots._cycle(lims,j)
    # d[:yticks] = (j==1)
    d[:yguide] = (j==1 ? StatsPlots._cycle(labs,i) : "")
    d[:ylims] = (i==j ? :auto : StatsPlots._cycle(lims,i))
end

@recipe function f(ps::ParameterScatter)

    mat = ps.args[1]
    w = ps.args[2]
    logww = ps.args[3]
    (N, n) = size(mat)

    labs = pop!(plotattributes, :label, [""])
    lims = pop!(plotattributes, :xlims, [""])

    link := :x  # need custom linking for y
    layout := @layout (n,n)
    legend := false
    foreground_color_border := nothing
#    margin := 1mm
    titlefont := font(11)
    fillcolor --> Plots.fg_color(plotattributes)
    linecolor --> Plots.fg_color(plotattributes)
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
                    seriestype := :scatter
                    markerstrokewidth --> 0
                    marker_z --> (i>j ? w : logww)
                    seriescolor --> (i>j ? cgrad(:lajolla) : cgrad(:blues, scale=:log))
                    #xformatter --> ((i == n) ? :auto : (x -> ""))
                    #yformatter --> ((j == 1) ? :auto : (y -> ""))
                    colorbar --> :right
                    vj, vi
                end
        end
    end
end

function parameterscatter(t::IndexedTable; columns, kwargs...)
    @df select(t, :θ) parameterscatter(cols(columns), select(t, :weight), sum.(select(t, :logww)); kwargs...)
end