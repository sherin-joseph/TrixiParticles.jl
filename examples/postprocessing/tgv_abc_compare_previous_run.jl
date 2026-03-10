using CSV, DataFrames, Plots

const ROOT = joinpath(@__DIR__, "..", "..", "out", "tgv_abc_batch")
const NEW_RUNS = joinpath(ROOT, "summary_runs.csv")
const OLD_RUNS = joinpath(ROOT, "summary_runs_derived.csv")
const OUT_DIR = joinpath(ROOT, "compare_previous")

mkpath(OUT_DIR)

parse_locale_float(x) = parse(Float64, replace(String(x), "," => "."))

function load_old_runs(path)
    old = CSV.read(path, DataFrame)
    old.t_final = parse_locale_float.(old.tFinal)
    old.ke_old = parse_locale_float.(old.keFinal)
    return select(old, :variant, :t_final, :ke_old)
end

function load_new_runs(path)
    new = CSV.read(path, DataFrame; missingstring="missing")
    return new
end

function build_delta_table(new, old)
    joined = innerjoin(new, old, on=[:variant, :t_final])
    joined.ke_delta = joined.ke_final .- joined.ke_old
    joined.ke_delta_pct = 100 .* joined.ke_delta ./ joined.ke_old
    sort!(joined, [:t_final, :variant])
    return joined
end

function make_c_variant_plot(delta_df)
    cdf = filter(:mode => ==("vms"), delta_df)

    p = plot(xlabel="C_s", ylabel="Final KE",
             title="VMS C-variants: old vs new KE",
             legend=:best, grid=true,
             marker=:circle, linewidth=2,
             size=(1000, 700), dpi=180)

    for t in sort(unique(cdf.t_final)), ms in sort(unique(skipmissing(cdf.min_shepard)))
        sub = filter(r -> r.t_final == t && r.min_shepard == ms, eachrow(cdf))
        sub = sort(collect(sub); by=r -> r.C_s)
        isempty(sub) && continue

        xs = [r.C_s for r in sub]
        y_old = [r.ke_old for r in sub]
        y_new = [r.ke_final for r in sub]
        label_old = "old t=$(t), ms=$(ms)"
        label_new = "new t=$(t), ms=$(ms)"

        plot!(p, xs, y_old; label=label_old, linestyle=:dash)
        plot!(p, xs, y_new; label=label_new, linestyle=:solid)
    end

    savefig(p, joinpath(OUT_DIR, "ke_C_old_vs_new.png"))
end

function main()
    isfile(NEW_RUNS) || error("Missing new summary file: $(NEW_RUNS)")
    isfile(OLD_RUNS) || error("Missing previous summary file: $(OLD_RUNS)")

    new = load_new_runs(NEW_RUNS)
    old = load_old_runs(OLD_RUNS)
    delta = build_delta_table(new, old)

    csv_path = joinpath(OUT_DIR, "summary_runs_delta_vs_previous.csv")
    CSV.write(csv_path, delta)

    make_c_variant_plot(delta)

    println("Wrote: $(csv_path)")
    println("Wrote: $(joinpath(OUT_DIR, "ke_C_old_vs_new.png"))")
end

main()
