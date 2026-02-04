module WindkesselModel

export Windkessel, update_windkessel!, auto_tune_windkessel!

"""
    Windkessel(; R1, R2, C, p0)

RCR Windkessel model. State: `p`, `t_last`, `Q_last`.
"""
mutable struct Windkessel
    R1::Float64
    R2::Float64
    C::Float64
    p::Float64
    t_last::Float64
    Q_last::Float64
    function Windkessel(; R1, R2, C, p0)
        new(R1, R2, C, p0, 0.0, 0.0)
    end
end

"""
    update_windkessel!(w, Qout, t)

Update Windkessel state at time `t` with flow rate `Qout`.
"""
function update_windkessel!(w::Windkessel, Qout::Float64, t::Float64)
    dt = t - w.t_last
    if dt <= 0.0
        return
    end

    α = 1.0 / (w.C * w.R2)
    β = (w.R1 + w.R2) / (w.C * w.R2)
    γ = w.R1 / (w.C * w.R2)

    Q_wk = Qout
    dQdt_n = (Q_wk - w.Q_last) / dt

    dpdt_n = -α * w.p + β * Q_wk + γ * dQdt_n
    p_star = w.p + dt * dpdt_n

    dpdt_star = -α * p_star + β * Q_wk + γ * dQdt_n
    w.p += 0.5 * dt * (dpdt_n + dpdt_star)

    w.t_last = t
    w.Q_last = Q_wk
end

"""
    auto_tune_windkessel!(w, domain_size, ν, Re, T_cycle; ρ, ΔP_target)

Auto-tune the Windkessel parameters based on domain & flow conditions.
Supports both 2D and 3D: pass domain_size as (Lx, Ly) or (Lx, Ly, Lz).
"""
function auto_tune_windkessel!(w::Windkessel,
                                domain_size::Tuple,
                                ν::Float64, Re::Float64,
                                T_cycle::Float64;
                                ρ::Float64 = 1050.0,
                                ΔP_target::Float64 = 1000.0)

    dims = length(domain_size)
    @assert dims == 2 || dims == 3 "domain_size must be a 2-tuple or 3-tuple"

    Lx, Ly = domain_size[1:2]
    H = Ly
    W = dims == 3 ? domain_size[3] : 1.0

    U_ref = Re * ν / H
    @info "Reference mean velocity U_ref = $U_ref m/s"

    Q_ref = U_ref * H * W
    @info "Reference flow rate Q_ref = $Q_ref m³/s"

    Rtot = ΔP_target / Q_ref
    @info "Estimated total downstream resistance Rtot = $Rtot Pa·s/m³"

    w.R1 = 0.1 * Rtot
    w.R2 = 0.9 * Rtot

    τ = T_cycle / 2
    w.C = τ / w.R2

    @info "Tuned Windkessel: R1=$(w.R1), R2=$(w.R2), C=$(w.C)"
end

"""
Pretty-print a Windkessel instance.
"""
function Base.show(io::IO, w::Windkessel)
    println(io, "Windkessel:")
    println(io, "  R1 = $(w.R1) Pa·s/m³")
    println(io, "  R2 = $(w.R2) Pa·s/m³")
    println(io, "  C  = $(w.C) m³/Pa")
    println(io, "  p  = $(w.p) Pa")
end

end # module
