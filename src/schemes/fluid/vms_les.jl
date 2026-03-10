@doc raw"""
    VMSLES(; C_s=0.12, epsilon=0.01, strain_mode=:fine, min_shepard=0.7)

Variational-multiscale (VMS) inspired LES model for WCSPH.

This implements the workflow discussed in Hughes et al. (1998, 2000) in a particle setting:

1. **Projection (coarse scale):** ``\bar u = P[u]`` using Shepard-normalized kernel averaging.
2. **Resolved residual:** ``R(\bar U) = P[\dot u^{base}] - F(\bar U)`` where ``F`` is the WCSPH
   momentum operator evaluated on coarse fields.
3. **Fine scales:** ``u' \approx -\tau\,R(\bar U)`` with ``\tau \sim h/(c_0+\|\bar u\|)``.
4. **LES dissipation on fine scales:** add an eddy-viscosity term that acts on ``u'`` only.

Designed to be:
- **PeriodicBox compatible** via `foreach_point_neighbor`.
- **GPU compatible** by allocating cache arrays with `KernelAbstractions.allocate` (via `allocate`)
  and using `@threaded semi` and `foreach_point_neighbor`.

Keywords
- `C_s`: Smagorinsky constant used for the eddy viscosity acting on fine scales.
- `epsilon`: Regularization parameter used in strain surrogate and Laplacian denominator.
- `strain_mode`: `:fine` uses ``u'`` for the strain surrogate; `:coarse` uses ``\bar u``.
- `min_shepard`: If the Shepard denominator is below this value, the model is disabled locally
  (prevents boundary/truncation artifacts when defining coarse/fine scales).
"""
struct VMSLES{T}
    C_s::T
    epsilon::T
    strain_mode::Symbol
    min_shepard::T

    function VMSLES(; C_s=0.12, epsilon=0.01, strain_mode=:fine, min_shepard=0.7)
        return new{typeof(C_s)}(C_s, epsilon, strain_mode, min_shepard)
    end
end

"""Allocate cache arrays needed by `VMSLES`.

Returns an empty named tuple if `vms_les === nothing`.
"""
function create_cache_vms_les(vms_les, initial_condition, NDIMS, ELTYPE, n_particles)
    if vms_les === nothing
        return (;)
    end

    backend = KernelAbstractions.get_backend(initial_condition.mass)

    ubar = allocate(backend, ELTYPE, (NDIMS, n_particles))
    rhobar = allocate(backend, ELTYPE, (n_particles,))
    pbar = allocate(backend, ELTYPE, (n_particles,))
    dvbar_pred = allocate(backend, ELTYPE, (NDIMS, n_particles))
    dv_resolved = allocate(backend, ELTYPE, (NDIMS, n_particles))
    residual = allocate(backend, ELTYPE, (NDIMS, n_particles))
    uprime = allocate(backend, ELTYPE, (NDIMS, n_particles))
    shepard_den = allocate(backend, ELTYPE, (n_particles,))

    return (; vms_les_model=vms_les,
            vms_les_ubar=ubar,
            vms_les_rhobar=rhobar,
            vms_les_pbar=pbar,
            vms_les_dvbar_pred=dvbar_pred,
            vms_les_dv_resolved=dv_resolved,
            vms_les_residual=residual,
            vms_les_uprime=uprime,
            vms_les_shepard_den=shepard_den)
end

"""Entry point called from `kick!`.

Adds VMS-LES model acceleration to `dv_ode` for WCSPH systems that have
`cache.vms_les_model`.
"""
function apply_vms_les!(dv_ode, v_ode, u_ode, semi, t)
    foreach_system(semi) do system
        system isa WeaklyCompressibleSPHSystem || return
        hasproperty(system.cache, :vms_les_model) || return
        model = system.cache.vms_les_model
        model === nothing && return

        dv = wrap_v(dv_ode, system, semi)
        v  = wrap_v(v_ode, system, semi)
        u  = wrap_u(u_ode, system, semi)

        apply_vms_les_system!(dv, v, u, system, semi)
    end

    return dv_ode
end

function apply_vms_les_system!(dv, v, u, system, semi)
    model = system.cache.vms_les_model::VMSLES
    NDIMS = ndims(system)

    # Cache arrays
    ubar        = system.cache.vms_les_ubar
    rhobar      = system.cache.vms_les_rhobar
    pbar        = system.cache.vms_les_pbar
    dvbar_pred  = system.cache.vms_les_dvbar_pred
    dv_resolved = system.cache.vms_les_dv_resolved
    residual    = system.cache.vms_les_residual
    uprime      = system.cache.vms_les_uprime
    den         = system.cache.vms_les_shepard_den

    # Reset caches (GPU-safe)
    set_zero!(ubar)
    set_zero!(rhobar)
    set_zero!(pbar)
    set_zero!(dvbar_pred)
    set_zero!(dv_resolved)
    set_zero!(residual)
    set_zero!(uprime)
    set_zero!(den)

    sound_speed = system_sound_speed(system)
    system_coords = current_coordinates(u, system)

    # ------------------------------------------------------------
    # (1) Projection: ubar = P[u], rhobar = P[rho], dvbar_pred = P[dv_base]
    #     using Shepard-normalized kernel averaging.
    # ------------------------------------------------------------
    dv_vel = current_velocity(dv, system)  # (NDIMS, N) for ContinuityDensity
    v_vel  = current_velocity(v, system)
    v_rho  = current_density(v, system)

    foreach_point_neighbor(system, system,
                           system_coords, system_coords, semi;
                           points=each_integrated_particle(system)) do particle,
                                                                        neighbor,
                                                                        pos_diff,
                                                                        distance
        W = smoothing_kernel(system, distance, particle)

        rho_b = @inbounds v_rho[neighbor]
        m_b   = @inbounds hydrodynamic_mass(system, neighbor)
        Vb    = m_b / rho_b

        w = Vb * W
        @inbounds den[particle] += w

        u_b = @inbounds extract_svector(v_vel, system, neighbor)
        for i in 1:NDIMS
            @inbounds ubar[i, particle] += w * u_b[i]
        end

        @inbounds rhobar[particle] += w * rho_b

        dv_b = @inbounds extract_svector(dv_vel, system, neighbor)
        for i in 1:NDIMS
            @inbounds dvbar_pred[i, particle] += w * dv_b[i]
        end
    end

    @threaded semi for particle in each_integrated_particle(system)
        d = @inbounds den[particle]
        if d > eps(eltype(d))
            for i in 1:NDIMS
                @inbounds ubar[i, particle] /= d
                @inbounds dvbar_pred[i, particle] /= d
            end
            @inbounds rhobar[particle] /= d
        else
            u_p = @inbounds extract_svector(v_vel, system, particle)
            for i in 1:NDIMS
                @inbounds ubar[i, particle] = u_p[i]
                @inbounds dvbar_pred[i, particle] = zero(eltype(ubar))
            end
            @inbounds rhobar[particle] = @inbounds v_rho[particle]
            @inbounds den[particle] = zero(eltype(d))
        end

        @inbounds pbar[particle] = system.state_equation(@inbounds rhobar[particle])
    end

    # ------------------------------------------------------------
    # (2) Resolved RHS: dv_resolved = F(ubar, rhobar, pbar)
    #     pressure + selected base viscosity model, self-interaction only.
    # ------------------------------------------------------------
    viscosity_model_ = viscosity_model(system, system)

    foreach_point_neighbor(system, system,
                           system_coords, system_coords, semi;
                           points=each_integrated_particle(system)) do particle,
                                                                        neighbor,
                                                                        pos_diff,
                                                                        distance
        rho_a = @inbounds rhobar[particle]
        rho_b = @inbounds rhobar[neighbor]
        rho_mean = (rho_a + rho_b) / 2

        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

        m_a = @inbounds hydrodynamic_mass(system, particle)
        m_b = @inbounds hydrodynamic_mass(system, neighbor)

        p_a = @inbounds pbar[particle]
        p_b = @inbounds pbar[neighbor]

        dv_pressure = pressure_acceleration(system, system,
                                            particle, neighbor,
                                            m_a, m_b, p_a, p_b,
                                            rho_a, rho_b,
                                            pos_diff, distance,
                                            grad_kernel, system.correction)

        dv_visc = zero(dv_pressure)
        if viscosity_model_ isa ArtificialViscosityMonaghan
            v_a = @inbounds extract_svector(ubar, system, particle)
            v_b = @inbounds extract_svector(ubar, system, neighbor)
            v_diff = v_a - v_b

            h_a = smoothing_length(system, particle)
            h_b = smoothing_length(system, neighbor)
            h_avg = (h_a + h_b) / 2

            nu_a = kinematic_viscosity(system, viscosity_model_, h_a, sound_speed)
            nu_b = kinematic_viscosity(system, viscosity_model_, h_b, sound_speed)

            pi_ab = viscosity_model_(sound_speed, v_diff, pos_diff, distance,
                                     rho_mean, rho_a, rho_b,
                                     h_avg, grad_kernel, nu_a, nu_b)

            dv_visc = m_b * pi_ab
        elseif viscosity_model_ isa ViscosityAdami || viscosity_model_ isa ViscosityAdamiSGS
            dv_visc = viscosity_model_(system, system,
                                       ubar, ubar,
                                       particle, neighbor,
                                       pos_diff, distance,
                                       sound_speed,
                                       m_a, m_b,
                                       rho_a, rho_b,
                                       grad_kernel)
        end

        dv_pair = dv_pressure + dv_visc
        for i in 1:NDIMS
            @inbounds dv_resolved[i, particle] += dv_pair[i]
        end
    end

    # ------------------------------------------------------------
    # (3) Residual and fine scales: uprime = -tau * (dvbar_pred - dv_resolved)
    # ------------------------------------------------------------
    min_shepard = model.min_shepard
    @threaded semi for particle in each_integrated_particle(system)
        for i in 1:NDIMS
            @inbounds residual[i, particle] = dvbar_pred[i, particle] - dv_resolved[i, particle]
        end

        d = @inbounds den[particle]
        if d < min_shepard
            for i in 1:NDIMS
                @inbounds uprime[i, particle] = zero(eltype(uprime))
            end
        else
            ubar_p = @inbounds extract_svector(ubar, system, particle)
            h_p = smoothing_length(system, particle)
            tau = h_p / (sound_speed + norm(ubar_p) + eps(eltype(h_p)))
            tau_max = h_p / (sound_speed + eps(eltype(h_p)))
            tau = min(tau, tau_max)
            tau = isfinite(tau) ? tau : zero(tau)
            uprime_max = 2 * sound_speed

            for i in 1:NDIMS
                up = -tau * (@inbounds residual[i, particle])
                up = isfinite(up) ? up : zero(up)
                up = clamp(up, -uprime_max, uprime_max)
                @inbounds uprime[i, particle] = up
            end
        end
    end

    # ------------------------------------------------------------
    # (4) LES term acting on fine scales only: viscosity force based on u'
    #     using Adami Laplacian form + Smagorinsky-type nu_T'.
    # ------------------------------------------------------------
    C_s = model.C_s
    eps_model = model.epsilon
    strain_mode = model.strain_mode

    dv_vel_ref = current_velocity(dv, system)

    foreach_point_neighbor(system, system,
                           system_coords, system_coords, semi;
                           points=each_integrated_particle(system)) do particle,
                                                                        neighbor,
                                                                        pos_diff,
                                                                        distance
        if (@inbounds den[particle] < min_shepard) || (@inbounds den[neighbor] < min_shepard)
            return nothing
        end

        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

        m_a = @inbounds hydrodynamic_mass(system, particle)
        m_b = @inbounds hydrodynamic_mass(system, neighbor)

        rho_a = @inbounds rhobar[particle]
        rho_b = @inbounds rhobar[neighbor]

        # strain surrogate uses either fine or coarse differences
        if strain_mode === :fine
            s_a = @inbounds extract_svector(uprime, system, particle)
            s_b = @inbounds extract_svector(uprime, system, neighbor)
        else
            s_a = @inbounds extract_svector(ubar, system, particle)
            s_b = @inbounds extract_svector(ubar, system, neighbor)
        end
        s_diff = s_a - s_b
        h_a = smoothing_length(system, particle)
        h_b = smoothing_length(system, neighbor)
        h_avg = (h_a + h_b) / 2

        S_mag = norm(s_diff) / (distance + eps_model)
        S_mag = isfinite(S_mag) ? S_mag : zero(S_mag)
        S_cap = sound_speed / (h_avg + eps_model)
        S_mag = min(S_mag, S_cap)

        nu_T = (C_s * h_avg)^2 * S_mag
        nu_T = isfinite(nu_T) ? nu_T : zero(nu_T)
        nu_T_max = C_s^2 * h_avg * sound_speed
        nu_T = min(nu_T, nu_T_max)

        # apply viscosity to u' only
        v_a = @inbounds extract_svector(uprime, system, particle)
        v_b = @inbounds extract_svector(uprime, system, neighbor)
        v_diff = v_a - v_b

        dv_model = adami_viscosity_force(h_avg, pos_diff, distance, grad_kernel,
                                         m_a, m_b, rho_a, rho_b, v_diff,
                                         nu_T, nu_T, eps_model)

        for i in 1:NDIMS
            dv_i = dv_model[i]
            dv_i = isfinite(dv_i) ? dv_i : zero(dv_i)
            @inbounds dv_vel_ref[i, particle] += dv_i
        end
    end

    return dv
end