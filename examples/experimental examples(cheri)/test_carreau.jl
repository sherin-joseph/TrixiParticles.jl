using TrixiParticles
using LinearAlgebra

# Parameters from test
nu0 = 3.5e-6
nu_inf = 1.0e-6
lambda = 3.313e-2
a_param = 2.0
n_param = 0.3
epsilon_param = 0.01

# v_diff and pos_diff from test setup
v_diff_vec = [0.1, -0.3]
pos_diff_vec = [0.1, -0.1]
distance = norm(pos_diff_vec)

# Calculate gamma_dot
gamma_dot = norm(v_diff_vec) / (distance + epsilon_param)
println("gamma_dot = ", gamma_dot)

# Calculate nu_eff using Carreau-Yasuda formula
nu_eff = nu_inf +
         (nu0 - nu_inf) * (1 + (lambda * gamma_dot)^a_param)^((n_param - 1) / a_param)
println("nu_eff = ", nu_eff)

# Now we need to compute the actual viscosity force
# This requires the full adami_viscosity_force calculation
# For now, let's just print what we have
println("\nExpected to be very small compared to Newtonian case since nu_eff is tiny")
