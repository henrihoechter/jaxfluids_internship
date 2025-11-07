import jax
import jax.numpy as jnp

from compressible_1d import numerics, plot, boundary_conditions, physics

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

"""This script runs a Sod's Shock Tube scenario using different solvers. 

Note: plots are generated automatically, though, the connection between docker and 
the browser on the host machine proved to be unreliable. If the plots do not show,
re-run the script.
"""

# --- user input start ---

end_time = 1e0  # [s]
tube_length = 1  # [m]
n_cells = 400
gamma = 1.4

boundary_condition: str = "reflective"  # periodic, reflective, transmissive

# init in primitives and not normalized
rho1 = 1.0  # density, [kg/m3]
u1 = 0.0  # velocity, [m/s]
p1 = 1.0  # pressure , [N/m2]

rho2 = 0.125  # density, [kg/m3]
u2 = 0.0  # velocity, [m/s]
p2 = 0.1  # pressure , [N/m2]

is_debug = False  # if True, conservation check will print results verbosely
is_abort = False  # must be False for reflective and transmissive BCs

n_ghost_cells = 1  # per side
if n_ghost_cells > 1:
    raise NotImplementedError("Physics not yet validated for higher order.")

# --- user input end ---


delta_x = tube_length / n_cells

U1_primitive = jnp.array([rho1, u1, p1])
U2_primitive = jnp.array([rho2, u2, p2])
U_primitive = jnp.stack([U1_primitive, U2_primitive], axis=1)
U_conserved = physics.to_conserved(U_primitive, rho_ref=rho1, p_ref=p1, gamma=gamma)

delta_t = numerics.calculate_dt(U_primitive, gamma=gamma, delta_x=delta_x, cmax=0.40)
print(f"delta_t calculated according to CFL: {delta_t:.3e}s")

t_step_of_interest = int(0.25 / delta_t)
print(f"{t_step_of_interest=}")

n_steps = int(end_time / delta_t)
print(f"total time steps: {n_steps}")


U_init = boundary_conditions.initialize_two_domains(
    rho_left=U_conserved[0, 0],
    rho_right=U_conserved[0, 1],
    u_left=U_conserved[1, 0],
    u_right=U_conserved[1, 1],
    p_left=U_conserved[2, 0],
    p_right=U_conserved[2, 1],
    n_cells=n_cells,
)
cfd_input = numerics.Input(
    delta_x=delta_x,
    delta_t=delta_t,
    U_init=U_init,
    gamma=gamma,
    n_steps=n_steps,
    n_ghost_cells=n_ghost_cells,
    is_debug=is_debug,
    is_abort=is_abort,
    boundary_condition=boundary_condition,
    solver_type="lf",
)
U_solutions_lf = numerics.run(cfd_input)

cfd_input.solver_type = "hllc"
U_solutions_hllc = numerics.run(cfd_input)

cfd_input.solver_type = "exact"
U_solutions_exact = numerics.run(cfd_input)

plot.plot_U_heatmaps(U_solutions_exact).show()

plot.plot_U_heatmaps(U_solutions_exact - U_solutions_lf).update_layout(
    title="Difference: Exact - Lax-Friedrichs"
).show()
plot.plot_U_heatmaps(U_solutions_exact - U_solutions_hllc).update_layout(
    title="Difference: Exact - HLLC"
).show()
plot.plot_U_heatmaps(U_solutions_hllc - U_solutions_lf).update_layout(
    title="Difference: HLLC - Lax-Friedrichs"
).show()

fig = plot.plot_U_slice(
    U_field=U_solutions_lf[:, :, t_step_of_interest],
    to_primitive=True,
    gamma=gamma,
    line_dash="solid",
    legend_label="lax-friedrichs",
)
fig = plot.plot_U_slice(
    U_field=U_solutions_hllc[:, :, t_step_of_interest],
    to_primitive=True,
    gamma=gamma,
    fig=fig,
    line_dash="dash",
    legend_label="hllc",
)
fig = plot.plot_U_slice(
    U_field=U_solutions_exact[:, :, t_step_of_interest],
    to_primitive=True,
    gamma=gamma,
    fig=fig,
    line_dash="dot",
    legend_label="exact",
)
fig.show()
