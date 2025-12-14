"""Simplified two-temperature test with thermal equilibrium initial conditions."""

import jax
import jax.numpy as jnp

from compressible_1d.two_temperature.config import TwoTemperatureModelConfig
from compressible_1d.two_temperature.thermodynamics import load_species_data
from compressible_1d.two_temperature.solver_adapter import (
    initialize_two_temperature_shock_tube,
    U_to_primitives,
)
from compressible_1d import numerics

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)

print("=" * 80)
print("Two-Temperature Model: Simplified Test (Thermal Equilibrium)")
print("=" * 80)

# Configuration
config = TwoTemperatureModelConfig(
    species_names=("N2", "N"),
    kinetics_model="park",
    controlling_temperature_mode="geometric_mean",
    diffusion_model="constant",
    constant_diffusion_coeff=1e-4,
    relaxation_model="millikan_white_park",
    dissociation_model="preferential",
    chemistry_substeps=10,  # More substeps for stability
    source_term_splitting="strang",
    gamma_mode="mixture",
)

print("\nConfiguration:")
print(f"  Species: {config.species_names}")
print(f"  Chemistry substeps: {config.chemistry_substeps}")

# Load species data
species_list = [load_species_data(name) for name in config.species_names]

# Simulation parameters
tube_length = 1.0  # m
n_cells = 200
delta_x = tube_length / n_cells
end_time = 1e-4  # s

# Initial conditions: LOWER TEMPERATURE and THERMAL EQUILIBRIUM
# Left state: Moderate temperature, equilibrium
Y_left = jnp.array([1.0, 0.0])  # Pure N2
u_left = 0.0
T_left = 2000.0  # K (reduced from 5000K)
Tv_left = 2000.0  # K (equilibrium: T = Tv)
rho_left = 1.0  # kg/m^3

# Right state: Low temperature, equilibrium
Y_right = jnp.array([1.0, 0.0])  # Pure N2
u_right = 0.0
T_right = 300.0  # K
Tv_right = 300.0  # K (equilibrium)
rho_right = 0.125  # kg/m^3

print("\nInitial conditions (THERMAL EQUILIBRIUM):")
print(f"  Left:  ρ={rho_left} kg/m³, T=Tv={T_left} K, Y_N2={Y_left[0]:.2f}")
print(f"  Right: ρ={rho_right} kg/m³, T=Tv={T_right} K, Y_N2={Y_right[0]:.2f}")

# Initialize
print("\nInitializing...")
U_init = initialize_two_temperature_shock_tube(
    Y_left,
    u_left,
    T_left,
    Tv_left,
    rho_left,
    Y_right,
    u_right,
    T_right,
    Tv_right,
    rho_right,
    n_cells,
    species_list,
    config,
)

# Calculate time step
print("\nCalculating time step...")
delta_t = numerics.calculate_dt_two_temperature(
    U_init, species_list, config, delta_x, cmax=0.3
)
n_steps = min(int(end_time / delta_t), 10000)  # Limit to 10000 steps
actual_end_time = n_steps * delta_t

print(f"  Time step: {delta_t:.3e} s")
print(f"  Number of steps: {n_steps}")
print(f"  Actual end time: {actual_end_time:.3e} s")

if delta_t < 1e-9:
    print(f"\n*** WARNING: Time step is very small ({delta_t:.1e} s) ***")
    print("*** This indicates stiff source terms - consider implicit integration ***")
    print("*** Simulation may take a long time or be numerically unstable ***")
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != "y":
        print("Simulation aborted.")
        exit(0)

# Run simulation
print("\nRunning simulation...")
print("  (This may take a moment for JIT compilation)")

U_solutions = numerics.run_two_temperature(
    U_init=U_init,
    delta_x=delta_x,
    delta_t=delta_t,
    n_steps=n_steps,
    species_list=species_list,
    config=config,
    boundary_condition="transmissive",
    n_ghost_cells=1,
    is_debug=True,
)

print("\nSimulation complete!")
print(f"  Solution shape: {U_solutions.shape}")

# Extract final state
U_final = U_solutions[:, :, -1]

# Convert to primitives
Y_final, u_final, T_final, Tv_final, p_final, rho_final = U_to_primitives(
    U_final, species_list, config
)

print("\nFinal state summary:")
print(
    f"  N2 mass fraction range: [{float(jnp.min(Y_final[0])):.4f}, {float(jnp.max(Y_final[0])):.4f}]"
)
print(
    f"  N mass fraction range: [{float(jnp.min(Y_final[1])):.4f}, {float(jnp.max(Y_final[1])):.4f}]"
)
print(
    f"  Temperature range: [{float(jnp.min(T_final)):.1f}, {float(jnp.max(T_final)):.1f}] K"
)
print(
    f"  Vib. temperature range: [{float(jnp.min(Tv_final)):.1f}, {float(jnp.max(Tv_final)):.1f}] K"
)
print(
    f"  Velocity range: [{float(jnp.min(u_final)):.1f}, {float(jnp.max(u_final)):.1f}] m/s"
)

# Check for NaN
if jnp.any(jnp.isnan(U_final)):
    print("\n*** ERROR: NaN detected in final solution! ***")
else:
    print("\n✓ Solution is valid (no NaN)")

# Conservation checks
mass_init = jnp.sum(U_init[0] + U_init[1])
mass_final = jnp.sum(U_final[0] + U_final[1])
mass_error = abs(mass_final - mass_init) / mass_init

print("\nConservation checks:")
print(f"  Mass conservation error: {float(mass_error):.3e}")

print("\n" + "=" * 80)
print("Simulation completed!")
print("=" * 80)
