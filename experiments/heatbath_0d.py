"""Zero-dimensional heat bath test for vibrational relaxation.

This script tests the vibrational relaxation model in a constant volume
heat bath (0D). The system should relax toward thermal equilibrium T = Tv
according to the Millikan-White relaxation time.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from compressible_1d.two_temperature.config import TwoTemperatureModelConfig
from compressible_1d.two_temperature.thermodynamics import load_species_data
from compressible_1d.two_temperature.source_terms import (
    apply_chemistry_source,
    extract_primitives_from_U,
)
from compressible_1d.two_temperature.solver_adapter import primitives_to_U

jax.config.update("jax_enable_x64", False)

print("=" * 80)
print("0D Heat Bath Test: Vibrational Relaxation")
print("=" * 80)

# Configuration
config = TwoTemperatureModelConfig(
    species_names=["N2", "N"],
    kinetics_model="park",
    controlling_temperature_mode="geometric_mean",
    diffusion_model="constant",
    constant_diffusion_coeff=1e-4,
    relaxation_model="millikan_white_park",
    dissociation_model="preferential",
    chemistry_substeps=1,  # Single substep for 0D
)

# Load species
species_list = [load_species_data(name) for name in config.species_names]

# Initial conditions (0D - single cell)
Y = jnp.array([[1.0], [0.0]])  # Pure N2
u = jnp.array([0.0])
T = jnp.array([3000.0])  # Translational temperature
Tv = jnp.array([1000.0])  # Vibrational temperature (cold)
rho = jnp.array([1.0])  # kg/m^3

print("\nInitial conditions:")
print(f"  ρ = {rho[0]:.3f} kg/m³")
print(f"  T = {T[0]:.1f} K (translational)")
print(f"  Tv = {Tv[0]:.1f} K (vibrational)")
print(f"  Y_N2 = {Y[0,0]:.3f}, Y_N = {Y[1,0]:.3f}")

# Convert to conserved variables
U = primitives_to_U(Y, u, T, Tv, rho, species_list, config)

print(f"\nState vector shape: {U.shape}")

# Time integration
end_time = 1e-3  # 1 ms
dt = 1e-6  # 1 microsecond
n_steps = int(end_time / dt)

print("\nTime integration:")
print(f"  End time: {end_time:.2e} s")
print(f"  Time step: {dt:.2e} s")
print(f"  Number of steps: {n_steps}")

# Storage for history
T_history = jnp.zeros(n_steps + 1)
Tv_history = jnp.zeros(n_steps + 1)
Y_N2_history = jnp.zeros(n_steps + 1)
time_history = jnp.zeros(n_steps + 1)

# Initial state
T_history = T_history.at[0].set(T[0])
Tv_history = Tv_history.at[0].set(Tv[0])
Y_N2_history = Y_N2_history.at[0].set(Y[0, 0])
time_history = time_history.at[0].set(0.0)

print("\nIntegrating...")
U_current = U

for i in range(n_steps):
    if i % (n_steps // 10) == 0:
        print(f"  Step {i}/{n_steps}")

    # Apply chemistry source term
    U_current = apply_chemistry_source(U_current, dt, species_list, config)

    # Extract primitives
    Y_curr, rho_curr, T_curr, Tv_curr, p_curr = extract_primitives_from_U(
        U_current, species_list, config
    )

    # Store
    T_history = T_history.at[i + 1].set(T_curr[0])
    Tv_history = Tv_history.at[i + 1].set(Tv_curr[0])
    Y_N2_history = Y_N2_history.at[i + 1].set(Y_curr[0, 0])
    time_history = time_history.at[i + 1].set((i + 1) * dt)

print("\nIntegration complete!")

# Final state
Y_final, rho_final, T_final, Tv_final, p_final = extract_primitives_from_U(
    U_current, species_list, config
)

print("\nFinal state:")
print(f"  T = {T_final[0]:.1f} K")
print(f"  Tv = {Tv_final[0]:.1f} K")
print(f"  ΔT = {abs(T_final[0] - Tv_final[0]):.1f} K (equilibrium when ΔT → 0)")
print(f"  Y_N2 = {Y_final[0,0]:.6f}")
print(f"  Y_N = {Y_final[1,0]:.6f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Temperature evolution
ax1.plot(time_history * 1e6, T_history, label="T (translational)", linewidth=2)
ax1.plot(
    time_history * 1e6,
    Tv_history,
    label="Tv (vibrational)",
    linewidth=2,
    linestyle="--",
)
ax1.set_xlabel("Time [µs]")
ax1.set_ylabel("Temperature [K]")
ax1.set_title("Vibrational Relaxation: Temperature Evolution")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Species evolution
ax2.plot(time_history * 1e6, Y_N2_history, label="Y_N2", linewidth=2)
ax2.plot(
    time_history * 1e6, 1.0 - Y_N2_history, label="Y_N", linewidth=2, linestyle="--"
)
ax2.set_xlabel("Time [µs]")
ax2.set_ylabel("Mass Fraction")
ax2.set_title("Chemical Evolution: N2 Dissociation")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "/home/hhoechter/tum/jaxfluids_internship/experiments/heatbath_0d_results.png",
    dpi=150,
)
print("\nPlot saved to: experiments/heatbath_0d_results.png")

plt.show()

print("\n" + "=" * 80)
print("0D Heat Bath Test Complete!")
print("=" * 80)
print("\nObservations:")
print("  - Tv should relax toward T exponentially with characteristic time τ_v")
print("  - At high T, N2 dissociation produces N atoms")
print("  - Preferential dissociation couples chemistry and vibrational relaxation")
