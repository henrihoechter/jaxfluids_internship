from compressible_1d import equation_manager_types
from jaxtyping import Array, Float


def apply_chemistry_source_terms(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    pass
