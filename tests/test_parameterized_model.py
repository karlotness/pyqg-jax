# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import jax
import pyqg_jax


def test_tree_flatten_roundtrip():
    param_model = pyqg_jax.parameterizations.ParameterizedModel(
        pyqg_jax.qg_model.QGModel(),
        lambda model_state, param_aux, model: (model.get_updates(model_state), None),
        lambda model_state, model: None,
    )
    leaves, treedef = jax.tree_util.tree_flatten(
        param_model,
        is_leaf=lambda obj: not isinstance(
            obj, pyqg_jax.parameterizations.ParameterizedModel
        ),
    )
    restored_model = jax.tree_util.tree_unflatten(treedef, leaves)
    assert vars(restored_model) == vars(param_model)
