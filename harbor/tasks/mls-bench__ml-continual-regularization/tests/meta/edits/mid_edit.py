"""Mid-edit operations for cl-regularization task.

Applied to the continual-learning workspace after pre_edit, before the agent starts.

1. Creates custom_regularization.py from template
2. Patches classifier.py to import and call custom regularization loss
3. Patches train_task_based.py to use custom importance estimation and
   per-step hooks

Line numbers reference the codebase AFTER pre_edit (which is a no-op for
this package, so they match the original commit e6d795a).
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Patch 1: classifier.py — import custom module and replace regularization loss ──

# Insert import at top of classifier.py (after line 9, which is the last import)
_CLASSIFIER_IMPORT = """\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_regularization import compute_regularization_loss, CONFIG_OVERRIDES
"""

# Replace lines 288-298 of classifier.py (the weight_penalty block)
# Original:
#   weight_penalty_loss = None
#   if self.weight_penalty:
#       if self.importance_weighting=='si':
#           weight_penalty_loss = self.surrogate_loss()
#       elif self.importance_weighting=='fisher':
#           if self.fisher_kfac:
#               weight_penalty_loss = self.ewc_kfac_loss()
#           else:
#               weight_penalty_loss = self.ewc_loss()
#       loss_total += self.reg_strength * weight_penalty_loss
_CLASSIFIER_REGULARIZATION = """\
        weight_penalty_loss = None
        if self.weight_penalty:
            if hasattr(self, '_custom_importance') and hasattr(self, '_custom_prev_params'):
                weight_penalty_loss = compute_regularization_loss(
                    self, self._custom_importance, self._custom_prev_params
                )
            else:
                weight_penalty_loss = torch.tensor(0., device=self._device())
            _reg_scale = CONFIG_OVERRIDES.get('reg_strength_scale', 1.0)
            loss_total += (self.reg_strength * _reg_scale) * weight_penalty_loss
"""

# ── Patch 2: train_task_based.py — import custom module and replace importance estimation ──

_TRAIN_IMPORT = """\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_regularization import estimate_importance
"""

# Replace the SI initialization block (lines 39-41)
# Original:
#     # Register starting parameter values (needed for SI)
#     if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
#         model.register_starting_param_values()
_TRAIN_INIT = """\
    # Register starting parameter values (needed for SI)
    if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
        model.register_starting_param_values()

    # Initialize custom regularization state
    if isinstance(model, ContinualLearner) and model.weight_penalty:
        model._custom_importance = {}
        model._custom_prev_params = {}
        model._custom_W = {}
        model._custom_p_old = {}
        # Snapshot initial params
        for gen_params in model.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    model._custom_prev_params[n] = p.detach().clone()
                    model._custom_W[n] = p.data.clone().zero_()
                    model._custom_p_old[n] = p.data.clone()
"""

# Replace the SI per-step update block (lines 310-312)
# Original:
#                 # Update running parameter importance estimates in W (needed for SI)
#                 if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
#                     model.update_importance_estimates(W, p_old)
_TRAIN_PERSTEP = """\
                # Update running parameter importance estimates in W (needed for SI)
                if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
                    model.update_importance_estimates(W, p_old)

                # Update custom per-step accumulation (W tracks gradient*delta for SI-like methods)
                if isinstance(model, ContinualLearner) and model.weight_penalty and hasattr(model, '_custom_W'):
                    for gen_params in model.param_list:
                        for n, p in gen_params():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None and n in model._custom_W:
                                    model._custom_W[n].add_(-p.grad * (p.detach() - model._custom_p_old[n]))
                                if n in model._custom_p_old:
                                    model._custom_p_old[n] = p.detach().clone()
"""

# Replace the post-context importance estimation block (lines 349-367)
# Original lines 349-367:
#         # Parameter regularization: update and compute the parameter importance estimates
#         if context<len(train_datasets) and isinstance(model, ContinualLearner):
#             # -find allowed classes
#             allowed_classes = ...
#             ...
#             ##--> SI: calculate and update the normalized path integral
#             if model.importance_weighting=='si' and (model.weight_penalty or model.precondition):
#                 model.update_omega(W, model.epsilon)
_TRAIN_POSTCONTEXT = """\
        # Parameter regularization: update and compute the parameter importance estimates
        if context<len(train_datasets) and isinstance(model, ContinualLearner):
            # -if needed, apply correct context-specific mask
            if model.mask_dict is not None:
                allowed_classes = active_classes[-1] if (per_context and not per_context_singlehead) else active_classes
                model.apply_XdGmask(context=context)

            # Custom importance estimation (replaces built-in EWC/SI/OWM)
            if model.weight_penalty and hasattr(model, '_custom_importance'):
                prev_snap = {}
                for gen_params in model.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            prev_snap[n] = model._custom_prev_params.get(n, p.detach().clone())
                new_importance = estimate_importance(
                    model, training_dataset, prev_snap, device
                )
                # Accumulate importance
                for n in new_importance:
                    if n in model._custom_importance:
                        model._custom_importance[n] = model._custom_importance[n] + new_importance[n]
                    else:
                        model._custom_importance[n] = new_importance[n]
                # Update prev_params snapshot to current
                for gen_params in model.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            model._custom_prev_params[n] = p.detach().clone()
                # Reset per-step accumulators
                for n in model._custom_W:
                    model._custom_W[n].zero_()
                for gen_params in model.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            model._custom_p_old[n] = p.data.clone()
"""


# ── Operations (ordered bottom-to-top within each file) ──

OPS = [
    # 1. Create custom_regularization.py
    {
        "op": "create",
        "file": "continual-learning/custom_regularization.py",
        "content": _CUSTOM_PY,
    },
    # 2. Patch classifier.py — replace regularization loss block (L289-298)
    {
        "op": "replace",
        "file": "continual-learning/models/classifier.py",
        "start_line": 289,
        "end_line": 298,
        "content": _CLASSIFIER_REGULARIZATION,
    },
    # 3. Patch classifier.py — insert import (after line 9)
    {
        "op": "insert",
        "file": "continual-learning/models/classifier.py",
        "after_line": 9,
        "content": _CLASSIFIER_IMPORT,
    },
    # 4. Patch train_task_based.py — replace post-context importance (L349-367)
    {
        "op": "replace",
        "file": "continual-learning/train/train_task_based.py",
        "start_line": 349,
        "end_line": 367,
        "content": _TRAIN_POSTCONTEXT,
    },
    # 5. Patch train_task_based.py — replace per-step SI update (L310-312)
    {
        "op": "replace",
        "file": "continual-learning/train/train_task_based.py",
        "start_line": 310,
        "end_line": 312,
        "content": _TRAIN_PERSTEP,
    },
    # 6. Patch train_task_based.py — replace SI init (L39-41)
    {
        "op": "replace",
        "file": "continual-learning/train/train_task_based.py",
        "start_line": 39,
        "end_line": 41,
        "content": _TRAIN_INIT,
    },
    # 7. Patch train_task_based.py — insert import (after line 10)
    {
        "op": "insert",
        "file": "continual-learning/train/train_task_based.py",
        "after_line": 10,
        "content": _TRAIN_IMPORT,
    },
]
