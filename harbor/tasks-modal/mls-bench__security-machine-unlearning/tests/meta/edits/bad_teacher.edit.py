"""Bad Teacher baseline for security-machine-unlearning.

Faithful implementation of "Bad Teacher" unlearning from:
  Chundawat, Tarun, Mandal, Kankanhalli.
  "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks
   using an Incompetent Teacher."  AAAI 2023.
  Paper: https://arxiv.org/abs/2205.08096 (AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/25879)
  Code:  https://github.com/vikram2000b/bad-teaching-unlearning

Two frozen teachers are used:
  * ``competent teacher`` = a copy of the original (pre-unlearning) model.
  * ``incompetent teacher`` = a freshly random-initialised copy of the
    same architecture, with (near-)random outputs.
Per sample, the student distils from the incompetent teacher if the
sample is in the forget set, else from the competent teacher
(``UnlearnerLoss`` in the reference repo):

    KL( student || label * incompetent + (1-label) * competent )

with KL temperature (default 1).  Each optimizer step is fed a
balanced mixture of forget and retain samples (concatenated here).
"""

_FILE = "pytorch-vision/bench/unlearning/custom_unlearning.py"

_CONTENT = """\
import copy
import torch.nn as nn

class UnlearningMethod:
    \"\"\"Bad Teacher: dual-teacher KD with competent + incompetent teachers.

    Paper: https://arxiv.org/abs/2205.08096
    Reference code: https://github.com/vikram2000b/bad-teaching-unlearning
    \"\"\"

    def __init__(self):
        self.KL_temperature = 1.0
        self.competent = None       # = frozen original model
        self.incompetent = None     # = randomly re-initialised same-arch model

    def _freeze(self, m):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    def _random_reinit(self, m):
        # Kaiming init identical to initialize_weights() in run_unlearning.py.
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)
            elif isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight, mode='fan_in', nonlinearity='relu')
                if mod.bias is not None:
                    nn.init.constant_(mod.bias, 0)

    def _capture_teachers(self, model):
        self.competent = copy.deepcopy(model)
        self._freeze(self.competent)

        self.incompetent = copy.deepcopy(model)
        self._random_reinit(self.incompetent)
        self._freeze(self.incompetent)

    def _unlearner_loss(self, student_logits, full_teacher_logits,
                        unlearn_teacher_logits, is_forget):
        # Ref: UnlearnerLoss in vikram2000b/bad-teaching-unlearning.
        T = self.KL_temperature
        f_t = F.softmax(full_teacher_logits / T, dim=1)
        u_t = F.softmax(unlearn_teacher_logits / T, dim=1)
        lbl = is_forget.view(-1, 1).float()
        target = lbl * u_t + (1.0 - lbl) * f_t
        log_s = F.log_softmax(student_logits / T, dim=1)
        return F.kl_div(log_s, target, reduction='batchmean')

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        if self.competent is None:
            self._capture_teachers(model)

        retain_x, _ = retain_batch
        forget_x, _ = forget_batch

        # Balanced mini-batch: concatenate retain + forget samples.
        x = torch.cat([retain_x, forget_x], dim=0)
        is_forget = torch.cat([
            torch.zeros(retain_x.size(0), device=retain_x.device),
            torch.ones(forget_x.size(0), device=forget_x.device),
        ], dim=0)

        student_logits = model(x)
        with torch.no_grad():
            full_t = self.competent(x)
            unl_t = self.incompetent(x)

        loss = self._unlearner_loss(student_logits, full_t, unl_t, is_forget)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": float(loss.item())}
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 8, "end_line": 22, "content": _CONTENT}
]
