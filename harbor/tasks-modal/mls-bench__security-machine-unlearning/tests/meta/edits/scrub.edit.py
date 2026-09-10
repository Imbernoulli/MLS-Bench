"""SCRUB baseline for security-machine-unlearning.

Faithful implementation of SCRUB from:
  Kurmanji, Triantafillou, Hayes, Triantafillou.
  "Towards Unbounded Machine Unlearning."  NeurIPS 2023.
  Paper: https://arxiv.org/abs/2302.09880
  Code:  https://github.com/meghdadk/SCRUB

SCRUB keeps a frozen teacher = the original pre-unlearning model.  The
student is initialised from the teacher and alternates between:
  * Max step (forget set): MAXIMISE KL(student || teacher), i.e. loss
    = -KL_T(student_logits, teacher_logits).
  * Min step (retain set): MINIMISE gamma * CE(student, y)
    + alpha * KL_T(student || teacher).
The max step is only performed for the first ``msteps`` epochs (rewind),
thereafter only the min step is applied.  Defaults (msteps=2,
kd_T=4, alpha=0.01, gamma=0.99) follow the authors' VGG notebook
(see meghdadk/SCRUB repo).
"""

_FILE = "pytorch-vision/bench/unlearning/custom_unlearning.py"

_CONTENT = """\
import copy

class UnlearningMethod:
    \"\"\"SCRUB: min-max KL distillation vs a frozen original model.

    Paper: https://arxiv.org/abs/2302.09880
    Reference code: https://github.com/meghdadk/SCRUB
    \"\"\"

    def __init__(self):
        # Defaults from the authors' VGG notebook.
        self.msteps = 2        # number of max-step epochs (rewind)
        self.kd_T = 4.0        # KD temperature
        self.alpha = 0.01      # weight on KL(student || teacher) in min step
        self.gamma = 0.99      # weight on CE(student, y) in min step
        self.teacher = None    # lazily captured on first step

    def _kd_kl(self, student_logits, teacher_logits):
        # KL(student || teacher) with temperature, as in Hinton KD.
        T = self.kd_T
        p_s = F.log_softmax(student_logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)

    def _capture_teacher(self, model):
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        if self.teacher is None:
            self._capture_teacher(model)

        retain_x, retain_y = retain_batch
        forget_x, _ = forget_batch

        # ---- Max step on forget set (only during the first msteps epochs) ----
        forget_kl_val = 0.0
        if epoch < self.msteps:
            optimizer.zero_grad()
            s_forget = model(forget_x)
            with torch.no_grad():
                t_forget = self.teacher(forget_x)
            forget_kl = self._kd_kl(s_forget, t_forget)
            (-forget_kl).backward()
            optimizer.step()
            forget_kl_val = forget_kl.item()

        # ---- Min step on retain set (every epoch) ----
        optimizer.zero_grad()
        s_retain = model(retain_x)
        with torch.no_grad():
            t_retain = self.teacher(retain_x)
        retain_ce = F.cross_entropy(s_retain, retain_y)
        retain_kl = self._kd_kl(s_retain, t_retain)
        loss = self.gamma * retain_ce + self.alpha * retain_kl
        loss.backward()
        optimizer.step()

        return {
            "loss": float(loss.item()),
            "retain_ce": float(retain_ce.item()),
            "retain_kl": float(retain_kl.item()),
            "forget_kl": float(forget_kl_val),
        }
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 8, "end_line": 22, "content": _CONTENT}
]
