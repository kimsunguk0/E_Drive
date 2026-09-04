"""non-finite loss/grad가 나오면 **그 스텝만 건너뛴다**.

왜 필요한가
----------
v2a 330 pilot이 iter 621에서 죽었다. 직접 원인은 det assigner의 cost에 NaN이
들어간 것인데, 그 NaN의 출처는 **한 스텝 전의 NaN gradient**였다.

PyTorch AdamW는 `lr=0`에서도
    p.addcdiv_(exp_avg, denom, value=-step_size)
를 실행하고 IEEE754에서 `0 * NaN = NaN`이다. 게다가 `clip_grad_norm_`은
`total_norm`이 non-finite면 **모든** grad에 그 값을 곱한다. 그래서 어느 한 곳의
NaN gradient가 **동결 파라미터(lr_mult=0)까지 포함한 전체 모델**을 한 스텝에
죽인다. `lr=0`은 NaN 격리 수단이 아니다.

이 hook은 AMP `GradScaler`와 같은 표준 방식으로 그 스텝을 버린다.
divergence를 숨기지 않는다 -- 건너뛴 횟수와 최근 사유를 로그에 남기므로,
드문 사건인지 체계적 발산인지 로그로 판별된다. `max_consecutive`번 연속이면
멈춘다 (조용히 망가진 모델로 24시간을 태우는 것을 막는다).

    optimizer_config = dict(type='NanSkipOptimizerHook',
                            grad_clip=dict(max_norm=35, norm_type=2))
"""
import torch
from mmcv.runner import HOOKS
from mmcv.runner.hooks.optimizer import OptimizerHook


@HOOKS.register_module()
class NanSkipOptimizerHook(OptimizerHook):

    def __init__(self, max_consecutive=20, log_interval=1, **kwargs):
        super().__init__(**kwargs)
        self.max_consecutive = int(max_consecutive)
        self.log_interval = int(log_interval)
        self.skipped = 0
        self.consecutive = 0

    def _bad_params(self, model):
        return [n for n, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()]

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        loss = runner.outputs['loss']
        if not torch.isfinite(loss):
            # backward를 아예 하지 않는다 (낭비 방지). grad는 이미 zero다.
            self._skip(runner, f"loss={float(loss)}", [])
            return
        loss.backward()

        # `clip_grad_norm_`이 반환하는 total_norm 하나로 전 grad를 커버한다 --
        # 어느 파라미터든 NaN/inf면 norm이 non-finite가 된다. 파라미터 480개를
        # 매 iteration 스캔하면 커널 런치만 480회라 낭비고, 스캔은 **건너뛸 때만**
        # 진단용으로 한다.
        assert self.grad_clip is not None, \
            "NanSkipOptimizerHook은 grad_clip을 전제로 한다 (total_norm으로 판정)"
        grad_norm = self.clip_grads(runner.model.parameters())
        gn = float(grad_norm) if grad_norm is not None else 0.0
        if gn != gn or gn in (float('inf'), float('-inf')):
            # 이 시점 grad는 이미 clip에서 NaN이 곱해져 있다. step하면 lr=0인
            # 동결 파라미터까지 전부 죽는다 (AdamW의 0 x NaN = NaN).
            self._skip(runner, f"grad_norm={gn}", self._bad_params(runner.model))
            runner.optimizer.zero_grad()
            return
        runner.log_buffer.update({'grad_norm': gn},
                                 runner.outputs['num_samples'])
        runner.optimizer.step()
        self.consecutive = 0

    def _skip(self, runner, reason, bad):
        self.skipped += 1
        self.consecutive += 1
        if self.skipped <= 20 or self.skipped % 50 == 0:
            runner.logger.warning(
                f"[NanSkip] iter {runner.iter + 1} 스텝 건너뜀 ({reason}). "
                f"누적 {self.skipped}회, 연속 {self.consecutive}회."
                + (f" 예: {bad[:3]}" if bad else ""))
        runner.log_buffer.update({'nan_skip': 1.0},
                                 runner.outputs['num_samples'])
        assert self.consecutive < self.max_consecutive, (
            f"non-finite가 {self.consecutive}회 연속이다 -- 드문 사건이 아니라 "
            f"발산이다. 조용히 망가진 모델로 학습을 계속하지 않는다. "
            f"마지막 사유: {reason}")
