import warnings
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.common.base_il_trainer_llm import BaseVLNCETrainerLLM
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

@baseline_registry.register_trainer(name="schedulesampler-OPENNAV")
class SSTrainer(BaseVLNCETrainerLLM):
    def __init__(self, config=None):
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    