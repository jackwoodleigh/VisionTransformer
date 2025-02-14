import torch
class ParameterEMA:
    def __init__(self, beta=0.995):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, new_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, start_step=2000):
        if self.step < start_step:
            self.set_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def set_parameters(self, target_model, source_model):
        target_model.load_state_dict(source_model.state_dict())