import torch
import time
from tqdm import tqdm
from .logger import get_logger
from .model_trainer import task_predict_fn_dict


class ModelInference(object):
    def __init__(self, job_dir, device, model, saver, saved_model_path, task):
        self.job_dir = job_dir
        self.device = device
        self.model = model
        self.logger = get_logger(job_dir)
        self.saver = saver
        self.saved_model_path = saved_model_path
        self.model.load_state_dict(torch.load(self.saved_model_path, map_location=self.device))
        self.predict_fn = task_predict_fn_dict[task]

    def inference(self, dataloader):
        self.model.eval()
        for inputs in dataloader:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.set_grad_enabled(False):
                logits, state = self.model(**inputs)
            prob, pred = self.predict_fn(logits)
            state.update({'pred': pred, 'prob': prob, 'logits': logits})
            yield state

    def __call__(self, dataloader):
        start_inference = time.time()
        self.model.to(self.device)
        for model_state in tqdm(self.inference(dataloader)):
            self.saver(model_state)
        time_elapsed = time.time() - start_inference
        self.logger.info(f'Inference completed in {time_elapsed // 3600:.0f}h {time_elapsed % 3600 // 60:.0f}m '
                         f'{time_elapsed % 3600 % 60:.0f}s')


class ModelInferenceEnsemble(ModelInference):
    def __init__(self, job_dir, device, model, saver, saved_model_paths, task):
        super().__init__(job_dir, device, model, saver, saved_model_paths[0], task)
        assert isinstance(saved_model_paths, list)
        self.saved_model_paths = saved_model_paths

    def inference(self, dataloader):
        self.model.eval()
        for inputs in dataloader:
            inputs = {key: value.to(self.device) for key, value in inputs.item()}
            state, prob_sum = {}, None
            for saved_model_path in self.saved_model_paths:
                self.model.load_state_dict(torch.load(saved_model_path, map_location=self.device))
                with torch.set_grad_enabled(False):
                    logits, state = self.model(**inputs)
                probs, preds = self.predict_fn(logits)
                prob_sum = probs if prob_sum is None else prob_sum + probs
            preds = torch.argmax(prob_sum, dim=1)
            probs = prob_sum / len(self.saved_model_paths)
            state.update({'pred': preds, 'prob': probs, 'logits': probs})
            yield state
