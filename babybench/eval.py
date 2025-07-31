import numpy as np
import os
import pickle
import babybench.utils as bb_utils

class Eval():

    # --- MODIFIED: Added 'training_log_dir' to __init__ ---
    def __init__(self, env, duration, render, save_dir, training_log_dir=None):
        self._env = env
        self._duration = duration
        self._render = render
        self._save_dir = save_dir
        self._images = []
        # If no specific training log dir is given, assume it's the same as the save_dir
        self._training_log_dir = training_log_dir if training_log_dir is not None else self._save_dir

    def reset(self):
        self._images = []
        self._init_track()
        
    def _eval_logs(self, logs):
        return None

    def eval_logs(self):
        # --- MODIFIED: Use the correct directory to read training logs ---
        log_path = os.path.join(self._training_log_dir, 'logs', 'training.pkl')
        try:
            with open(log_path, 'rb') as f:
                logs = pickle.load(f)
        except FileNotFoundError:
            print(f'Training logs not found at: {log_path}')
            print(f'Ensure this is the correct base directory from your training run.')
            return None
        score = self._eval_logs(logs)
        print(f'Preliminary training score: {score}')
        return None

    def _init_track(self):
        self._trajectories = {}
        self._trajectories['qpos'] = []
        self._trajectories['info'] = []
    
    def track(self, info):
        self._trajectories['qpos'].append(self._env.data.qpos.copy())
        self._trajectories['info'].append(info)

    def _render_image(self):
        self._images.append(bb_utils.evaluation_img(self._env))

    def eval_step(self, info):
        self.track(info)
        if self._render:
            self._render_image()

    def end(self, episode=0):
        # This part now works correctly because the directories are created beforehand
        # in the main evaluation.py script.
        with open(f'{self._save_dir}/logs/evaluation_{episode}.pkl', 'wb') as f:
            pickle.dump(self._trajectories, f, -1)
        if self._render:
            bb_utils.evaluation_video(self._images, f'{self._save_dir}/videos/evaluation_{episode}.avi')

class EvalSelfTouch(Eval):

    def __init__(self, **kwargs):
        super(EvalSelfTouch, self).__init__(**kwargs)

    def _eval_logs(self, logs):
        n_episodes = min(10001, len(logs))
        right_touches = np.array([])
        left_touches = np.array([])
        for ep in range(1,n_episodes):
            right_touches = np.unique(np.concatenate(
                (right_touches, logs[-ep]['right_hand_touches']),0))
            left_touches = np.unique(np.concatenate(
                (left_touches, logs[-ep]['left_hand_touches']),0))
        score = (len(right_touches) + len(left_touches)) / (34*2)
        return score

    def _render_image(self):
        self._images.append(bb_utils.evaluation_img(self._env, up='touches_with_hands'))

class EvalHandRegard(Eval):

    def __init__(self, **kwargs):
        super(EvalHandRegard, self).__init__(**kwargs)

    def _eval_logs(self, logs):
        n_episodes = min(10001, len(logs))
        hand_in_view = 0
        steps = 0
        for ep in range(1,n_episodes):
            hand_in_view += logs[-ep]['right_eye_right_hand']
            hand_in_view += logs[-ep]['left_eye_right_hand']
            hand_in_view += logs[-ep]['right_eye_left_hand']
            hand_in_view += logs[-ep]['left_eye_left_hand']
            steps += logs[-ep]['steps']
        score = hand_in_view / (4 * steps)
        return score

    def _render_image(self):
        self._images.append(bb_utils.evaluation_img(self._env, up='binocular'))

EVALS = {
    'none' : Eval,
    'self_touch' : EvalSelfTouch,
    'hand_regard' : EvalHandRegard,
}