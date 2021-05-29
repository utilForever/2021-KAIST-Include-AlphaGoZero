import numpy as np
from tensorflow.keras.optimizers import SGD

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil


class PolicyAgent(Agent):
    def __init__(self, model, encoder):
        self._model = model
        self._encoder = encoder

    def set_collector(self, collector):
        self._collector = collector

    def select_move(self, game_state):
        board_tensor = self._encoder.encode(game_state)
        X = np.array([board_tensor])
        move_probs = self._model.predict(X)[0]

        move_probs = clip_probs(move_probs)

        num_moves = self._encoder.board_width * self._encoder.board_height
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)

        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            is_valid = game_state.is_valid_move(move)
            is_an_eye = is_point_an_eye(game_state.board, point, game_state.next_player)
            if is_valid and (not is_an_eye):
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    def train(self, experience, lr=0.0000001, clipnorm=1.0, batch_size=512):
        opt = SGD(lr=lr, clipnorm=clipnorm)
        self._model.compile(loss='categorical_crossentropy', optimizer=opt)

        n = experience.states.shape[0]
        # Translate the actions/rewards.
        num_moves = self._encoder.board_width * self._encoder.board_height
        y = np.zeros((n, num_moves))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            y[i][action] = reward

        self._model.fit(
            experience.states, y,
            batch_size=batch_size,
            epochs=1)


def load_policy_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(
        h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(encoder_name, (board_width, board_height))
    return PolicyAgent(model, encoder)
