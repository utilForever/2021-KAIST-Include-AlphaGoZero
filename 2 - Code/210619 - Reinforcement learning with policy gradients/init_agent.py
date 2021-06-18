import argparse
import h5py

from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from dlgo import agent
from dlgo import encoders
from dlgo import networks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('output_file')
    args = parser.parse_args()

    encoder = encoders.get_encoder_by_name('simple', args.board_size)
    model = Sequential()
    for layer in networks.large.layers(encoder.shape()):
        model.add(layer)
    model.add(Dense(encoder.num_points()))
    model.add(Activation('softmax'))
    opt = SGD(lr=0.02)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    new_agent = agent.PolicyAgent(model, encoder)
    with h5py.File(args.output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()