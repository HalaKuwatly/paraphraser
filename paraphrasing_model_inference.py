import numpy as np
from keras import layers, models

from config import encoder_output_dim, SOS_TOKEN, EOS_TOKEN, max_length
from paraphrasing_model import ParaphrasingModel
from quora_dataset_loader import apply_vocab, encode_questions


class ParaphrasingModelInference:
    def __init__(self, itow, weights_file):
        self.itow = itow
        self.wtoi = {w: i for i, w in self.itow.items()}
        self.vocab_size = len(itow)
        self.trained_model = ParaphrasingModel(
            data_loader=None, itow=itow, load_generators=False
        )
        self.trained_model.model.load_weights(weights_file)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def _convert_to_sentence(self, seq):
        return " ".join([self.itow[i] for i in seq if i > 3])

    def _convert_to_tokens(self, seq):
        return [self.itow[i] for i in seq if i > 3]

    def encode_sents(self, sents):
        words = [sent.split() for sent in sents]
        sents = [apply_vocab(w, self.wtoi) for w in words]
        encoded = encode_questions(self.wtoi, sents)
        return encoded

    def generate_paraphrase(self, input_seq):
        shape = (1, input_seq.size, self.vocab_size)
        one_hot = np.zeros(shape)
        rows = np.arange(input_seq.size)
        one_hot[0, rows, input_seq] = 1

        # Encode the input as state vectors.
        enc_output, states_h, states_c = self.encoder.predict(one_hot)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first token of target sequence with the start token.
        target_seq[0, 0] = self.wtoi[SOS_TOKEN]

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, states_h, states_c = self.decoder.predict(
                [target_seq, states_h, states_c]
            )

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, 0, :])
            sampled_token = self.itow[sampled_token_index]
            if sampled_token != EOS_TOKEN:
                decoded_sentence.append(sampled_token)

            # Exit condition: either hit max length
            # or find EOS token.
            if (
                sampled_token == EOS_TOKEN
                or len(decoded_sentence) > max_length
            ):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

        return " ".join(decoded_sentence), decoded_sentence

    def build_encoder(self):
        input_question = self.trained_model.layers[3].inputs[0]

        trained_encoder = self.trained_model.layers[3]
        (
            encoded_question,
            encoder_state_h,
            encoder_state_c,
        ) = trained_encoder.outputs
        return models.Model(
            inputs=input_question,
            outputs=[encoded_question, encoder_state_h, encoder_state_c],
            name="InferenceEncoderModel",
        )

    def build_decoder(self):
        target_question = self.trained_model.layers[4].inputs[0]

        encoder_state_h = layers.Input(
            shape=(encoder_output_dim,), name="EncoderStateH"
        )  # encoder_state_h (batch_size, encoder_output_dim)

        encoder_state_c = layers.Input(
            shape=(encoder_output_dim,), name="EncoderStateC"
        )  # encoder_state_c (batch_size, encoder_output_dim)
        decoder_states_inputs = [encoder_state_h, encoder_state_c]
        decoder_lstm = self.trained_model.layers[4].layers[4]
        emb_layer = self.trained_model.layers[4].layers[1]
        emb = emb_layer(target_question)
        decoder_outputs, state_h, state_c = decoder_lstm(
            emb, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_dense = self.trained_model.layers[4].layers[5]
        decoder_outputs = decoder_dense(decoder_outputs)

        return models.Model(
            inputs=[target_question] + decoder_states_inputs,
            outputs=[decoder_outputs] + decoder_states,
            name="InferenceDecoderModel",
        )
