from keras import layers, models, losses, callbacks, optimizers, backend as K

from quora_dataset_loader import QuoraDatasetLoader
from data_generator import DataGenerator
from config import (
    max_length,
    emb_hid_dim,
    encoder_output_dim,
    encoder_dropout,
    encoder_rnn_units,
    emb_dim,
    decoder_rnn_units,
    batch_size,
    lr,
)


class ParaphrasingModel:
    def __init__(
        self, data_loader: QuoraDatasetLoader, itow, load_generators=True
    ):
        self.vocab_size = len(itow)
        self.model = self.build_model()
        self.model.compile(optimizer=optimizers.RMSprop(lr=lr))
        if load_generators:
            limit = len(data_loader.train_questions[0])
            self.train_generator = DataGenerator(
                x=[
                    data_loader.train_questions[0][:limit],
                    data_loader.train_questions[2][:limit],
                ],
                y=data_loader.train_questions[2][:limit],
                max_length=max_length,
                vocab_size=self.vocab_size,
                batch_size=batch_size,
            )

            self.val_generator = DataGenerator(
                x=[data_loader.val_questions[0], data_loader.val_questions[2]],
                y=data_loader.val_questions[2],
                max_length=max_length,
                vocab_size=self.vocab_size,
                batch_size=batch_size,
            )

    def build_encoder(self):
        input_question = layers.Input(
            shape=(None, self.vocab_size), name="EncoderInput"
        )  # input (batch_size, max_length, vocab_size)
        lin1 = layers.Dense(units=emb_hid_dim, name="Linear1")(
            input_question
        )  # lin1 (batch_size, max_length, emb_hid_dim)
        threshold = layers.ThresholdedReLU(theta=0.000001, name="Threshold")(
            lin1
        )  # threshold (batch_size, max_length, emb_hid_dim)
        lin2 = layers.Dense(units=emb_dim, name="Linear2")(
            threshold
        )  # lin2 (batch_size, max_length, emb_dim)
        lstm_output, state_h, state_c = layers.LSTM(
            units=encoder_rnn_units, name="LSTM", return_state=True
        )(
            lin2
        )  # state_h,state_c  (batch_size, encoder_output_dim)
        model = models.Model(
            inputs=[input_question],
            outputs=[lstm_output, state_h, state_c],
            name="Encoder",
        )
        return model

    def build_decoder(self):
        target_question = layers.Input(
            shape=(None,), name="TargetQuestion", dtype="int32"
        )  # target_question (batch_size)
        encoder_state_h = layers.Input(
            shape=(encoder_output_dim,), name="EncoderStateH"
        )  # encoder_state_h (batch_size, encoder_output_dim)

        encoder_state_c = layers.Input(
            shape=(encoder_output_dim,), name="EncoderStateC"
        )  # encoder_state_c (batch_size, encoder_output_dim)
        emb = layers.Embedding(
            self.vocab_size, emb_dim, name="DecoderEmbedding"
        )(
            target_question
        )  # emb (batch_size, max_length, emb_dim)
        lstm, _, _ = layers.LSTM(
            units=decoder_rnn_units,
            return_state=True,
            return_sequences=True,
            name="LSTM",
        )(
            emb, initial_state=[encoder_state_h, encoder_state_c]
        )  # lstm (batch_size, max_length, decoder_rnn_units)
        decoder_output = layers.Dense(self.vocab_size, activation="softmax")(
            lstm
        )  # decoder_output (batch_size, max_length, vocab_size)
        model = models.Model(
            inputs=[target_question, encoder_state_h, encoder_state_c],
            outputs=decoder_output,
            name="Decoder",
        )
        return model

    def _build_model(self):
        input_question = layers.Input(
            shape=(max_length,), name="InputQuestion", dtype="int32"
        )

        decoder_input_question = layers.Input(
            shape=(max_length,), name="DecoderInputQuestion", dtype="int32"
        )

        decoder_target_question = layers.Input(
            shape=(max_length,), name="DecoderTargetQuestion", dtype="int32"
        )

        onehot_input = layers.Lambda(
            K.one_hot,
            arguments={"num_classes": self.vocab_size},
            output_shape=(max_length, self.vocab_size),
            name="OneHot",
        )(input_question)

        onehot_target = layers.Lambda(
            K.one_hot,
            arguments={"num_classes": self.vocab_size},
            output_shape=(max_length, self.vocab_size),
            name="OneHotTarget",
        )(decoder_target_question)

        encoder = self.build_encoder()
        decoder = self.build_decoder()

        encoded_question, encoder_state_h, encoder_state_c = encoder(
            onehot_input
        )
        generated_question = decoder(
            [decoder_input_question, encoder_state_h, encoder_state_c]
        )

        discriminator_output, _, _ = encoder(generated_question)

        local_loss = K.mean(
            losses.categorical_crossentropy(onehot_target, generated_question)
        )
        global_loss = cosine_distance_batch(
            encoded_question, discriminator_output
        )

        loss = local_loss + global_loss

        model = models.Model(
            inputs=[
                input_question,
                decoder_input_question,
                decoder_target_question,
            ],
            outputs=generated_question,
            name="ParaphraseGenerationModel",
        )
        model.add_loss(loss)
        return model

    def fit(self):
        checkpoint = callbacks.ModelCheckpoint(
            "weights_{epoch:02d}-{val_loss:.2f}.hdf5",
            verbose=1,
            save_weights_only=True,
        )

        self.model.fit_generator(
            generator=self.train_generator,
            validation_data=self.val_generator,
            callbacks=[checkpoint],
            verbose=1,
            initial_epoch=0,
            epochs=100,
        )


def cosine_distance_batch(x, y):
    pos = K.sum(x * y, axis=-1)
    neg = K.dot(x, K.transpose(y))
    return K.sum(K.relu(neg - pos + 1)) / (batch_size * batch_size)


def joint_loss(true, model_output):
    target_question = true[:, 0]
    generated_question = model_output[:, 0]
    discriminator_output = model_output[:, 1]
    encoded_question = model_output[:, 2]
    target_question = model_output[:, 3]
    local_loss = losses.categorical_crossentropy(
        target_question, generated_question
    )
    global_loss = 1 - losses.cosine_proximity(
        encoded_question, discriminator_output
    )

    return local_loss
