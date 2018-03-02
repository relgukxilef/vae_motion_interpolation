import numpy as np
import scipy.misc as scm
import tensorflow as tf
import os

from math import floor
from glob import glob
from tqdm import tqdm

class VAEMotionInterpolation:
    def __init__(
        self, session, continue_train = True, 
        learning_rate = 1e-3, batch_size = 32
    ):
        self.session = session
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.continue_train = continue_train
        self.size = 256
        self.filters = 4
        self.dimensions = 2**11 #self.filters * self.size**2 // 64
        self.checkpoint_path = "checkpoints"

        self.global_step = tf.Variable(0, name = 'global_step')

        # build model
        self.image_shape = [None, self.size, self.size, 3]
        
        print("lookup training data...")
        self.paths = glob("data/*.jpg")
        
        def tile_frame(path):
            image = tf.image.decode_image(tf.read_file(path), 3)
            return tf.data.Dataset.from_tensor_slices([
                tf.reshape(
                    image[y:y + self.size, x:x + self.size, :],
                    [self.size, self.size, 3]
                )
                for x in range(0, 1718 - self.size, self.size)
                for y in range(0, 720 - self.size, self.size)
            ])
                    
        def tile_frames(a, b, c):
            return tf.data.Dataset.zip(
                (tile_frame(a), tile_frame(b), tile_frame(c))
            )
        
        d = tf.data.Dataset.from_tensor_slices(tf.constant(self.paths))
        d = tf.data.Dataset.zip((d, d.skip(1), d.skip(2)))
        d = d.shuffle(1000000)
        d = d.flat_map(tile_frames).shuffle(200).repeat()
        d = d.batch(self.batch_size)
        
        iterator = d.make_one_shot_iterator()
        self.reals = iterator.get_next()
        
        float_reals = [self.preprocess(r) for r in self.reals]
        
        self.codes = [self.encoder(r) for r in float_reals]

        def sample(code):
            return code[0] + code[1] * tf.random_normal(
                [self.batch_size, 8, 8, self.dimensions]
            )

        float_fakes = [self.decoder(sample(c)) for c in self.codes]
        
        self.fakes = [self.postprocess(f) for f in float_fakes]

        self.center = (sample(self.codes[0]) + sample(self.codes[2])) * 0.5
        float_interpolated = self.decoder(self.center)
        self.interpolated = self.postprocess(float_interpolated)
        
        self.random = self.postprocess(self.decoder(tf.random_normal(
            [self.batch_size, 8, 8, self.dimensions]
        )))

        # losses
        def difference(real, fake):
            return tf.reduce_mean(tf.norm(tf.abs(real - fake) + 1e-8, axis = -1))

        self.reconstruction_loss = tf.reduce_mean(
            [difference(r, f) for r, f in zip(float_reals, float_fakes)]
        )
            
        def divergence(code):
            # from
            # https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py
            return tf.reduce_mean(
                0.5 * tf.reduce_sum(
                    tf.square(code[0]) +
                    tf.square(code[1]) -
                    tf.log(1e-8 + tf.square(code[1])) - 1,
                    3
                )
            )

        self.latent_loss = tf.reduce_mean(
            [divergence(c) for c in self.codes]
        )

        self.motion_loss = difference(
            float_reals[1], 
            float_interpolated
        )
        
        self.blend_loss = difference(
            float_reals[1],
            (float_fakes[0] + float_fakes[2]) * 0.5
        )

        self.loss = sum([
            self.reconstruction_loss,
            self.latent_loss * 1e-6,
            self.motion_loss
        ])

        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate
        ).minimize(self.loss, self.global_step)

        self.saver = tf.train.Saver()

        self.session.run(tf.global_variables_initializer())

        # load checkpoint
        if self.continue_train:
            print(" [*] Reading checkpoint...")

            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(
                    self.session,
                    self.checkpoint_path + "/" + os.path.basename(
                        checkpoint.model_checkpoint_path
                    )
                )
                print(" [*] before training, Load SUCCESS ")

            else:
                print(" [!] before training, failed to load ")
        else:
            print(" [!] before training, no need to load ")
            
    def preprocess(self, images):
        return tf.cast(images, tf.float32) / 127.5 - 1.0
        #return tf.cast(images, tf.float32) / 255.0
        
    def postprocess(self, images):
        return tf.cast(tf.minimum(tf.maximum(
            (images + 1.0) * 127.5, 0
        ), 255), tf.int32)

    def encoder(self, images):
        with tf.variable_scope(
            'encoder', reuse = tf.AUTO_REUSE
        ):
            def layer(x, out, name):
                return tf.layers.separable_conv2d(
                    tf.nn.elu(x), 
                    out, [4, 4], [2, 2], 'same', name = 'conv' + name
                )

            filters = self.filters
            
            x = tf.pad(
                images, [[0, 0], [0, 0], [0, 0], [0, 1]], constant_values = 1.0
            )
            print(x.shape)

            x = tf.layers.separable_conv2d(
                x, filters * 4 * 2, [4, 4], [2, 2], 'same', name = 'conv1'
            )

            for i in range(2, 6):
                x = layer(x, min(filters * 4**i, self.dimensions), str(i))

            print(x.shape)
            #assert(x.shape[1] == 8 and x.shape[2] == 8)

            mean = x
            deviation = tf.nn.softplus(
                tf.layers.dense(x, self.dimensions, name = 'dense') - 10.0
            ) + 1e-9

            return mean, deviation

    def decoder(self, feature):
        with tf.variable_scope(
            'decoder', reuse = tf.AUTO_REUSE
        ):
            def layer(x, out, name):
                return tf.nn.elu(tf.layers.conv2d_transpose(
                    x, out, [4, 4], [2, 2], 'same', name = 'deconv' + name
                ))

            filters = self.filters

            x = feature

            for i in reversed(range(2, 6)):
                x = layer(x, min(filters * 4**(i - 1), self.dimensions), str(i))
                
            assert(x.shape[3] == filters * 4)

            images = tf.nn.sigmoid(tf.layers.conv2d_transpose(
                x, 3, [4, 4], [2, 2], 'same', name = 'deconv1'
            )) * 2.0 - 1.0

            print(images.shape)
            #assert(images.shape[1] == self.size and images.shape[2] == self.size)

            return images
 
    def train(self):
        step = 0
        
        while True:
            if step % 100 == 0:
                real_a, real_b, real_c, interpolated, \
                fake_a, fake_c, rl, ll, ml, bl = \
                    self.session.run([
                        self.reals[0][:4, :, :, :],
                        self.reals[1][:4, :, :, :],
                        self.reals[2][:4, :, :, :],
                        self.interpolated[:4, :, :, :], 
                        self.fakes[0][:4, :, :, :], 
                        self.fakes[2][:4, :, :, :],
                        self.reconstruction_loss, self.latent_loss, 
                        self.motion_loss, self.blend_loss
                    ])
                
                print(
                    "rl: {:.4f}, ll: {:.4f}, ml: {:.4f}, bl: {:.4f}"
                    .format(rl, ll, ml, bl)
                )

                i = np.concatenate(
                    (
                        real_a, 
                        fake_a, interpolated, real_b, fake_c,
                        real_c
                    ),
                    axis = 2
                )
                i = np.concatenate(
                    [np.squeeze(x, 0) for x in np.split(i, i.shape[0])]
                )

                scm.imsave("samples/{}.jpg".format(step) , i)
                
            for _ in tqdm(range(100)):
                _, step = self.session.run(
                    [self.optimizer, self.global_step]
                )
                
            if step % 500 == 0:
                print("saving iteration " + str(step))
                self.saver.save(
                    self.session,
                    self.checkpoint_path + "/vaemi",
                    global_step=step
                )

    def test(self):
        r, step = self.session.run(
            [self.random, self.global_step]
        )

        i = np.concatenate(
            [np.squeeze(x, 0) for x in np.split(r, r.shape[0])]
        )

        scm.imsave("test/{}.jpg".format(step) , i)
        
    def interpolate_video(self, start, end, factor):
        befores = []
        afters = []
        ratios = []
        
        for f in range(round(start * factor), round(end * factor)):
            p = f / factor
            befores += [self.paths[floor(p)]]
            afters += [self.paths[floor(p) + 1]]
            ratios += [p - floor(p)]
        
        d = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(befores),
            tf.data.Dataset.from_tensor_slices(afters),
            tf.data.Dataset.from_tensor_slices(ratios),
            tf.data.Dataset.from_tensor_slices(
                ["out/{:06d}.jpg".format(i) for i in range(0, len(ratios))]
            )
        ))
        
        def load_frame(path):
            image = self.preprocess(
                tf.image.decode_image(tf.read_file(path), 3)
            )
            width = (1718 // 32) * 32
            height = (720 // 32) * 32
            return tf.reshape(image[:height, :width, :], [1, height, width, 3])
            
        def load_frames(a, b, r, i):
            return load_frame(a), load_frame(b), r, i
        
        d = d.map(load_frames)
        
        a, b, r, i = d.make_one_shot_iterator().get_next()
        
        print(a.shape)
        
        interpolate_frame = tf.write_file(
            i,
            tf.image.encode_jpeg(
                tf.cast(self.postprocess(self.decoder(
                    self.encoder(a)[0] * (1 - r) +
                    self.encoder(b)[0] * r
                ))[0, :, :, :], tf.uint8),
                quality = 100
            )
        )
        
        for i in tqdm(range(len(ratios))):
            self.session.run(interpolate_frame)