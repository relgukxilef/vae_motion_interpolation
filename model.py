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
        learning_rate = 1e-3, batch_size = 16
    ):
        self.session = session
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.continue_train = continue_train
        self.size = 256
        self.filters = 4
        self.dimensions = 2**11
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
        
        def sample(code):
            #return code[:, :, :, :self.dimensions]
            return (
                code[:, :, :, :self.dimensions] + 
                code[:, :, :, self.dimensions:] * tf.random_normal(
                    [self.batch_size, 8, 8, self.dimensions]
                )
            )

        code_0 = self.encoder(float_reals[0])
        code_2 = self.encoder(float_reals[2])
        float_fake = self.decoder(sample((code_0 + code_2) * 0.5))
        
        self.fake = self.postprocess(float_fake)
        
        self.random = self.postprocess(self.decoder(tf.random_normal(
            [self.batch_size, 8, 8, self.dimensions]
        )))

        # losses        
        def difference(real, fake):
            return tf.reduce_mean(tf.norm(tf.abs(real - fake) + 1e-8, axis = -1))

        self.reconstruction_loss = -tf.reduce_mean(
            self.discriminator(float_reals[0], float_fake, float_reals[2])
        ) #+ difference(float_fake, float_reals[1]) * 1e-2# * tf.maximum((1e3 - tf.cast(self.global_step, tf.float32)) / 1e3, 0.0)
        
        #self.reconstruction_loss = difference(float_fake, float_reals[1])
            
        def divergence(code):
            # from
            # https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py
            mean = code[:, :, :, :self.dimensions]
            deviation = code[:, :, :, self.dimensions:]
            return tf.reduce_mean(
                0.5 * tf.reduce_sum(
                    tf.square(mean) +
                    tf.square(deviation) -
                    tf.log(1e-8 + tf.square(deviation)) - 1,
                    3
                )
            )

        self.latent_loss = tf.reduce_mean(
            [divergence(code_0), divergence(code_2)]
        )

        self.g_loss = sum([
            self.reconstruction_loss,# * 1e-2,
            self.latent_loss * 1e-6
        ])
        
        real_score = self.discriminator(
            float_reals[0], float_reals[1], float_reals[2]
        )
        fake_score = self.discriminator(
            float_reals[0], float_fake, float_reals[2]
        )
        
        ratio = tf.random_uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        random_mix = float_reals[1] * (1 - ratio) + float_fake * ratio
        self.random_mix = self.postprocess(random_mix)
        
        gradients = tf.gradients(
            tf.reduce_mean(
                self.discriminator(float_reals[0], random_mix, float_reals[2])
            ), 
            random_mix
        )[0]
        
        gradient_penalty = tf.reduce_mean(
            (
                tf.sqrt(
                    tf.reduce_sum(
                        (tf.abs(gradients) + 1e-8) ** 2, axis=[1, 2, 3]
                    )
                ) - 1.0
            ) ** 2
        )
        
        self.gradient_penalty = gradient_penalty
        
        self.d_loss = (
            tf.reduce_mean(fake_score) - # * 1e-2 - 
            tf.reduce_mean(real_score) + # * 1e-2 + 
            gradient_penalty * 1e1
        )
        
        variables = tf.trainable_variables()
        g_variables = [v for v in variables if 'discriminator' not in v.name]
        d_variables = [v for v in variables if 'discriminator' in v.name]

        self.g_optimizer = tf.train.AdamOptimizer(
            self.learning_rate#, epsilon = 1e-2
        ).minimize(
            self.g_loss, self.global_step, var_list = g_variables
        )

        self.d_optimizer = tf.train.AdamOptimizer(
            self.learning_rate * 2#, epsilon = 1e-2
        ).minimize(
            self.d_loss, var_list = d_variables
        )

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

            mean = tf.layers.dense(x, self.dimensions, name = 'dense1')
            
            deviation = tf.nn.softplus(
                tf.layers.dense(x, self.dimensions, name = 'dense2') - 10.0
            ) + 1e-9

            return tf.concat([mean, deviation], 3)

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
            
            images = tf.layers.conv2d_transpose(
                x, 3, [4, 4], [2, 2], 'same', name = 'deconv1'
            )

            print(images.shape)
            #assert(images.shape[1] == self.size and images.shape[2] == self.size)

            return images
 
    def discriminator(self, a, b, c):
        with tf.variable_scope(
            'discriminator', reuse = tf.AUTO_REUSE
        ):
            def layer(x, out, name):
                return tf.nn.leaky_relu(tf.layers.max_pooling2d(
                    tf.layers.conv2d(
                        x, out, [4, 4], [1, 1], 'same', 
                        name = 'conv' + name
                    ), 2, 2, 'same'
                ))
                
            filters = 8
            x = tf.concat([a, b, c], 3)
            
            while x.shape[1] > 8:
                x = layer(x, filters, str(filters))
                filters *= 2
                
            x = tf.nn.leaky_relu(tf.reduce_max(x, [1, 2]))
                
            return tf.layers.dense(x, 1, name = 'dense')
    
    def train(self):
        step = 0
        
        while True:
            if step % 50 == 0:
                real_a, real_b, real_c, interpolated, \
                g_loss, d_loss, gradient_penalty, ll = \
                    self.session.run([
                        self.reals[0][:4, :, :, :],
                        self.reals[1][:4, :, :, :],
                        self.reals[2][:4, :, :, :],
                        self.fake[:4, :, :, :], 
                        self.g_loss, self.d_loss, self.gradient_penalty,
                        self.latent_loss
                    ])
                
                print(
                    "g_loss: {:.4f}, d_loss: {:.4f}, gp: {:.4f}, {:.4f}bits"
                    .format(g_loss, d_loss, gradient_penalty, ll)
                )

                i = np.concatenate(
                    (
                        real_a, 
                        real_b, interpolated,
                        real_c
                    ),
                    axis = 2
                )
                i = np.concatenate(
                    [np.squeeze(x, 0) for x in np.split(i, i.shape[0])]
                )

                scm.imsave("samples/{}.jpg".format(step) , i)
                
            for _ in tqdm(range(50)):
                for i in range(1):
                    self.session.run(
                        self.d_optimizer
                    )
                _, step = self.session.run(
                    [self.g_optimizer, self.global_step]
                )
                #for i in range(2):
                #    self.session.run(
                #        self.d_optimizer
                #    )
                
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
            width = (1920 // 32) * 32
            height = (1080 // 32) * 32
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