import numpy as np
import scipy.misc as scm
import tensorflow as tf
import os

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
        self.dimensions = 1024 #self.filters * self.size**2 // 64
        self.checkpoint_path = "checkpoints"

        self.global_step = tf.Variable(0, name = 'global_step')

        # build model
        self.image_shape = [None, self.size, self.size, 3]
        
        print("lookup training data...")
        paths = glob("data/*.jpg")
        
        def tile_frame(path):
            image = tf.image.decode_image(tf.read_file(path), 3)
            return tf.data.Dataset.from_tensor_slices([
                tf.reshape(
                    image[y:y + self.size, x:x + self.size, :],
                    [self.size, self.size, 3]
                )
                for x in range(0, 1920 - self.size, self.size)
                for y in range(0, 1080 - self.size, self.size)
            ])
                    
        def tile_frames(path_a, path_b):
            return tf.data.Dataset.zip((
                tile_frame(path_a), tile_frame(path_b)
            ))
        
        d = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
        d = tf.data.Dataset.zip((d, d.skip(1)))
        d = d.shuffle(1000000)
        d = d.flat_map(tile_frames).shuffle(200).repeat()
        d = d.batch(self.batch_size)
        
        iterator = d.make_one_shot_iterator()
        self.real_a, self.real_b = iterator.get_next()
        
        float_real_a = self.preprocess(self.real_a)
        float_real_b = self.preprocess(self.real_b)

        self.mean_a, self.scale_a = self.encoder(float_real_a)
        self.mean_b, self.scale_b = self.encoder(float_real_b)
        
        def sample(mean, scale):
            return mean + scale * tf.random_normal(
                [self.batch_size, 8, 8, self.dimensions]
            )

        float_fake_a = self.decoder(sample(self.mean_a, self.scale_a))
        float_fake_b = self.decoder(sample(self.mean_b, self.scale_b))
        
        self.fake_a = self.postprocess(float_fake_a)
        self.fake_b = self.postprocess(float_fake_b)

        self.interpolated = self.postprocess(self.decoder(
            (self.mean_a + self.mean_b) * 0.5
        ))
        #self.interpolated = self.postprocess(self.decoder(self.mean_a))
        
        self.random = self.decoder(tf.random_normal(
            [self.batch_size, 8, 8, self.dimensions]
        ))

        # losses
        def difference(real, fake):
            return tf.reduce_mean(tf.norm(tf.abs(real - fake) + 1e-8, axis = -1))

        self.reconstruction_loss = \
            difference(float_real_a, float_fake_a) + \
            difference(float_real_b, float_fake_b)
            
        print("real_a", float_real_a.shape, "real_b", float_real_b.shape)

        def divergence(mean, scale):
            # from
            # https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.py
            return tf.reduce_mean(
                0.5 * tf.reduce_sum(
                    tf.square(mean) +
                    tf.square(scale) -
                    tf.log(1e-8 + tf.square(scale)) - 1,
                    1
                )
            )

        self.latent_loss = \
            divergence(self.mean_a, self.scale_a) + \
            divergence(self.mean_b, self.scale_b)

        self.motion_loss = difference(
            sample(self.mean_a, self.scale_a), 
            sample(self.mean_b, self.scale_b)
        )

        self.loss = \
            self.reconstruction_loss + \
            self.latent_loss * 1e-4 + \
            self.motion_loss# * 1e-2

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
            assert(x.shape[1] == 8 and x.shape[2] == 8)

            mean = x
            deviation = tf.nn.softplus(
                tf.layers.dense(x, self.dimensions, name = 'dense')
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
            assert(images.shape[1] == self.size and images.shape[2] == self.size)

            return images
 
    def train(self):
        step = 0
        
        while True:
            if step % 100 == 0:
                real_a, real_b, interpolated, fake_a, fake_b, rl, ll, ml = \
                    self.session.run([
                        self.postprocess(self.preprocess(self.real_a[:4, :, :, :])),
                        self.real_b[:4, :, :, :],
                        self.interpolated[:4, :, :, :], 
                        self.fake_a[:4, :, :, :], 
                        self.fake_b[:4, :, :, :],
                        self.reconstruction_loss, self.latent_loss, 
                        self.motion_loss
                    ])
                
                print(
                    "rl: {:.4f}, ll: {:.4f}, ml: {:.4f}"
                    .format(rl, ll, ml)
                )

                i = np.concatenate(
                    (
                        real_a[:4, :, :, :], 
                        fake_a, interpolated, fake_b,
                        real_b[:4, :, :, :]
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

        i = (i + 1.0) * 127.5
        i[i > 255] = 255.0
        i[i < 0] = 0.0
        scm.imsave("test/{}.jpg".format(step) , i)