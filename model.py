import numpy as np
import scipy.misc as scm
import tensorflow as tf
import os

from glob import glob
from tqdm import tqdm

class VAEMotionInterpolation:
    def __init__(
        self, session, continue_train = True, 
        learning_rate = 2e-3, batch_size = 8
    ):
        self.session = session
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.continue_train = continue_train
        self.size = 256
        self.filters = 4
        self.dimensions = 1024 #self.filters * self.size**2 // 64
        self.checkpoint_path = "checkpoints"

        self.global_step = tf.Variable(1, name = 'global_step')

        # build model
        self.image_shape = [None, self.size, self.size, 3]

        self.real_a = tf.placeholder(
            tf.int32, self.image_shape, name = 'real_a'
        )
        self.real_b = tf.placeholder(
            tf.int32, self.image_shape, name = 'real_b'
        )
        
        float_real_a = self.preprocess(self.real_a)
        float_real_b = self.preprocess(self.real_b)

        self.mean_a, self.scale_a = self.encoder(float_real_a)
        self.mean_b, self.scale_b = self.encoder(float_real_b)
        
        def sample(mean, scale):
            return mean + scale * tf.random_normal(tf.shape(mean))

        float_fake_a = self.decoder(sample(self.mean_a, self.scale_a))
        float_fake_b = self.decoder(sample(self.mean_b, self.scale_b))
        
        self.fake_a = self.postprocess(float_fake_a)
        self.fake_b = self.postprocess(float_fake_b)

        self.interpolated = self.decoder(
            (self.mean_a + self.mean_b) * 0.5
        )
        self.interpolated = self.decoder(self.mean_a)
        self.random = self.decoder(tf.random_normal(
            [self.batch_size, 1, 1, self.dimensions]
        ))

        # losses
        def difference(real, fake):
            return tf.reduce_mean(tf.abs(real - fake))

        self.reconstruction_loss = \
            difference(float_real_a, float_fake_a) + \
            difference(float_real_b, float_fake_b)

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
            self.latent_loss * 1e-3 #+ \
            #self.motion_loss

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
        
    def postprocess(self, images):
        return tf.cast((images + 1.0) * 127.5, tf.int32)

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
            
            x = images

            x = tf.layers.separable_conv2d(
                x, filters * 4 * 2, [4, 4], [2, 2], 'same', name = 'conv1'
            )

            for i in range(2, 9):
                x = layer(x, min(filters * 4**i, self.dimensions) * 2, str(i))

            print(x.shape)
            assert(x.shape[1] == 1 and x.shape[2] == 1)

            mean = x[:, :, :, :self.dimensions]
            deviation = tf.nn.softplus(x[:, :, :, self.dimensions:]) + 1e-8

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

            for i in reversed(range(2, 9)):
                x = layer(x, min(filters * 4**i, self.dimensions), str(i))

            images = tf.nn.sigmoid(tf.layers.conv2d_transpose(
                x, 3, [4, 4], [2, 2], 'same', name = 'deconv1'
            )) * 2.0 - 1.0

            print(images.shape)
            assert(images.shape[1] == self.size and images.shape[2] == self.size)

            return images

    def train(self):
        print("lookup training data...")
        frames = glob("data/*.jpg")

        pairs = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]

        np.random.shuffle(pairs)

        for batch in tqdm(np.array_split(pairs, len(pairs) // self.batch_size)):
            path_a = [p[0] for p in batch]
            path_b = [p[1] for p in batch]
        
            # select random crop
            x = np.random.randint(0, 1920 - self.size)
            y = np.random.randint(0, 1080 - self.size)

            def load(paths):
                images = []
                for path in paths:
                    i = scm.imread(path)
                    if i.shape != (1080, 1920, 3):
                        print("wrong shape: " + str(i.shape))
                    i = i[y:y + self.size, x:x + self.size, :]
                    images += [i]
                return np.array(images)

            image_a = load(path_a)
            image_b = load(path_b)

            _, step = self.session.run(
                [self.optimizer, self.global_step],
                {
                    self.real_a: image_a,
                    self.real_b: image_b
                }
            )

            if step % 500 == 0:
                print("saving iteration " + str(step))
                self.saver.save(
                    self.session,
                    self.checkpoint_path + "/vaemi",
                    global_step=step
                )

            if step % 10 == 0:
                interpolated, rl, ll, ml = self.session.run(
                    [
                        self.interpolated, 
                        self.reconstruction_loss, self.latent_loss, 
                        self.motion_loss
                    ],
                    {
                        self.real_a: image_a[:4, :, :, :],
                        self.real_b: image_b[:4, :, :, :]
                    }
                )
                
                print(
                    "rl: {:.4f}, ll: {:.4f}, ml: {:.4f}"
                    .format(rl, ll, ml)
                )

                i = np.concatenate(
                    (image_a[:4, :, :, :], interpolated, image_b[:4, :, :, :]),
                    axis = 2
                )
                i = np.concatenate(
                    [np.squeeze(x, 0) for x in np.split(i, i.shape[0])]
                )

                scm.imsave("samples/{}.jpg".format(step) , i)

    def test(self):
        r, step = self.session.run(
            [self.random, self.global_step],
            {}
        )

        i = np.concatenate(
            [np.squeeze(x, 0) for x in np.split(r, r.shape[0])]
        )

        i = (i + 1.0) * 127.5
        i[i > 255] = 255.0
        i[i < 0] = 0.0
        scm.imsave("test/{}.jpg".format(step) , i)