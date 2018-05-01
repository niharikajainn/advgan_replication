import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

batch_size = 100
image_size = 28
image_as_vector = image_size * image_size  # 784
num_classes = 10  # 0-9

# Original instance for D and G
x = tf.placeholder(tf.float32, [None, 784])
input_tensor = tf.reshape(x, shape=[-1, 28, 28, 1])

# for training target model
num_outputs = 10
t = [0,0,0,0,0,0,0,0,1,0] # target class
labels = []
for i in range (batch_size):
    labels.append(t)
labels = np.array(labels)

c = 3 #User-defined bound for hinge loss


theta_G = []
theta_T = []


def discriminator(x):

    with tf.variable_scope("D", reuse=tf.AUTO_REUSE):

        # C8, C16, C32, FC

        # Input x is a one-dimensional tensor for the image
        input = x

        # First: C8 layer. Does not need instance normalization
        # 4 x 4, filters 8, stride 2, leaky ReLU
        conv_1 = tf.contrib.layers.conv2d(input, 8, 4, stride=2, activation_fn=tf.nn.leaky_relu, scope="C8")

        # Second: C16 layer.
        # 4 x 4, filters 16, stride 2, instance_norm, leaky ReLU
        conv_2 = tf.contrib.layers.conv2d(conv_1, 16, 4, stride=2, activation_fn=None, scope="C16")
        conv_2 = tf.contrib.layers.instance_norm(conv_2, activation_fn=tf.nn.leaky_relu)

        # Third: C32 layer.
        # 4 x 4, filters 32, stride 2, instance_norm, leaky ReLU
        conv_3 = tf.contrib.layers.conv2d(conv_2, 32, 4, stride=2, activation_fn=None, scope="C32")
        conv_3 = tf.contrib.layers.instance_norm(conv_3, activation_fn=tf.nn.leaky_relu)

        # Final: FC layer. 1-dimensional output
        # Decided to use Softmax as the activation function
        output = tf.contrib.layers.fully_connected(conv_3, batch_size, activation_fn=tf.nn.softmax, scope="FC")

    # Output is a 128x1 tensor
    return output

def generator(z):

    with tf.variable_scope("G", reuse=tf.AUTO_REUSE):

        # c3s1-8, d16, d32, r32, r32, r32, r32, u16, u8, c3s1-3

        # input is some one-dimensional vector
        input = z

        # First: c3s1-8 layer.
        # 3 x 3, filters 8, stride 1, instance_norm, ReLU
        conv_1 = tf.contrib.layers.conv2d(input, 8, 3, stride=1, padding = 'same', activation_fn=None, variables_collections = theta_G)
        conv_1 = tf.contrib.layers.instance_norm(conv_1, activation_fn=tf.nn.relu, variables_collections = theta_G)
        print "c3s1-8:" + str(conv_1)

        # Second: d16 layer.
        # 3 x 3, filters 16, stride 2, instance_norm, ReLU
        conv_2 = tf.contrib.layers.conv2d(conv_1, 16, 3, stride=2, padding= 'same', activation_fn=None, variables_collections = theta_G)
        conv_2 = tf.contrib.layers.instance_norm(conv_2, activation_fn=tf.nn.relu, variables_collections = theta_G)
        print "d16:" + str(conv_2)

        # Third: d32 layer.
        # 3 x 3, filters 32, stride 1, instance_norm, ReLU
        conv_3 = tf.contrib.layers.conv2d(conv_2, 32, 3, stride=2, padding='same', activation_fn=None, variables_collections = theta_G)
        conv_3 = tf.contrib.layers.instance_norm(conv_3, activation_fn=tf.nn.relu, variables_collections = theta_G)
        print "d32:" + str(conv_3)

        residual_block_input = conv_3

        # Fourth: 4 r32 layers
        # Each residual block has 3x3 conv, batch_norm, ReLU, 3x3 conv, batch norm
        #      where the two conv layers have the same number of filters
        for i in range(4):
            residual_1 = tf.contrib.layers.conv2d(residual_block_input, 32, 8, stride=1, padding='same', activation_fn=None, variables_collections = theta_G)
            residual_1 = tf.contrib.layers.batch_norm(residual_1, activation_fn=tf.nn.relu, variables_collections = theta_G)
            residual_2 = tf.contrib.layers.conv2d(residual_1, 32, 8, stride=1, padding='same', activation_fn=None, variables_collections = theta_G)
            residual_2 = tf.contrib.layers.batch_norm(residual_2, activation_fn=None, variables_collections = theta_G)
            residual_block_input = residual_2
            print "r32:" + str(residual_block_input)

        residual_blocks = residual_block_input

        # Fifth: u16 layer.
        # 3 x 3, filters 16, stride 0.5, instance_norm, ReLU
        conv_4 = tf.contrib.layers.conv2d_transpose(residual_blocks, 16, 3, stride=2, padding='valid', activation_fn=None, variables_collections = theta_G)
        conv_4 = tf.contrib.layers.instance_norm(conv_4, activation_fn=tf.nn.relu, variables_collections = theta_G)
        print "u16:" + str(conv_4)

        # Sixth: u8 layer.
        # 3 x 3, filters 8, stride 0.5, instance_norm, ReLU
        conv_5 = tf.contrib.layers.conv2d_transpose(conv_4, 8, 3, stride=2, padding='valid', activation_fn=None, variables_collections = theta_G)
        conv_5 = tf.contrib.layers.instance_norm(conv_5, activation_fn=tf.nn.relu, variables_collections = theta_G)
        print "u8:" + str(conv_5)

        # Final: c3s1-3
        ## Outputs changed from 3 to 1 for dimension match
        ## Kernel size changed from 3 to 4 for dimension match
        output = tf.contrib.layers.conv2d(conv_5, 1, 4, stride=1, padding='valid', activation_fn=None, variables_collections = theta_G)
        output = tf.contrib.layers.instance_norm(output, activation_fn=tf.nn.relu, variables_collections = theta_G)
        print "c3s1-3:" + str(output)

    return output

train_input = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
y_true_class = tf.placeholder(tf.int64, shape=[None])

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

"""def target(x): #MNIST Model C
    with tf.variable_scope("T", reuse=tf.AUTO_REUSE):
        weights = {
                   'W_conv1a': tf.Variable(tf.random_normal([3,3,1,32])),
                   'W_conv1b': tf.Variable(tf.random_normal([3,3,32,32])),
                   'W_conv2a': tf.Variable(tf.random_normal([3,3,32,64])),
                   'W_conv2b': tf.Variable(tf.random_normal([3,3,64,64])),
                   'W_fc1': tf.Variable(tf.random_normal([7*7*64, 200])),
                   'W_fc2': tf.Variable(tf.random_normal([200, 200])),
                   'out': tf.Variable(tf.random_normal([200, num_outputs]))
                   }
        biases = {
                   'b_conv1a': tf.Variable(tf.random_normal([32])),
                   'b_conv1b': tf.Variable(tf.random_normal([32])),
                   'b_conv2a': tf.Variable(tf.random_normal([64])),
                   'b_conv2b': tf.Variable(tf.random_normal([64])),
                   'b_fc1': tf.Variable(tf.random_normal([200])),
                   'b_fc2': tf.Variable(tf.random_normal([200])),
                   'out': tf.Variable(tf.random_normal([num_outputs]))
                   }

        x = tf.reshape(x, shape=[-1, 28, 28, 1]) # Input Layer
        conv1a = tf.nn.relu(conv(x, weights['W_conv1a']) + biases['b_conv1a']) # Conv(32,3,3)+Relu
        print "conv1a: " + str(conv1a)
        conv1b = tf.nn.relu(conv(conv1a, weights['W_conv1b']) + biases['b_conv1b']) # Conv(32,3,3,)+Relu
        print "conv1b: " + str(conv1b)
        conv1 = maxpool(conv1b) # MaxPooling(2,2)
        print "conv1: " + str(conv1)
        conv2a = tf.nn.relu(conv(conv1, weights['W_conv2a']) + biases['b_conv2a']) # Conv(64,3,3)+Relu
        print "conv2a: " + str(conv2a)
        conv2b = tf.nn.relu(conv(conv2a, weights['W_conv2b']) + biases['b_conv2b']) # Conv(64,3,3)+Relu
        print "conv2b: " + str(conv2b)
        conv2 = maxpool(conv2b) # MaxPooling(2,2)
        print "conv2: " + str(conv2)
        fc1 = tf.reshape(conv2, [-1, 7*7*64])
        fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1']) + biases['b_fc1']) # FC(200)+Relu
        print "fc1: " + str(fc1)
        #fc2 = tf.reshape(fc1, 200)
        fc2 = tf.nn.softmax(tf.matmul(fc1, weights['W_fc2']) + biases['b_fc2']) # FC(200)+Softmax
        print "fc2: " + str(fc2)

        logits = tf.matmul(fc2, weights['out']) + biases['out'] # Output Layer
        print "logits: " + str(logits)
        y_predicted = tf.nn.softmax(logits)

    return logits, y_predicted

def train_target():
    logits, y_predicted = target(train_input)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    y_predicted_class = tf.argmax(y_predicted, axis=1)
    correctness = tf.equal(y_predicted_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))

    num_epochs = 2
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        loss_over_all_data = 0
        for _ in range(int(mnist.train.num_examples/batch_size)):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, cost], feed_dict={train_input: x_batch, y_true: y_batch})
            loss_over_all_data += loss_batch
        print "\nEpoch " + str(epoch + 1)
        print "Loss: " + str(round(loss_over_all_data,2))

    test_labels = mnist.test.labels
    test_labels_class = np.argmax(test_labels, axis=1)
    test_images = mnist.test.images

    accuracy_test, correctness_test = sess.run([accuracy, correctness], feed_dict={train_input:test_images, y_true_class:test_labels_class})
    print "\nTotal Accuracy: " + str(round(accuracy_test * 100,2)) + "%"
    """


def target(x):
    with tf.variable_scope("T", reuse=tf.AUTO_REUSE):
        weights = {
                    'W_layer1': tf.Variable(tf.random_normal([image_as_vector, 600])),
                    'W_layer2': tf.Variable(tf.random_normal([600, 500])),
                    'W_layer3': tf.Variable(tf.random_normal([500, 400])),
                    'logits': tf.Variable(tf.random_normal([400, num_classes]))
                  }
        biases = {
                    'b_layer1': tf.Variable(tf.random_normal([600])),
                    'b_layer2': tf.Variable(tf.random_normal([500])),
                    'b_layer3': tf.Variable(tf.random_normal([400])),
                    'logits': tf.Variable(tf.random_normal([num_classes]))
                 }
        x = tf.reshape(x, shape=[-1,784])
        layer1 = tf.matmul(x, weights['W_layer1']) + biases['b_layer1']
        layer1 = tf.nn.relu(layer1)
        layer2 = tf.matmul(layer1, weights['W_layer2']) + biases['b_layer2']
        layer2 = tf.nn.relu(layer2)
        layer3 = tf.matmul(layer2, weights['W_layer3']) + biases['b_layer3']
        layer3 = tf.nn.relu(layer3)
        logits = tf.matmul(layer3, weights['logits']) + biases['logits']

        y_predicted = tf.nn.softmax(logits)
    return logits, y_predicted

train_input = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
y_true_class = tf.placeholder(tf.int64, shape=[None])

def train_target():
    logits, y_predicted = target(train_input)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    y_predicted_class = tf.argmax(y_predicted, axis=1)
    correctness = tf.equal(y_predicted_class, y_true_class) # returns vector of type bool
    accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    for epoch in range(5):
        loss_over_all_data = 0
        for i in range(len(mnist.train.labels)/batch_size):
            x_next_batch, y_next_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run(
                [optimizer, loss],
                feed_dict={train_input: x_next_batch, y_true: y_next_batch})
            loss_over_all_data += loss_batch
        print "\nEpoch " + str(epoch + 1)
        print "Loss: " + str(round(loss_over_all_data,2))

    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_labels_class = np.argmax(test_labels, axis=1)

    accuracy_test, correctness_test, y_predicted_class_test = sess.run(
        [accuracy, correctness, y_predicted_class],
        feed_dict={train_input: test_images, y_true_class: test_labels_class})

    print "\nTotal Accuracy: " + str(round(accuracy_test * 100,2)) + "%"

    return correctness_test, y_predicted_class_test, test_labels_class

train_target()




# Notes for probabilities:
#      log(D(x)): probability of marking original instance as "real"
#      log(D(G(z))): probability of marking perturbed instance as "real"
#      log(1-D(G(z))): probability of marking perturbed instance as "fake"

D_real = discriminator(input_tensor)

# The generator G takes the original instance x as its input
#      and generates a perturbation G(x)
perturbation = generator(input_tensor)

# Then x + G(x) will be sent to the discriminator D
D_fake = discriminator(input_tensor + perturbation)

theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")
# Train D to maximize the (averages of) log(D(x)) and log(1-D(G(z)))
D_loss = -1. * tf.reduce_mean(tf.reduce_mean(tf.log(D_real)) + tf.reduce_mean(tf.log(1. - D_fake)))

theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")
# to train G, better to maximize log(D(G(Z)) than minimize (1-logD(G(z)))
G_loss = -1. * tf.reduce_mean(tf.log(D_fake))


D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = theta_G)

loss_GAN = D_loss + G_loss

t_adv = target(input_tensor + perturbation)[0]

theta_T = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="T")
loss_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=t_adv, labels=labels))/10000
t_solver = tf.train.AdamOptimizer().minimize(loss_adv, var_list = theta_T)


#loss_hinge = max(0, tf.reduce_max(perturbation) - c)

loss = loss_adv + loss_GAN #+ loss_hinge


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for it in range(1000000):
    theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D")
    theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="G")
    theta_T = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="T")
    X_batch, _ = mnist.train.next_batch(batch_size)

    #_, _, _, loss_GAN_curr, loss_adv_curr, loss_hinge_curr, loss_curr = sess.run([G_solver, D_solver, t_solver, loss_GAN, loss_adv, loss_hinge, loss], feed_dict={X: X_batch})
    _, _, _, loss_GAN_curr, loss_adv_curr, loss_curr = sess.run([G_solver, D_solver, t_solver, loss_GAN, loss_adv, loss], feed_dict={x: X_batch})

    if it % 1000 == 0:
        print 'Iter: {}'.format(it)
        print 'GAN loss: {:.4}'. format(loss_GAN_curr)
        print 'Target loss: {:.4}'.format(loss_adv_curr)
        #print('Hinge loss: {:.4}'.format(loss_hinge_curr))
        print 'Total loss: {:.4}'.format(loss_curr)
        print '\n'
