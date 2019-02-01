import tensorflow as tf
from ops import *

def CAN_loss(model):
    
    # builds optimizers and losses
    model.G = model.generator(model, model.z)
    model.D, model.D_logits, model.D_c, model.D_c_logits = model.discriminator(
        model,
        model.inputs,
        reuse=False)

    if model.experience_flag:
        try:
            # Select a random subset of 16 elements
            random_elements = random.sample(model.experience_buffer, 16)
            experience_selection = tf.convert_to_tensor(random_elements)
        except ValueError:
            experience_selection = tf.convert_to_tensor(model.experience_buffer)
        
        # Combine the experience selection 
        model.G = tf.concat([model.G, experience_selection], axis=0)

    model.D_, model.D_logits_, model.D_c_, model.D_c_logits_ = model.discriminator(model, model.G, reuse=True)
    model.d_sum = histogram_summary("d", model.D)
    model.d__sum = histogram_summary("d_", model.D_)
    model.d_c_sum = histogram_summary("d_c", model.D_c)
    model.d_c__sum = histogram_summary("d_c_", model.D_c_)
    model.G_sum = image_summary("G", model.G)

    max_y = tf.argmax(input=model.y, axis=1)
    max_d_c = tf.argmax(input=model.D_c, axis=1)
    correct_prediction = tf.equal(x=max_y, y=max_d_c)

    prediction_to_float = tf.cast(x=correct_prediction, dtype=tf.float32)
    model.accuracy = tf.reduce_mean(input_tensor=prediction_to_float)

    true_label = tf.random_uniform(
        shape=tf.shape(model.D),
        minval=.8,
        maxval=1.2
    )

    false_label = tf.random_uniform(
        shape=tf.shape(model.D_),
        minval=0.0,
        maxval=0.3
    )

    tensor = sigmoid_cross_entropy_with_logits(
        x=model.D_logits,
        y=true_label * tf.ones_like(model.D)
    )
    model.d_loss_real = tf.reduce_mean(input_tensor=tensor)

    tensor = sigmoid_cross_entropy_with_logits(
        x=model.D_logits_,
        y=false_label * tf.ones_like(model.D_)
    )
    model.d_loss_fake = tf.reduce_mean(input_tensor=tensor)

    tensor = tf.nn.softmax_cross_entropy_with_logits(logits=model.D_c_logits, labels=model.smoothing * model.y)
    model.d_loss_class_real = tf.reduce_mean(input_tensor=tensor)

    # if classifier is set, then use the classifier,
    # otherwise use the clasification layers in the discriminator
    ls = ( 1.0 / model.y_dim ) * tf.ones_like(model.D_c_)
    if model.style_net_checkpoint is None:
        tensor = tf.nn.softmax_cross_entropy_with_logits(logits=model.D_c_logits_, ls=ls)
    else:
        model.classifier = model.make_style_net(model.G)
        tensor = tf.nn.softmax_cross_entropy_with_logits(logits=model.classifier, labels=ls)
    
    model.g_loss_class_fake = tf.reduce_mean(input_tensor=tensor)

    model.g_loss_fake = -tf.reduce_mean(tf.log(model.D_))

    model.d_loss = model.d_loss_real + model.d_loss_class_real + model.d_loss_fake
    model.g_loss = model.g_loss_fake + model.lamb * model.g_loss_class_fake

    model.d_loss_real_sum = scalar_summary("d_loss_real", model.d_loss_real)
    model.d_loss_fake_sum = scalar_summary("d_loss_fake", model.d_loss_fake)
    model.d_loss_class_real_sum = scalar_summary("d_loss_class_real", model.d_loss_class_real)
    model.g_loss_class_fake_sum = scalar_summary("g_loss_class_fake", model.g_loss_class_fake)
    model.g_loss_sum = scalar_summary("g_loss", model.g_loss)
    model.d_loss_sum = scalar_summary("d_loss", model.d_loss)
    model.d_sum = merge_summary([
        model.z_sum,
        model.d_sum,
        model.d_loss_real_sum,
        model.d_loss_sum,
        model.d_loss_class_real_sum,
        model.g_loss_class_fake_sum
    ])
    model.g_sum = merge_summary([
        model.z_sum,
        model.d__sum,
        model.G_sum,
        model.d_loss_fake_sum,
        model.g_loss_sum
    ])

    adam_args = {
        "learning_rate": model.learning_rate,
        "beta1": 0.5
    }
    model.g_opt = tf.train.AdamOptimizer(**adam_args)
    model.d_opt = tf.train.AdamOptimizer(**adam_args)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    d_update = model.d_opt.minimize(model.d_loss, var_list=d_vars)
    g_update = model.g_opt.minimize(model.g_loss, var_list=g_vars)

    return d_update, g_update, [model.d_loss, model.g_loss], [model.d_sum, model.g_sum]

def WCAN_loss(model):
    pass


def GAN_loss(model):
    model.G = model.generator(model.z, model.y)
    model.D, model.D_logits = model.discriminator(model.inputs, model.y, reuse=False)
    model.D_, model.D_logits_ = model.discriminator(model.G, model.y, reuse=True)

    true_label = tf.random_uniform(tf.shape(model.D), .8, 1.2)
    false_label = tf.random_uniform(tf.shape(model.D_), 0.0, 0.3)

    model.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(model.D_logits, true_label * tf.ones_like(model.D)))

    model.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(model.D_logits_, false_label * tf.ones_like(model.D_)))

    model.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(model.D_logits_, tf.ones_like(model.D_)))
    model.d_loss = model.d_loss_real + model.d_loss_fake

    model.d_sum = histogram_summary("d", model.D)
    model.d__sum = histogram_summary("d_", model.D_)
    model.G_sum = image_summary("G", model.G)

    model.g_loss_sum = scalar_summary("g_loss", model.g_loss)
    model.d_loss_sum = scalar_summary("d_loss", model.d_loss)
    model.d_loss_real_sum = scalar_summary("d_loss_real", model.d_loss_real)
    model.d_loss_fake_sum = scalar_summary("d_loss_fake", model.d_loss_fake)
    model.d_sum = merge_summary([model.z_sum, model.d_sum, model.d_loss_real_sum, model.d_loss_sum])
    model.g_sum = merge_summary([model.z_sum, model.d__sum, model.G_sum, model.d_loss_fake_sum, model.g_loss_sum])

    model.g_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)
    model.d_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]
    d_update = model.d_opt.minimize(model.d_loss, var_list=d_vars)
    g_update = model.g_opt.minimize(model.g_loss, var_list=g_vars)

    return d_update, g_update, [model.d_loss, model.g_loss], [model.d_sum, model.g_sum]

def WGAN_loss(model):
    model.g_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)
    model.d_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)

    model.G = model.generator(model, model.z, model.y)
    model.D_real = model.discriminator(model, model.inputs, model.y, reuse=False)
    model.D_fake = model.discriminator(model, model.G, model.y, reuse=True)
    model.g_loss = -tf.reduce_mean(model.D_fake)
    model.wp= -tf.reduce_mean(model.D_fake) + tf.reduce_mean(model.D_real)

    epsilon = tf.random_uniform(
        shape= [model.batch_size, 1,1,1],
        minval=0.,
        maxval=1.
    )
    x_hat = model.inputs + epsilon * (model.G - model.inputs)
    D_x_hat = model.discriminator(model, x_hat, model.y,reuse=True)
    grad_D_x_hat = tf.gradients(D_x_hat, [x_hat])[0]
    model.slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_x_hat), reduction_indices=[1,2,3]))
    model.gradient_penalty = tf.reduce_mean((model.slopes - 1.) ** 2)
    model.d_loss = -model.wp + 10 * model.gradient_penalty

    t_vars = tf.trainable_variables()
    model.d_vars = [var for var in t_vars if 'd_' in var.name]
    model.g_vars = [var for var in t_vars if 'g_' in var.name]

    g_update = model.g_opt.minimize(model.g_loss, var_list=model.g_vars)
    d_update = model.d_opt.minimize(model.d_loss, var_list=model.d_vars)

    loss_ops = [model.d_loss, model.g_loss]

    model.G_sum = image_summary("G", model.G)
    model.g_loss_sum = scalar_summary("g_loss", model.g_loss)
    model.d_loss_sum = scalar_summary("d_loss", model.d_loss)
    model.wp_sum = scalar_summary("wasserstein_penalty", model.wp)
    model.gp_sum = scalar_summary("gradient_penalty", model.gradient_penalty)

    model.d_sum = merge_summary([model.d_loss_sum, model.wp_sum, model.gp_sum])
    model.g_sum = merge_summary([model.g_loss_sum, model.G_sum])
    
    return d_update, g_update, loss_ops, [model.d_sum, model.g_sum]
