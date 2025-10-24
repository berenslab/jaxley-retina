import tensorflow as tf



class AdaptiveBinarizationLayer(tf.keras.layers.Layer):
    """Custom layer that binarizes input using a learnable threshold"""
    
    def __init__(self, temperature=1.0, **kwargs):
        super(AdaptiveBinarizationLayer, self).__init__(**kwargs)
        self.temperature = temperature
    
    def build(self, input_shape):
        # Create a learnable threshold parameter
        self.threshold = self.add_weight(
            name='threshold',
            shape=(),  # Scalar threshold
            initializer=tf.keras.initializers.Constant(0.5),  # Start at 0
            trainable=True
        )
        super(AdaptiveBinarizationLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Scale threshold by maximum input value
        max_input = tf.reduce_max(inputs)
        scaled_threshold = self.threshold * max_input
        
        # Use sigmoid for smooth approximation that has gradients
        smooth_binary = tf.sigmoid((inputs - scaled_threshold) / self.temperature)
        
        # Hard binarization for forward pass
        hard_binary = tf.cast(inputs > scaled_threshold, tf.float32)
        
        # Straight-through: forward uses hard binary, backward uses smooth gradients
        return smooth_binary + tf.stop_gradient(hard_binary - smooth_binary)
    
    def get_config(self):
        config = super(AdaptiveBinarizationLayer, self).get_config()
        config.update({'temperature': self.temperature})
        return config


class StaticBinarizationLayer(tf.keras.layers.Layer):
    """Custom layer that binarizes input using a learnable threshold"""
    
    def __init__(self, temperature=1.0, threshold_init=0, trainable_threshold=True, **kwargs):
        super(StaticBinarizationLayer, self).__init__(**kwargs)
        self.temperature = temperature
        self.threshold_init = threshold_init
        self.trainable_threshold = trainable_threshold
    
    def build(self, input_shape):
        # Create a learnable threshold parameter
        self.threshold = self.add_weight(
            name='threshold',
            shape=(),  # Scalar threshold
            initializer=tf.keras.initializers.Constant(self.threshold_init),
            trainable=self.trainable_threshold
        )
        super(StaticBinarizationLayer, self).build(input_shape)
    
    def call(self, inputs):        
        # Use sigmoid for smooth approximation that has gradients
        smooth_binary = tf.sigmoid((inputs - self.threshold) / self.temperature)
        
        # Hard binarization for forward pass
        hard_binary = tf.cast(inputs > self.threshold, tf.float32)
        
        # Straight-through: forward uses hard binary, backward uses smooth gradients
        return smooth_binary + tf.stop_gradient(hard_binary - smooth_binary)
    
    def get_config(self):
        config = super(StaticBinarizationLayer, self).get_config()
        config.update({'temperature': self.temperature})
        return config


def build_logistic_regression(input_shape, num_classes): 
    """Build a simple logistic regression classifier"""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def build_thresholding_logistic(input_shape, num_classes, adaptive=False, threshold_init=0, trainable_threshold=True):
    """Build the simple logistic regression with initial thresholding of the stim.
    NOTE: only the static binarization has the threshold init and trainability options rn.
    """
    if adaptive:
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            AdaptiveBinarizationLayer(), 
            tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            StaticBinarizationLayer(threshold_init=threshold_init, trainable_threshold=trainable_threshold), 
            tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)
        ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def build_normalizing_logistic(input_shape, num_classes):
    """Build a simple logistic regression classifier with layer norm"""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def build_2layer_mlp(input_shape, num_classes):
    """Build a 2-layer mlp baseline with a hidden layer of 128 units (arbitrary) and biases"""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        # tf.keras.layers.LayerNormalization(), # layer norm saves the day of course
        tf.keras.layers.Dense(
            128, 
            activation="sigmoid", 
            kernel_initializer='he_normal', 
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1.0, learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model