from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def build_sgd_optimizer(initial_learning_rate=0.001,
                        decay_steps=1000,
                        decay_rate=0.9,
                        momentum=0.0,
                        nesterov=False,
                        weight_decay=1e-7):
    # setup schedule
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    optimizer = SGD(learning_rate=learning_rate_schedule,
                    momentum=momentum,
                    nesterov=nesterov,
                    weight_decay=weight_decay)  # Adding weight decay here

    return optimizer


def build_adam_optimizer(initial_learning_rate=0.001,
                         decay_steps=1000,
                         decay_rate=0.9,
                         epsilon=1e-7,
                         weight_decay=1e-7):
    # Setup schedule
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    # Initialize Adam optimizer with the learning rate schedule
    optimizer = Adam(learning_rate=learning_rate_schedule, epsilon=epsilon)

    return optimizer


def build_sgd_optimizer_wo_schedule(initial_learning_rate=0.001,
                                    momentum=0.0, nesterov=False):
    # setup schedule
    optimizer = SGD(learning_rate=initial_learning_rate,
                    momentum=momentum, nesterov=nesterov)
    return optimizer
