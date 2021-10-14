import time
from sklearn.metrics import mean_squared_error
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

from mpl_toolkits.mplot3d import Axes3D
%pylab inline
# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'
%config InlineBackend.figure_format = 'png'

# load data
q3_train = pd.read_csv("train.csv")
q3_test = pd.read_csv("test.csv")

# since 0 mean assumed
X_train = np.array(q3_train['x']).reshape(-1, 1)
y_train = q3_train['y']
mean_y = np.mean(y_train)
y_train = y_train - mean_y

X_test = np.array(q3_test['x']).reshape(-1, 1)
y_test = q3_test['y'] -mean_y

observation_index_points_ = X_train
observations_ = y_train

# Exponential quadratic
def build_gp(amplitude, length_scale, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_,
      observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp,
})
# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.

constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in 
                       [amplitude_var,
                       length_scale_var,
                       observation_noise_variance_var]]
                       
# Now we optimize the model parameters.
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    #tape.watch(trainable_variables)
    loss = -gp_joint_model.log_prob({
      'amplitude': amplitude_var,
      'length_scale': length_scale_var,
      'observation_noise_variance': observation_noise_variance_var,
      'observations': observations_
  })
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss

print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))
print('marginal log likelihood: {}'.format(lls_[-1]))

# plot to check convergence
plt.figure(figsize=(12, 4))
plt.plot(lls_)
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
plt.show()


predictive_index_points_ = np.linspace(-5, 5, 400, dtype=np.float64)
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)


expq_mean = gprm.mean()
expq_std = gprm.stddev()

plt.figure(figsize=(12, 6))
plt.scatter(observation_index_points_[:, 0], observations_,
            label='Observations')
plt.plot(predictive_index_points_, expq_mean, c='r',
           label='Posterior Sample')
plt.fill(np.concatenate([predictive_index_points_, predictive_index_points_[::-1]]),
         np.concatenate([expq_mean - 1.9600 * expq_std,
                        (expq_mean + 1.9600 * expq_std)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
plt.title('Square Exponential Kernel')
plt.legend(loc='upper right')
plt.show()

predictive_index_points_ = observation_index_points_
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)
print('train MSE: {}'.format(mean_squared_error(y_train, gprm.mean())))
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=X_test,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)
print('test MSE: {}'.format(mean_squared_error(y_test, gprm.mean())))

# Rational Quadratic
def build_gp(amplitude, length_scale, observation_noise_variance, scale_mixture_rate):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  kernel = tfk.RationalQuadratic(amplitude, length_scale, scale_mixture_rate)

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_,
      observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'scale_mixture_rate': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp,
})

# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.

constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

scale_mixture_rate_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='scale_mixture_rate',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in 
                       [amplitude_var,
                       length_scale_var,
                       observation_noise_variance_var,
                        scale_mixture_rate_var]]

# optimize the model parameters.
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    #tape.watch(trainable_variables)
    loss = -gp_joint_model.log_prob({
      'amplitude': amplitude_var,
      'length_scale': length_scale_var,
      'scale_mixture_rate': scale_mixture_rate_var,
      'observation_noise_variance': observation_noise_variance_var,
      'observations': observations_
  })
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss

print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('scale_mixture_rate: {}'.format(scale_mixture_rate_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))
print('marginal log likelihood: {}'.format(lls_[-1]))

predictive_index_points_ = np.linspace(-5, 5, 400, dtype=np.float64)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.RationalQuadratic(amplitude_var, length_scale_var, scale_mixture_rate_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

expq_mean = gprm.mean()
expq_std = gprm.stddev()

plt.figure(figsize=(12, 6))
plt.scatter(observation_index_points_[:, 0], observations_,
            label='Observations')
plt.plot(predictive_index_points_, expq_mean, c='r',
           label='Posterior Sample')
plt.fill(np.concatenate([predictive_index_points_, predictive_index_points_[::-1]]),
         np.concatenate([expq_mean - 1.9600 * expq_std,
                        (expq_mean + 1.9600 * expq_std)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
plt.title('Rational Quadratic Kernel')
plt.legend(loc='upper right')
plt.show()

# plot fit on test set
predictive_index_points_ = np.linspace(-5, 5, 400, dtype=np.float64)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.RationalQuadratic(amplitude_var, length_scale_var, scale_mixture_rate_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

expq_mean = gprm.mean()
expq_std = gprm.stddev()

plt.figure(figsize=(12, 6))
plt.scatter(observation_index_points_[:, 0], observations_,
            label='Observations')
plt.plot(predictive_index_points_, expq_mean, c='r',
           label='Posterior Sample')
plt.fill(np.concatenate([predictive_index_points_, predictive_index_points_[::-1]]),
         np.concatenate([expq_mean - 1.9600 * expq_std,
                        (expq_mean + 1.9600 * expq_std)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
plt.title('Rational Quadratic Kernel')
plt.legend(loc='upper right')
plt.show()

predictive_index_points_ = observation_index_points_
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)
print('train MSE: {}'.format(mean_squared_error(y_train, gprm.mean())))
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=X_test,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)
print('test MSE: {}'.format(mean_squared_error(y_test, gprm.mean())))

def build_gp(amplitude, length_scale, observation_noise_variance, period):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""

  # Create the covariance kernel, which will be shared between the prior (which we
  # use for maximum likelihood training) and the posterior (which we use for
  # posterior predictive sampling)
  kernel = tfk.ExpSinSquared(
    amplitude, length_scale, period)

  # Create the GP prior distribution, which we will use to train the model
  # parameters.
  return tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_,
      observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'period': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp,
})

# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.

constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

period_var = tfp.util.TransformedVariable(
    initial_value=10.,
    bijector=constrain_positive,
    name='period',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in 
                       [amplitude_var,
                       length_scale_var,
                       period_var,
                       observation_noise_variance_var]]
                       
# Now we optimize the model parameters.
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    #tape.watch(trainable_variables)
    loss = -gp_joint_model.log_prob({
      'amplitude': amplitude_var,
      'length_scale': length_scale_var,
      'period': period_var,
      'observation_noise_variance': observation_noise_variance_var,
      'observations': observations_
  })
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss
 
print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('period: {}'.format(period_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))
print('marginal log likelihood: {}'.format(lls_[-1]))

predictive_index_points_ = np.linspace(-5, 5, 400, dtype=np.float64)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.ExpSinSquared(amplitude_var, length_scale_var, period_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

expq_mean = gprm.mean()
expq_std = gprm.stddev()

plt.figure(figsize=(12, 6))
plt.scatter(observation_index_points_[:, 0], observations_,
            label='Observations')
plt.plot(predictive_index_points_, expq_mean, c='r',
           label='Posterior Sample')
plt.fill(np.concatenate([predictive_index_points_, predictive_index_points_[::-1]]),
         np.concatenate([expq_mean - 1.9600 * expq_std,
                        (expq_mean + 1.9600 * expq_std)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
plt.title('Square Exponential Kernel')
plt.legend(loc='upper right')
plt.show()

predictive_index_points_ = observation_index_points_
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)
print('train MSE: {}'.format(mean_squared_error(y_train, gprm.mean())))
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=X_test,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)
print('test MSE: {}'.format(mean_squared_error(y_test, gprm.mean())))


#### cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def cv(n_folds, X, y):
  j = 0
  mse_per = np.zeros(n_folds)
  mse_rbf = np.zeros(n_folds)
  mse_rq = np.zeros(n_folds)
 
  
  kf = KFold(n_splits=n_folds, shuffle=True)

  for train_index, test_index in kf.split(X, y):
    X_cv_train, X_cv_test = X[train_index,:], X[test_index,:]
    y_cv_train, y_cv_test = y[train_index], y[test_index]
    X_cv_train = np.array(X_cv_train).reshape(-1, 1)
    mean_y = np.mean(y_cv_train)
    y_cv_train = y_cv_train - mean_y

    X_cv_test = np.array(X_cv_test).reshape(-1, 1)
    y_cv_test = y_cv_test-mean_y

    # initialise kernels
    def build_gp(amplitude, length_scale, observation_noise_variance, period):
      """Defines the conditional dist. of GP outputs, given kernel parameters."""

      # Create the covariance kernel, which will be shared between the prior (which we
      # use for maximum likelihood training) and the posterior (which we use for
      # posterior predictive sampling)
      kernel = tfk.ExpSinSquared(
        amplitude, length_scale, period)

      # Create the GP prior distribution, which we will use to train the model
      # parameters.
      return tfd.GaussianProcess(
          kernel=kernel,
          index_points=X_cv_train,
          observation_noise_variance=observation_noise_variance)

    gp_joint_model = tfd.JointDistributionNamed({
        'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'period': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observations': build_gp,
    })

    # Create the trainable model parameters, which we'll subsequently optimize.
    # Note that we constrain them to be strictly positive.

    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    amplitude_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='amplitude',
        dtype=np.float64)

    length_scale_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='length_scale',
        dtype=np.float64)

    period_var = tfp.util.TransformedVariable(
        initial_value=10.,
        bijector=constrain_positive,
        name='period',
        dtype=np.float64)

    observation_noise_variance_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='observation_noise_variance_var',
        dtype=np.float64)

    trainable_variables = [v.trainable_variables[0] for v in 
                          [amplitude_var,
                          length_scale_var,
                          period_var,
                          observation_noise_variance_var]]
      
      # Now we optimize the model parameters.
    num_iters = 750
    optimizer = tf.optimizers.Adam(learning_rate=.01)

    # Store the likelihood values during training, so we can plot the progress
    lls_ = np.zeros(num_iters, np.float64)
    for i in range(num_iters):
      with tf.GradientTape() as tape:
        #tape.watch(trainable_variables)
        loss = -gp_joint_model.log_prob({
          'amplitude': amplitude_var,
          'length_scale': length_scale_var,
          'period': period_var,
          'observation_noise_variance': observation_noise_variance_var,
          'observations': y_cv_train
      })
      grads = tape.gradient(loss, trainable_variables)
      optimizer.apply_gradients(zip(grads, trainable_variables))
      lls_[i] = loss
    
    optimized_kernel = tfk.ExpSinSquared(amplitude_var, length_scale_var, period_var)
    gprm = tfd.GaussianProcessRegressionModel(
      kernel=optimized_kernel,
      index_points=X_cv_test,
      observation_index_points=X_cv_train,
      observations=y_cv_train,
      observation_noise_variance=observation_noise_variance_var,
      predictive_noise_variance=0.)
    mse_per[j] = mean_squared_error(y_cv_test, gprm.mean())
    
    
    #RQ kernel

    def build_gp(amplitude, length_scale, observation_noise_variance, scale_mixture_rate):
      """Defines the conditional dist. of GP outputs, given kernel parameters."""

      # Create the covariance kernel, which will be shared between the prior (which we
      # use for maximum likelihood training) and the posterior (which we use for
      # posterior predictive sampling)
      kernel = tfk.RationalQuadratic(amplitude, length_scale, scale_mixture_rate)

      # Create the GP prior distribution, which we will use to train the model
      # parameters.
      return tfd.GaussianProcess(
          kernel=kernel,
          index_points=X_cv_train,
          observation_noise_variance=observation_noise_variance)

    gp_joint_model = tfd.JointDistributionNamed({
        'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'scale_mixture_rate': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observations': build_gp,
    })

    # Create the trainable model parameters, which we'll subsequently optimize.
    # Note that we constrain them to be strictly positive.

    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    amplitude_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='amplitude',
        dtype=np.float64)

    length_scale_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='length_scale',
        dtype=np.float64)

    scale_mixture_rate_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='scale_mixture_rate',
        dtype=np.float64)

    observation_noise_variance_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='observation_noise_variance_var',
        dtype=np.float64)

    trainable_variables = [v.trainable_variables[0] for v in 
                          [amplitude_var,
                          length_scale_var,
                          observation_noise_variance_var,
                            scale_mixture_rate_var]]
    # Now we optimize the model parameters.
    num_iters = 750
    optimizer = tf.optimizers.Adam(learning_rate=.01)

  # Store the likelihood values during training, so we can plot the progress
    lls_ = np.zeros(num_iters, np.float64)
    for i in range(num_iters):
      with tf.GradientTape() as tape:
        #tape.watch(trainable_variables)
        loss = -gp_joint_model.log_prob({
          'amplitude': amplitude_var,
          'length_scale': length_scale_var,
          'scale_mixture_rate': scale_mixture_rate_var,
          'observation_noise_variance': observation_noise_variance_var,
          'observations': y_cv_train
      })
      grads = tape.gradient(loss, trainable_variables)
      optimizer.apply_gradients(zip(grads, trainable_variables))
      lls_[i] = loss

    optimized_kernel = tfk.RationalQuadratic(amplitude_var, length_scale_var, scale_mixture_rate_var)
    gprm = tfd.GaussianProcessRegressionModel(
      kernel=optimized_kernel,
      index_points=X_cv_test,
      observation_index_points=X_cv_train,
      observations=y_cv_train,
      observation_noise_variance=observation_noise_variance_var,
      predictive_noise_variance=0.)
    mse_rq[j] = mean_squared_error(y_cv_test, gprm.mean())
      
    def build_gp(amplitude, length_scale, observation_noise_variance):
      """Defines the conditional dist. of GP outputs, given kernel parameters."""

      # Create the covariance kernel, which will be shared between the prior (which we
      # use for maximum likelihood training) and the posterior (which we use for
      # posterior predictive sampling)
      kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

      # Create the GP prior distribution, which we will use to train the model
      # parameters.
      return tfd.GaussianProcess(
          kernel=kernel,
          index_points=X_cv_train,
          observation_noise_variance=observation_noise_variance)

    gp_joint_model = tfd.JointDistributionNamed({
        'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observations': build_gp,
    })
    

    # Create the trainable model parameters, which we'll subsequently optimize.
    # Note that we constrain them to be strictly positive.

    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    amplitude_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='amplitude',
        dtype=np.float64)

    length_scale_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='length_scale',
        dtype=np.float64)

    observation_noise_variance_var = tfp.util.TransformedVariable(
        initial_value=1.,
        bijector=constrain_positive,
        name='observation_noise_variance_var',
        dtype=np.float64)

    trainable_variables = [v.trainable_variables[0] for v in 
                          [amplitude_var,
                          length_scale_var,
                          observation_noise_variance_var]]
      
    # Now we optimize the model parameters.
    num_iters = 750
    optimizer = tf.optimizers.Adam(learning_rate=.01)

    # Store the likelihood values during training, so we can plot the progress
    lls_ = np.zeros(num_iters, np.float64)
    for i in range(num_iters):
      with tf.GradientTape() as tape:
        #tape.watch(trainable_variables)
        loss = -gp_joint_model.log_prob({
          'amplitude': amplitude_var,
          'length_scale': length_scale_var,
          'observation_noise_variance': observation_noise_variance_var,
          'observations': y_cv_train
      })
      grads = tape.gradient(loss, trainable_variables)
      optimizer.apply_gradients(zip(grads, trainable_variables))
      lls_[i] = loss
    optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)

    gprm = tfd.GaussianProcessRegressionModel(
      kernel=optimized_kernel,
      index_points=X_cv_test,
      observation_index_points=X_cv_train,
      observations=y_cv_train,
      observation_noise_variance=observation_noise_variance_var,
      predictive_noise_variance=0.)
    mse_rbf[j] = mean_squared_error(y_cv_test, gprm.mean())
    
    j += 1
  # average all test mse's
  av_per = np.mean(mse_per)
  av_rbf = np.mean(mse_rbf)
  av_rq = np.mean(mse_rq)
  return (av_per, av_rbf, av_rq)

# run cross validation
av_cv = cv(4, X_train, y_train)
