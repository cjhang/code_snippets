import numpy as np
import matplotlib.pyplot as plt

###############################################
# a more complex implementation, but more flexible

class Parameter(object):
    def __init__(self, value=None, limits=None, name=''):
        self.value = value
        self.limits = limits
        self.name = name
        self.is_fixed = False
    def get_log_prior(self, value):
        if self.limits is None:
            return 0
        limits = self.limits
        if limits[0] < value < limits[1]:
            return 0
        else:
            return -np.inf

class Model:
    def __init__(self, name=None):
        self._name = name
    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__
        else:
            return self._name
    @name.setter
    def name(self, name):
        self._name = name

class LineModel(Model):
    def __init__(self, m=None, b=None, name=None):
        super().__init__(name=name)
        self.m = m #Parameter(m)
        self.b = b #Parameter(b)
    def evaluate(self, x):
        return self.m.value * x + self.b.value

class ScaleUncertainty(Model):
    def __init__(self, factor=None, name=None):
        super().__init__(name=name)
        self.factor = factor #Parameter(factor)
    def evaluate(self, x):
        return 0

class Models:
    """
    If a model compose of several same Models, they should have different name
    """
    def __init__(self, models):
        self.models = models
    def get_parameters(self):
        params_list = {}
        for model in self.models:
            for key,value in model.__dict__.items():
                if not isinstance(value, Parameter):
                    continue
                if not value.is_fixed:
                    params_list[model.name+'.'+key] = value
        return params_list
    def update_parameters(self, params_dict):
        for model in self.models:
            for key in params_dict.keys():
                model_name, model_param_name = key.split('.')
                if model_param_name in model.__dict__.keys():
                    model.__dict__[model_param_name].value = params_dict[key]
    def get_priors(self, params_dict):
        log_priors = 0
        for model in self.models:
            for key in params_dict.keys():
                model_name, model_param_name = key.split('.')
                if model_name != model.name:
                    continue
                if model_param_name in model.__dict__.keys():
                    log_priors += model.__dict__[model_param_name].get_log_prior(params_dict[key])
        return log_priors
    def create_model(self, var):
        # parameter_names = self.get_parameters()
        # self.update_parameters(dict(zip(parameter_names, theta)))
        model_value = 0.
        for model in self.models:
            model_value += model.evaluate(var)
        return model_value

def mcmc_log_probability(theta, models, var, data, data_err):
    """
    args: [var, data, data_err]
    """
    # var, data, data_err = args
    model_parameters = models.get_parameters()
    # for key, value in model_parameters.items():
        # print(key, value.value)
    dict_theta = dict(zip(model_parameters.keys(), theta))
    models.update_parameters(dict_theta)
    params_priors = models.get_priors(dict_theta)
    if not np.isfinite(params_priors):
        return -np.inf 
    model = models.create_model(var)
    for m in models.models:
        if isinstance(m, ScaleUncertainty):
            log_f = m.factor.value
            err2_addition = model**2*np.exp(2*log_f)
        else:
            err2_addition = 0
    sigma2 = data_err**2 + err2_addition
    # model_log_likelihood = -0.5*np.sum((data-model)**2)/data_err**2 + np.log(data_err**2)
    model_log_likelihood = -0.5*np.sum((data-model)**2/sigma2 + np.log(sigma2))
    return model_log_likelihood + params_priors



if __name__ == '__main__':
    # prepare the data
    m_true = -0.9594
    b_true = 4.294
    f_true = 0.534
    N = 50
    x = np.sort(10 * np.random.rand(N))
    yerr = 0.1 + 0.5 * np.random.rand(N) # with 0.5 variation
    y = m_true * x + b_true
    y += np.abs(f_true * y) * np.random.randn(N)
    y += yerr * np.random.randn(N)

    # fit in a wrapper way
    # line_model = LineModel(m=Parameter(m_true+1e-4*np.random.randn()), 
                           # b=Parameter(b_true+1e-4*np.random.randn()))
    line_model = LineModel(m=Parameter(m_true, limits=[-5.,0.5]), 
                           b=Parameter(b_true, limits=[0., 10.]))
    scale_uncertainty_model = ScaleUncertainty(factor=Parameter(f_true, limits=[-10.,1.0]))
    models = Models([line_model,scale_uncertainty_model])
    # models = Models([line_model,])
    paramters = models.get_parameters()

    import emcee

    pos = np.array([m_true, b_true, f_true]) + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, mcmc_log_probability, args=(models, x, y, yerr)
    )
    sampler.run_mcmc(pos, 5000, progress=True)


    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["m", "b", "log(f)"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.show()


