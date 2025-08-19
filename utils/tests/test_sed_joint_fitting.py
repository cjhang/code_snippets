"""
This is demo how to fit SED combined with resolved and unresolved data

To make the example simple, we will focus only on the FIR dust SED fitting
"""

import os, sys
sys.path.append('/Users/jhchen/Documents/projects/code_snippets/utils')

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

from model_utils import Parameter, Grid1D, Models, MBlackbody, MBlackbody_log, MinimizeFitter, EmceeFitter

# define the free parameters
# assume three source resolved by ALMA
# temp = Parameter(35, limits=[20,80], name='uniform_temp')
t1 = Parameter(40, limits=[20,80])
t2 = Parameter(35, limits=[20,80])
t3 = Parameter(45, limits=[20,80])
beta = Parameter(2.0, limits=[1, 3], name='uniform_beta')
z0 = 3.0
z = z0 #Parameter(z0, limits=[1, 5], name='common_z')
# Mdust1 = Parameter(2e8, limits=[1e7, 1e9]) # Msun
# Mdust2 = Parameter(1e8, limits=[1e7, 1e9])
# Mdust3 = Parameter(3e8, limits=[1e7, 1e9])
area = 1 #kpc2

Mdust1 = Parameter(7, limits=[6, 10]) # Msun
Mdust2 = Parameter(8.1, limits=[6, 10])
Mdust3 = Parameter(8.2, limits=[6, 10])
#
# define the model
mbb1 = MBlackbody_log(temperature=t1, z=z, beta=beta,
                  dustmass=Mdust1, area=area,
                  name='mbb1')
mbb2 = MBlackbody_log(temperature=t2, z=z, beta=beta,
                  dustmass=Mdust2, area=area,
                  name='mbb2')
mbb3 = MBlackbody_log(temperature=t3, z=z, beta=beta,
                  dustmass=Mdust3, area=area,
                  name='mbb3')
#
# joint model
joint_model = Models([mbb1, mbb2, mbb3])
print('parameters', joint_model.parameters)

#
# define the customized fitter


def log_probability(theta, models, grid, data):
    models.update_parameters(theta)
    params_priors = models.get_priors()
    # check the priors first to save sometime if it becomes inf
    if not np.isfinite(params_priors):
        return -np.inf
    # customized way of reading the data
    var_Herschel = data['var_Herschel']
    var_ALMA = data['var_ALMA'] 
    # data['obs_ALMA'] = [obs1,obs2,obs3]
    obs_ALMA = data['obs_ALMA'] 
    obs_ALMA_err = data['obs_ALMA_err']
    obs_ALMA_err_log = obs_ALMA_err / obs_ALMA / np.log(10)
    obs_Herschel = data['obs_Herschel'] 
    obs_Herschel_err = data['obs_Herschel_err'] 
    obs_Herschel_err_log = obs_Herschel_err / obs_Herschel / np.log(10)

    # calculate the log_likelihood
    model_log_likelihood = 0
    # calculate the ALMA bands, for each source
    for i,model_i in enumerate(models.models):
        model_alma_i = model_i(var_ALMA)
        # model_log_likelihood += -0.5*np.sum((obs_ALMA[i]-model_alma_i)**2/obs_ALMA_err[i]**2)
        model_log_likelihood += -0.5*np.sum((np.log10(obs_ALMA[i])-np.log10(model_alma_i))**2/obs_ALMA_err_log[i]**2)
    # calculate the Hearchel/PACS bands, for total
    model_Herschel = models.create_model(var_Herschel)
    # model_log_likelihood = -0.5*np.sum((obs_Herschel-model_Herschel)**2/obs_Herschel_err**2)
    model_log_likelihood += -0.5*np.sum((np.log10(obs_Herschel)-np.log10(model_Herschel))**2/obs_Herschel_err_log**2)
    # print('log_probability:', model_log_likelihood+params_priors)
    # print('theta=',theta)
    return model_log_likelihood + params_priors

class Joint_EmceeFitter(EmceeFitter):
    def __init__(self, models, grid, data, **kwargs):
        super().__init__(models=models, grid=grid, data=data, 
                         **kwargs)
        self.log_probability = log_probability

class Joint_MinimizerFitter(MinimizeFitter):
    def __init__(self, models, grid, data):
        super().__init__(models=models, grid=grid, data=data)
    def cost(self, theta):
        return np.sum(self.vector_cost(theta))
    def vector_cost(self, theta):
        self.models.update_parameters(theta)
        params_priors = self.models.get_priors()
        data = self.data
        # customized way of reading the data
        var_Herschel = data['var_Herschel']
        var_ALMA = data['var_ALMA'] 
        # data['obs_ALMA'] = [obs1,obs2,obs3]
        obs_ALMA = data['obs_ALMA'] 
        obs_ALMA_err = data['obs_ALMA_err']
        obs_ALMA_err_log = obs_ALMA_err / obs_ALMA / np.log(10)
        obs_Herschel = data['obs_Herschel'] 
        obs_Herschel_err = data['obs_Herschel_err'] 
        obs_Herschel_err_log = obs_Herschel_err / obs_Herschel / np.log(10)

        # allocate the array for cost
        ALMA_cost = np.zeros_like(obs_ALMA)
        Herschel_cost = np.zeros_like(obs_Herschel)
        # calculate the ALMA bands, for each source
        for i,model_i in enumerate(self.models.models):
            model_alma_i = model_i(var_ALMA)
            ALMA_cost[i] = (np.log10(obs_ALMA[i])-np.log10(model_alma_i))**2/obs_ALMA_err_log[i]**2
        # calculate the Hearchel/PACS bands, for total
        model_Herschel = self.models.create_model(var_Herschel)
        Herschel_cost = (np.log10(obs_Herschel)-np.log10(model_Herschel))**2/obs_Herschel_err_log**2
        total_cost = np.hstack([Herschel_cost.ravel(), ALMA_cost.ravel()])
        return total_cost + params_priors

#
# generate mock data
#
# wavelength constructed similar to Herschel/PACS and ALMA (band 3+6+7)
wave_obs = np.array([250, 350, 500, 870, 1200, 3000])
wave_rest = np.array([250, 350, 500, 870, 1200, 3000]) / (1.+z0)
freq = (const.c / (wave_rest*u.um)).to(u.GHz).value
freq_Herschel = freq[:3]
freq_ALMA = freq[3:]
grid1d = Grid1D()
grid1d.x = freq
model_data = joint_model.create_model(freq)

# setup the resolved ALMA observation
flux1 = mbb1(freq_ALMA) * (1+0.11*np.random.randn(3))
flux2 = mbb2(freq_ALMA) * (1+0.11*np.random.randn(3))
flux3 = mbb3(freq_ALMA) * (1+0.11*np.random.randn(3))
obs_ALMA = np.array([flux1, flux2, flux3])
obs_ALMA_err = 0.1*obs_ALMA
print('flux1 shape', flux1.shape)
print('obs_ALMA.shape', obs_ALMA.shape)

# setup the unresolved Herschel observation
flux_Herschel = joint_model.create_model(freq_Herschel)
obs_Herschel = flux_Herschel * (1+0.1*np.random.randn(3))
print('obs_Herschel.shape', obs_Herschel.shape)
obs_Herschel_err = 0.11*flux_Herschel

#model_err = np.random.randn(model_data.size) * model_data / model_data
#y_data = model_data + model_err
#y_data_err = np.abs(y_err)
data = {'var_ALMA':freq_ALMA, 'var_Herschel':freq_Herschel, 
        'obs_ALMA': obs_ALMA, 'obs_ALMA_err':obs_ALMA_err,
        'obs_Herschel': obs_Herschel, 
        'obs_Herschel_err': obs_Herschel_err}

truths = np.array(list(joint_model.parameters.values()))
initial_guess = truths * (1 + 0.1*np.random.randn(len(truths)))
joint_model.update_parameters(initial_guess)
print('truths:', truths)
print('initial_guess', initial_guess)

#
# start the fitting, mcmc
if __name__ == '__main__':
    if 0:
        fitter_mcmc = Joint_EmceeFitter(joint_model, grid1d, data, 
                                        )#backend='test1.h5',)
        steps = 40000
        # fitter_mcmc.auto_run(progress=True, max_steps=steps,)
        fitter_mcmc.multi_run(progress=True, steps=steps, ncores=6)

        samples = fitter_mcmc.samples(discard=int(steps*0.2), thin=1, flat=True)
        fitter_mcmc.plot()
        fitter_mcmc.corner_plot(truths=truths)
        best_fit_mcmc = np.percentile(samples, 50, axis=0)
        best_fit_mcmc_low = np.percentile(samples, 16, axis=0)
        best_fit_mcmc_up = np.percentile(samples, 84, axis=0)
        best_fit_mcmc_err = list(zip(best_fit_mcmc_low, best_fit_mcmc_up))
        # print("Noiseless truths:", initial_guess)
        # print('Best fit minimize:', best_fit_minimize)
        print(joint_model.parameters.keys())
        print('Best fit mcmc:', best_fit_mcmc)
        best_fit_params = dict(zip(joint_model.parameters.keys(), best_fit_mcmc))
        print(best_fit_params)
     
    #
    # start the fitting, minimize
    if 1:
        fitter_minimize = Joint_MinimizerFitter(joint_model, grid1d, data)
        # fitter_minimize.run(initial_guess=initial_guess, debug=1)
        fitter_minimize.auto_run(initial_guess=initial_guess, )#debug=1, maxiter=100000)
        best_fit_params = dict(zip(joint_model.parameters.keys(), fitter_minimize.best_fit))
        print(joint_model.parameters.keys())
        print('best_fit:', best_fit_params.values())
        print('best_fit_err:', fitter_minimize.best_fit_error)

    if True:
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.errorbar(freq_Herschel, obs_Herschel, yerr=obs_Herschel_err, linestyle='none', marker='o', label='total (model)')
        # show three models
        ax.errorbar(freq_ALMA, obs_ALMA[0], yerr=obs_ALMA_err[0], linestyle='none', 
                    marker='*', label='t1')
        ax.errorbar(freq_ALMA, obs_ALMA[1], yerr=obs_ALMA_err[1], linestyle='none', 
                    marker='s', label='t2')
        ax.errorbar(freq_ALMA, obs_ALMA[2], yerr=obs_ALMA_err[2], linestyle='none', 
                    marker='^', label='t3')

        # ge the best for individual model
        joint_model.update_parameters(best_fit_params)
        best_fit1 = joint_model.models[0](freq)
        best_fit2 = joint_model.models[1](freq)
        best_fit3 = joint_model.models[2](freq)
        best_fit_total = joint_model.create_model(freq)
        # show the best fit
        ax.plot(freq, best_fit1, label='best_fit1')
        ax.plot(freq, best_fit2, label='best_fit2')
        ax.plot(freq, best_fit3, label='best_fit3')
        ax.plot(freq, best_fit_total, label='best_fit_total')
        ax.legend()
        plt.show()
