#/bin/python
"""
Useful utilize

"""

class Parameter(object):
    def __init__(self, value, prior=None, prior_type=None):
        self.value = value
        self.prior = prior
        self.prior_type = prior_type
        if prior_type == 'gaussian':
            self.set_gaussian_prior(prior)
        elif prior_type == 'uniform':
            self.set_uniform_prior(prior)

    def set_gaussian_prior(self, mean=0, ):
        """
        Parameters
        ----------
        type : str
            The type of the prior, including "gaussian" and "uniform"

        """
        self.prior = 0

    def set_uniform_prior(self, range):
        self.prior = 0
