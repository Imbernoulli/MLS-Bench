import numpy as np
import copy

'''
    Scheduler for diffusion sampling following EDM framework.
    schedule (\sigma(t)): linear, sqrt, vp
    timestep (discretization of t): log, poly-n, vp
    scaling: none, vp

    Example:
    VP: Scheduler(num_steps=1000, schedule='vp', timestep='vp', scaling='vp')
    VE: Scheduler(num_steps=1000, schedule='sqrt', timestep='log', scaling='none')
    EDM: Scheduler(num_steps=200, schedule='linear', timestep='poly-7', scaling='none')
    
    Example Usage: See DiffusionSampler in utils/diffusion.py for unconditional diffusion sampling.
'''
class Scheduler:
    """
        Scheduler for diffusion sigma(t) and discretization step size Delta t
    """

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, schedule='linear',
                 timestep='poly-7', scaling='none'):
        """
            Initializes the scheduler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the schedule.
                sigma_max (float): Maximum value of sigma.
                sigma_min (float): Minimum value of sigma.
                sigma_final (float): Final value of sigma, defaults to sigma_min.
                schedule (str): Type of schedule for sigma ('linear' or 'sqrt').
                timestep (str): Type of timestep function ('log' or 'poly-n').
                scaling (str): Type of scaling function ('none' or 'vp').
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final
        if self.sigma_final is None:
            self.sigma_final = self.sigma_min
        self.schedule = schedule
        self.timestep = timestep

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)
        scaling_fn, scaling_derivative_fn = self.get_scaling_fn(scaling)
        if self.schedule == 'vp':
            self.sigma_max = sigma_fn(1) * scaling_fn(1)
            
        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])
        scaling_steps = np.array([scaling_fn(t) for t in time_steps])
        # scaling_factor = 1 - \dot s(t)/s(t) * \Delta t
        scaling_factor = np.array(
            [1 -  scaling_derivative_fn(time_steps[i]) / scaling_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        # factor = 2 s(t)^2 \dot\sigma(t)\sigma(t)\Delta t
        factor_steps = np.array(
            [2 * scaling_fn(time_steps[i])**2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        self.sigma_steps, self.time_steps, self.factor_steps, self.scaling_factor, self.scaling_steps = sigma_steps, time_steps, factor_steps, scaling_factor, scaling_steps
        self.factor_steps = [max(f, 0) for f in self.factor_steps]

    def get_sigma_fn(self, schedule):
        """
            Returns the sigma function, its derivative, and its inverse based on the given schedule.
        """
        if schedule == 'sqrt':
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / 2 / np.sqrt(t)
            sigma_inv_fn = lambda sigma: sigma ** 2

        elif schedule == 'linear':
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        
        elif schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            sigma_fn = lambda t: np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t) - 1)
            sigma_derivative_fn = lambda t: (beta_d * t + beta_min)*np.exp(beta_d * t**2/2 + beta_min * t) / 2 / sigma_fn(t)
            sigma_inv_fn = lambda sigma: np.sqrt(beta_min**2 + 2*beta_d*np.log(sigma**2 + 1))/beta_d - beta_min/beta_d

        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_scaling_fn(self, schedule):
        if schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            scaling_fn = lambda t: 1/ np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t))
            scaling_derivative_fn = lambda t: - (beta_d * t + beta_min)/ 2 / np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t))
        else:
            scaling_fn = lambda t: 1
            scaling_derivative_fn = lambda t: 0
        return scaling_fn, scaling_derivative_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        """
            Returns the time step function based on the given timestep type.
        """
        if timestep == 'log':
            get_time_step_fn = lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r
        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            get_time_step_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        elif timestep == 'vp':
            get_time_step_fn = lambda r: 1 - r * (1 - 1e-3)
        else:
            raise NotImplementedError
        return get_time_step_fn

    @classmethod
    def get_partial_scheduler(cls, scheduler, new_sigma_max):
        """
            Generates a new scheduler with the given sigma_max value.
        """
        new_scheduler = copy.deepcopy(scheduler)
        num_steps = sum([s < new_sigma_max for s in scheduler.sigma_steps]) + 1

        new_scheduler.num_steps = num_steps - 1
        new_scheduler.sigma_max = new_sigma_max
        new_scheduler.sigma_steps = scheduler.sigma_steps[-num_steps:]
        new_scheduler.time_steps = scheduler.time_steps[-num_steps:]
        new_scheduler.factor_steps = scheduler.factor_steps[-num_steps:]
        new_scheduler.scaling_factor = scheduler.scaling_factor[-num_steps:]
        new_scheduler.scaling_steps = scheduler.scaling_steps[-num_steps:]
        return new_scheduler