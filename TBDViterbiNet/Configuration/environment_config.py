""" Setting the image dimensions and bounding region """
import torch

# Global parameters ( determine the image size)
num_range_bins = 200
num_doppler_bins = 64

# Possible range for our simulation
r_min = 0.0
r_max = 2985.0

# Possible velocities in our simulation
vr_min = -369.1406
vr_max = 369.1406

range_bounds = (r_min, r_max)
doppler_bound = (vr_min, vr_max)

environment = {'image_size': (num_range_bins, num_doppler_bins),
               'range_bounds': range_bounds,
               'doppler_bound': doppler_bound}



class Environment:
    """
        Represents the radar environment with range and doppler information.

        Args:
            environment (dict): A dictionary containing information about the radar environment.
                - 'image_size' (tuple): A tuple (num_range_bins, num_doppler_bins) specifying the image size.
                - 'range_bounds' (tuple): A tuple (min_range, max_range) specifying the range bounds.
                - 'doppler_bounds' (tuple): A tuple (min_doppler, max_doppler) specifying the doppler bounds.
        """

    def __init__(self, environment: dict):
        self.num_range_bins, self.num_doppler_bins = environment['image_size'][0], environment['image_size'][1]
        # Create a range vector and a doppler vector with specified bounds and number of bins.
        self.range_vec = torch.linspace(environment['range_bounds'][0], environment['range_bounds'][1],
                                        self.num_range_bins)
        self.doppler_vec = torch.linspace(environment['doppler_bound'][0], environment['doppler_bound'][1],
                                          self.num_doppler_bins)

    def val2idx(self, range_value: float, doppler_value: float) -> tuple:
        """
        Convert range and doppler values to their corresponding indices in the environment grid.

        Args:
            range_value (float): The range value to be converted to an index.
            doppler_value (float): The doppler value to be converted to an index.

        Returns:
            tuple: A tuple (range_idx, doppler_idx) representing the indices in the environment grid.
        """
        range_idx = torch.argmin(torch.abs(self.range_vec - range_value))
        doppler_idx = torch.argmin(torch.abs(self.doppler_vec - doppler_value))
        return int(range_idx), int(doppler_idx)

    def idx2val(self, range_idx:int , doppler_idx:int) -> tuple:
        """
                Convert range and doppler indices to their corresponding values in the environment grid.

                Args:
                    range_idx (int): The index of the range value.
                    doppler_idx (int): The index of the doppler value.

                Returns:
                    tuple: A tuple (range_value, doppler_value) representing the values in the environment grid.
                """
        range_value = self.range_vec[range_idx]
        doppler_value = self.doppler_vec[doppler_idx]
        return range_value, doppler_value
