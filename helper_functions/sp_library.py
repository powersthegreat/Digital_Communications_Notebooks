import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as intp


# FUNCITON DEFINITIONS
################################################################################################
def interpolate(x, n, mode="linear"):
    """
    Perform interpolation on an upsampled signal.

    :param x: Input signal (already upsampled with zeros).
    :param n: Upsampled factor.
    :param mode: Interpolation type. Modes = "linear", "quadratic".
    :return: Interpolated signal.
    """
    nonzero_indices = np.arange(0, len(x), n)
    nonzero_values = x[nonzero_indices]
    interpolation_function = intp.interp1d(nonzero_indices, nonzero_values, kind=mode, fill_value='extrapolate')
    new_indices = np.arange(len(x))
    interpolated_signal = interpolation_function(new_indices)
    return interpolated_signal

def upsample(x, L, offset=0, interpolate_flag=False):
    """
    Discrete signal upsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param L: Int type. Upsample factor.
    :param offset: Int type. Offset size for input array.
    :param interpolate: Boolean type. Flag indicating whether to perform interpolation.
    :return: Numpy array type. Upsampled signal.
    """
    x_upsampled = [0] * offset  # Initialize with offset zeros
    if interpolate_flag:
        x_upsampled.extend(interpolate(x, L))
    else:
        for i, sample in enumerate(x):
            x_upsampled.append(sample)
            x_upsampled.extend([0] * (L - 1))
    return np.array(x_upsampled)

def plot_complex_points(complex_array, constellation):
    """
    Plot complex points on a 2D plane with constellation points labeled.

    :param complex_array: List or numpy array of complex points to plot.
    :param constellation: List of lists, where each inner list contains a complex point and a label.
    """
    # Extract real and imaginary parts of the complex points
    plt.plot([point.real for point in complex_array], [point.imag for point in complex_array], 'ro', label='Received Points')
    
    # Plot constellation points and add labels
    for point, label in constellation:
        plt.plot(point.real, point.imag, 'b+', markersize=10)
        plt.text(point.real, point.imag, f' {label}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    
    # Label axes
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Complex Constellation Plot')
    plt.axhline(0, color='gray', lw=0.5, ls='--')
    plt.axvline(0, color='gray', lw=0.5, ls='--')
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()

def srrc(alpha, m, length):
    """
    Generates a square root raised cosine pulse.

    :param alpha: Roll-off or excess factor.
    :param m: Number of symbols per symbol.
    :param length: Length of pulse. Should be k*m+1 where k is an integer.
    :return: List. Square root raised cosine pulse.
    """
    pulse = []
    for n in range(length):
        n_prime = n - np.floor(length/2)
        if n_prime == 0:
            n_prime = sys.float_info.min  # Handle case when n_prime is zero
        if alpha != 0:
            if np.abs(n_prime) == m/(4*alpha):
                n_prime += 0.1e-12
        num = np.sin(np.pi*((1-alpha)*n_prime/m)) + (4*alpha*n_prime/m)*np.cos(np.pi*((1+alpha)*n_prime/m))
        den = (np.pi*n_prime/m)*(1-(4*alpha*n_prime/m)**2)*np.sqrt(m)
        if den == 0:
            pulse.append(1.0)  # Handle division by zero case
        else:
            pulse.append(num/den)
    return pulse

def convolve(x, h, mode='full'):
    """
    Convolution between two sequences. Can return full or same lengths.

    :param x: List or numpy array. Input sequence one.
    :param h: List or numpy array. Input sequence two.
    :param mode: String. Specifies return sequence length.
    :return: Numpy array. Resulting convolution output.
    """
    N = len(x) + len(h) - 1
    x_padded = np.pad(x, (0, N - len(x)), mode='constant')
    h_padded = np.pad(h, (0, N - len(h)), mode='constant')
    X = np.fft.fft(x_padded)
    H = np.fft.fft(h_padded)
    y = np.fft.ifft(X * H)

    if mode == 'same':
        start = (len(h) - 1) // 2
        end = start + len(x)
        y = y[start:end]
    return y

import numpy as np

def modulate_by_exponential(x, f_c, f_s, phase=0, noise=0):
    """
    Modulates a signal by exponential carrier (cos(x) + jsin(x)) and adds AWGN noise.

    :param x: List or numpy array. Input signal to modulate.
    :param f_c: Float. Carrier frequency of the modulation.
    :param f_s: Float. Sampling frequency of the input signal.
    :param phase: Float. Phase of the modulation in radians. Default is 0.
    :param noise: Float. Standard deviation of the AWGN noise to be added. Default is 0 (no noise).
    :return: Numpy array. Modulated signal with optional noise.
    """
    y = []
    for i, value in enumerate(x):
        modulation_factor = np.exp(-1j * 2 * np.pi * f_c * i / f_s + phase)
        y.append(value * modulation_factor)
    y = np.array(y)
    if noise > 0:
        awgn_noise = np.random.normal(0, noise, y.shape) + 1j * np.random.normal(0, noise, y.shape)
        y += awgn_noise
    return y


def downsample(x, l, offset=0):
    """
    Discrete signal downsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param l: Int type. Downsample factor.
    :param offset: Int type. Offset size for input array.
    :return: Numpy array type. Downsampled signal.
    """
    x_downsampled = [0+0j] * offset  # Initialize with offset zeros
    if l > len(x):
        raise ValueError("Downsample rate larger than signal size.")
    # Loop over the signal, downsampling by skipping every l elements
    for i in range(math.floor(len(x) / l)):
        x_downsampled.append(x[i * l])
    
    return np.array(x_downsampled)

def nearest_neighbor(x, constellation=None, binary=True):
    """
    Find the nearest neighbor in a given constellation.

    :param x: Complex number or array of complex numbers. Point(s) to find the nearest neighbor for.
    :param constellation: 2D numpy array containing point-value pairs. List of complex numbers 
           representing the constellation point and its binary value. Defaults to BPAM/BPSK.
    :return: List of binary values corresponding to the nearest neighbors in the constellation.
    """
    if constellation is None:
        constellation = [[complex(1+0j), 0b1], [complex(-1+0j), 0b0]]
    output = []
    for input_value in x:
        smallest_distance = float('inf')
        value = None
        for point in constellation:
            distance = np.abs(input_value - point[0])
            if distance < smallest_distance:
                smallest_distance = distance
                if binary:
                    value = point[1]
                else:
                    value = point[0]
        output.append(value)
    return output

def bin_to_char(x):
    """
    Converts a binary array into 7-bit ASCII equivalents.

    :param x: List or numpy array type. Input binary signal.
    :return: String containing concatenated ASCII characters.
    """
    segmented_arrays = [x[i:i+7] for i in range(0, len(x), 7)]

    bin_chars = []

    for segment in segmented_arrays:
        binary_string = ''.join(str(bit) for bit in segment)
        decimal_value = int(binary_string, 2)
        ascii_char = chr(decimal_value)
        bin_chars.append(ascii_char)

    return ''.join(bin_chars)

def string_to_ascii_binary(string, num_bits=7):
    """
    Convert a string into its binary representation.

    :param string: Input string to be converted.
    :param num_bits: Number of bits to represent each character (default is 7).
    :return: List of binary strings representing the ASCII values of the input string.
    """
    return ['{:0{width}b}'.format(ord(char), width=num_bits) for char in string]

def error_count(x, y):
    """
    Count the number of errors between two sequences.

    :param x: First list or array for comparison.
    :param y: Second list or array for comparison.
    :return: Integer count of differing elements between the two sequences.
    """
    max_len = max(len(x), len(y))
    x = x + [0] * (max_len - len(x))
    y = y + [0] * (max_len - len(y))
    
    return sum(1 for i in range(max_len) if x[i] != y[i])

def find_subarray_index(small_array, large_array):
    """
    Find the starting index of a small array within a larger array.

    :param small_array: The subarray to search for.
    :param large_array: The array to search within.
    :return: Integer index of the first occurrence of the small array in the large array, or -1 if not found.
    """
    small_len = len(small_array)
    for i in range(len(large_array) - small_len + 1):
        if large_array[i:i + small_len] == small_array:
            return i
    return -1

def clock_offset(signal, sample_rate, offset_fraction):
    """
    Apply a clock offset to a signal based on a specified sample rate and offset fraction.

    :param signal: The input signal to which the clock offset will be applied.
    :param sample_rate: The sample rate of the signal.
    :param offset_fraction: The fraction of a sample to offset the signal.
    :return: The signal with the applied clock offset.
    """
    t = np.arange(0, len(signal) / sample_rate, 1 / sample_rate)
    clock_offset = (1/sample_rate) * offset_fraction
    interpolator = intp.interp1d(t, signal, kind='linear', fill_value='extrapolate')
    t_shifted = t + clock_offset 
    x_shifted = interpolator(t_shifted)
    return x_shifted

def check_unique_word(uw_register, phase_ambiguities):
    """
    Check if the current state of the unique word register matches any entry in the phase ambiguities table.

    :param uw_register: The current state of the unique word shift register as a list of symbols.
    :return: The corresponding phase ambiguity if a match is found, otherwise None.
    """
    uw_register = ''.join(uw_register)
    if uw_register in phase_ambiguities.keys():
        return phase_ambiguities[uw_register]
    else:
        return None


# CLASS DEFINITIONS
################################################################################################
class PLL():
    '''
    This class is used to simulate a Phase-Locked Loop (PLL) discretely.
    Components can be called individually or as a whole depending on user needs.
    Use as an object and initialize variables in __init__ if you want full functionality.
    '''
    lfk2_prev = 0
    phase = 0
    sig_out = 0

    def __init__(self, sample_rate, loop_bandwidth=None, damping_factor=None, gain=1, open_loop=False):
        '''
        Initialize the PLL object with the specified parameters.

        :param sample_rate: Float type. The sampling frequency.
        :param loop_bandwidth: Float type, optional. Loop bandwidth. If specified with damping factor, will compute loop filter gains.
        :param damping_factor: Float type, optional. Damping factor. If specified with loop bandwidth, will compute loop filter gains.
        :param gain: Float type. Gain applied to loop filter output.
        :param open_loop: Boolean type. Allows for open loop testing of required system gain for normalizaiton.
        '''
        self.gain = gain
        self.open_loop = open_loop
        self.compute_loop_constants(sample_rate, loop_bandwidth, damping_factor)

        self.sample_rate = sample_rate
        self.w0 = 1
        self.phase = 0
        self.sig_out = np.exp(1j * self.phase)

    def compute_loop_constants(self, fs, lb, df):
        """
        Compute the loop filter constants based on the given parameters.

        :param fs: Float type. Sampling frequency.
        :param lb: Float type. Loop bandwidth.
        :param df: Float type. Damping factor.
        """
        denominator = 1 + ((2 * df) * ((lb * (1 / fs)) / (df + (1 / (4 * df))))) + ((lb * (1 / fs)) / (df + (1 / (4 * df)))) ** 2
        self.k1 = (1/self.gain) * ((4 * df) * ((lb * (1 / fs)) / (df + (1 / (4 * df))))) / denominator
        self.k2 = (1/self.gain) * (((lb * (1 / fs)) / (df + (1 / (4 * df)))) ** 2) / denominator


    def insert_new_sample(self, incoming_signal, n, internal_signal=None):
        """
        Process a new sample and return the output signal.

        :param incoming_signal: Complex number. The current sample of the received signal.
        :param internal_signal: Complex number, optional. The current signal your LO (local oscillator) is at. Will use default from constructor if left blank.
        :param n: Int type. The current sample index, used to insert a new sample of the received signal and LO.
        :return: Complex number. The output signal after processing.
        """
        if internal_signal is None:
            internal_signal = np.exp(1j * (2 * np.pi * (self.w0 / self.sample_rate) * n + self.phase))
        phase_error = self.phase_detector(internal_signal, incoming_signal)
        v_t = self.loop_filter(phase_error)
        point_out = self.dds(n, v_t)
        if self.open_loop == True:
            return v_t
        else:
            return point_out

    def phase_detector(self, sample1, sample2):
        """
        Calculate the phase difference between two samples.

        :param sample1: Complex number. The first sample.
        :param sample2: Complex number. The second sample.
        :return: Float type. The phase difference between the two samples, scaled by kp.
        """
        angle = np.angle(sample2) - np.angle(sample1)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def loop_filter(self, phase_error):
        """
        Apply the loop filter to the phase error.

        :param phase_error: Float type. The phase error.
        :param k1: Float type, optional. Loop filter gain according to Fig C.2.6.
        :param k2: Float type, optional. Loop filter gain according to Fig C.2.6.
        :return: Float type. The output of the loop filter.
        """
        lfk2 = self.k2 * phase_error + self.lfk2_prev
        output = self.k1 * phase_error + lfk2
        self.lfk2_prev = lfk2
        return output

    def dds(self, n, v):
        """
        Direct Digital Synthesis (DDS) implementation.

        :param n: Int type. The current sample index.
        :param v: Float type. The output of the loop filter.
        :return: Complex number. The output signal of the DDS.
        """
        self.phase += v
        self.sig_out = np.exp(1j * (2 * np.pi * (self.w0 / self.sample_rate) * n + self.phase))
        return self.sig_out
    
    def get_current_phase(self):
        """
        Get the current phase of the PLL.

        :return: Float type. The current phase of the PLL.
        """
        return self.phase

class SCS:
    def __init__(self, samples_per_symbol, loop_bandwidth, damping_factor, gain=1, open_loop=False, invert=False):
        '''
        Initialize the SCS (Symbol Clock Synchronization) subsystem class.

        :param samples_per_symbol: Int type. Number of samples per symbol.
        :param loop_bandwidth: Float type. Determines the lock-on speed to the timing error (similar to PLL).
        :param damping_factor: Float type. Determines the oscillation during lock-on to the timing error (similar to PLL).
        :param gain: Float type. Gain added to timing error detector output (symbolizes Kp).
        '''
        self.samples_per_symbol = samples_per_symbol
        self.gain = gain
        self.open_loop = open_loop
        self.invert = invert

        self.compute_loop_constants(loop_bandwidth, damping_factor, samples_per_symbol)

        self.delay_register_1 = np.zeros(3, dtype=complex)
        self.delay_register_2 = np.zeros(3, dtype=complex)
        self.interpolation_register = np.zeros(3, dtype=complex)

        self.strobe = None
        self.LFK2_prev = 0
        self.decrementor_prev = 0
        self.mu = 0

        self.ted_output_record = []
        self.loop_filter_output_record = []

    def compute_loop_constants(self, loop_bandwidth, damping_factor, samples_per_symbol):
        """
        Compute the loop filter gains based on the loop bandwidth and damping factor.

        :param loop_bandwidth: Float type. Loop bandwidth of control loop.
        :param damping_factor: Float type. Damping factor of control loop.
        :param samples_per_symbol: Float type. Number of samples per symbol.
        :param kp: Float type. Proportional loop filter gain.
        """
        theta_n = (loop_bandwidth/samples_per_symbol)/(damping_factor + 1/(4*damping_factor))
        factor = (4*theta_n)/(1+2*damping_factor*theta_n+theta_n**2)
        self.k1 = damping_factor * factor/self.gain
        self.k2 = theta_n * factor/self.gain

    def insert_new_sample(self, input_sample):
        """
        Insert a new sample into the SCS system, performing interpolation and updating the timing error.

        :param input_sample: Numpy array. Input samples as complex numbers.
        :return: Complex. The interpolated output sample.
        """
        interpolated_sample = self.farrow_interpolator_parabolic(input_sample)
        
        # timing error detector
        error = self.early_late_ted()

        # loop filter
        filtered_error = self.loop_filter(error)

        # calculate w(n)
        w_n = filtered_error + (1 / self.samples_per_symbol)

        # update mod 1 decrementor
        decrementor = self.decrementor_prev - w_n

        # check mod 1 decrementor
        if decrementor < 0:
            self.strobe = True
            decrementor = decrementor + 1 # mod 1
        else:
            self.strobe = False
        
        # calculate mu
        if self.strobe:
            self.mu = self.decrementor_prev / w_n
    
        # update interpolation register (shift)
        self.interpolation_register = np.roll(self.interpolation_register, -1)
        self.interpolation_register[-1] = interpolated_sample

        # store decrementor value
        self.decrementor_prev = decrementor

        if self.open_loop == False:
            if self.invert == False:
                if self.strobe:
                    return interpolated_sample
                else:
                    return None
            else:
                if self.strobe:
                    return None
                else:
                    return interpolated_sample
        else:
            return filtered_error

    def farrow_interpolator_parabolic(self, input_sample):
        """
        Perform parabolic interpolation on the input signal.

        :param input_sample: Numpy array. The input signal to be interpolated.
        :return: Complex. The interpolated output sample.
        """
        tmp = self.mu
        d1next = -0.5 * input_sample
        d2next = input_sample
    
        v2 = -d1next + self.delay_register_1[2] + self.delay_register_1[1] - self.delay_register_1[0]
        v1 = d1next - self.delay_register_1[2] + self.delay_register_2[1] + self.delay_register_1[1] + self.delay_register_1[0]
        v0 = self.delay_register_2[0]
        output = (((v2 * self.mu) + v1) * self.mu + v0)

        self.delay_register_1 = np.roll(self.delay_register_1, -1)
        self.delay_register_2 = np.roll(self.delay_register_2, -1)
        self.delay_register_1[-1] = d1next
        self.delay_register_2[-1] = d2next

        self.mu = tmp
        return output

    def early_late_ted(self):
        """
        Perform early-late timing error detection.

        :return: Float. The calculated timing error based on early and late samples.
        """
        out = 0
        if self.strobe:
            real_est = (np.real(self.interpolation_register[2]) - np.real(self.interpolation_register[0])) * (-1 if np.real(self.interpolation_register[1]) < 0 else 1)
            imag_est = (np.imag(self.interpolation_register[2]) - np.imag(self.interpolation_register[0])) * (-1 if np.imag(self.interpolation_register[1]) < 0 else 1)
            out = real_est
            self.ted_output_record.append(out)
        return out
    
    def loop_filter(self, phase_error):
        """
        Apply a loop filter to the phase error to compute the filtered output.

        :param phase_error: Float type. The timing phase error to be filtered.
        :return: Float. The filtered output from the loop filter.
        """
        k1 = self.k1
        k2 = self.k2
        LFK2 = k2 * phase_error + self.LFK2_prev
        output = k1 * phase_error + LFK2
        self.LFK2_prev = LFK2
        self.loop_filter_output_record.append(output)
        return output

# FUNCTIONS DEFINITIONS (NOT SHOWN IN EXAMPLES)
################################################################################################
window_lut= {"rectangular": {"sidelobe amplitude": 10**(-13/10), 
                             "mainlobe width": 4*np.pi, 
                             "approximation error": 10**(-21/10)},
             "bartlett": {"sidelobe amplitude": 10**(-25/10), 
                          "mainlobe width": 8*np.pi, 
                          "approximation error": 10**(-25/10)},
             "hanning": {"sidelobe amplitude": 10**(-31/10), 
                         "mainlobe width": 8*np.pi, 
                         "approximation error": 10**(-44/10)},
             "hamming": {"sidelobe amplitude": 10**(-41/10), 
                         "mainlobe width": 8*np.pi, 
                         "approximation error": 10**(-53/10)},
             "blackman": {"sidelobe amplitude": 10**(-57/10), 
                          "mainlobe width": 12*np.pi, 
                          "approximation error": 10**(-74/10)}
            }

def apply_window(n, window_type):
    """
    Windowing function used in aid of FIR design flows.

    :param N: Window length (number of coefficients).
    :param window_type: Window type (see below).
    :return w_n: Numpy array type. Calculated window filter coefficients.
    """
    if window_type == "rectangular":
        w_n = np.array([1 for i in range(n)])
    elif window_type == "bartlett":
        w_n = np.array([(1 - (2 * np.abs(i - (n - 1) / 2)) / (n - 1)) for i in range(n)])
    elif window_type == "hanning":
        w_n = np.array([0.5 * (1 - np.cos((2 * np.pi * i) / (n - 1))) for i in range(n)])
    elif window_type == "hamming":
        w_n = np.array([0.54 - 0.46 * np.cos((2 * np.pi * i) / (n - 1)) for i in range(n)])
    elif window_type == "blackman":
        w_n = np.array([0.42 - 0.5 * np.cos((2 * np.pi * i) / (n - 1)) + 0.08 * np.cos((4 * np.pi * i) / (n - 1)) for i in range(n)])
    else: #default to 'rectangular'
        w_n = np.array([1 for i in range(n)])
    return w_n

def fir_low_pass(fc, window=None, fp=None, fs=None, ks=10**(-40/10)):
    """
    FIR low pass filter design.

    :param fc: Digital cutoff frequency.
    :param window: Window used for filter truncation.
    :param fp: Passband digital frequency cutoff.
    :param fs: Stopband digital frequency cutoff.
    :param ks: Stopband attenuation level.
    :return: Numpy array. Coefficients (numerator) of digital lowpass filter.
    """
    if fp is None or fs is None:
        fp = fc - (.125 * fc)
        fs = fc + (.125 * fc)
    if window is None:
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else:
        window_type = window
    n = math.ceil(window_lut[window_type]["mainlobe width"] / np.abs(fs - fp)) # calculating filter order
    alpha = (n - 1) / 2
    h_dn = np.array([(np.sin(fc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else fc / np.pi for i in range(n)]) # generate filter coefficients
    w_n = apply_window(n, window_type)
    return h_dn * w_n

def fir_high_pass(fc, window=None, fp=None, fs=None, ks=10**(-40/10)):
    """
    FIR high pass filter design.

    :param fc: Digital cutoff frequency.
    :param window: Window used for filter truncation (see dictionary below).
    :param fp: Passband digital frequency cutoff.
    :param fs: Stopband digital frequency cutoff.
    :param ks: Stopband attenuation level.
    :return: Numpy array. Coefficients (numerator) of digital highpass filter.
    """
    if fp is None or fs is None:
        fp = fc - (.125 * fc)
        fs = fc + (.125 * fc)
    if window is None:
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else:
        window_type = window
    n = math.ceil(window_lut[window_type]["mainlobe width"] / np.abs(fs - fp)) # calculating filter order
    alpha = (n - 1) / 2
    h_dn = np.array([-(np.sin(fc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else 1 - (fc / np.pi) for i in range(n)]) # generate filter coefficients
    w_n = apply_window(n, window_type)
    return h_dn * w_n

def fir_band_pass(fc1, fc2, window=None, fs1=None, fp1=None, fp2=None, fs2=None, ks1=10**(-40/10), ks2=10**(-40/10)):
    """
    FIR band pass filter design.

    :param fc1: Digital cutoff frequency one.
    :param fc2: Digital cutoff frequency two.
    :param window: Window used for filter truncation (see dictionary below).
    :param fp1: Passband digital frequency cutoff one.
    :param fs1: Stopband digital frequency cutoff one.
    :param fp2: Passband digital frequency cutoff two.
    :param fs2: Stopband digital frequency cutoff two.
    :param ks1: Stopband attenuation level one.
    :param ks2: Stopband attenuation level two.
    :return: Numpy array. Coefficients (numerator) of digital bandpass filter.
    """
    if fp1 is None or fs1 is None or fp2 is None or fs2 is None:
        fs1 = fc1 + (.125 * fc1)
        fp1 = fc1 - (.125 * fc1)
        fp2 = fc2 - (.125 * fc2)
        fs2 = fc2 + (.125 * fc2)
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < min(ks1, ks2)), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else:
        window_type = window
    n = math.ceil(window_lut[window_type]["mainlobe width"] / min(np.abs(fs1 - fp1), np.abs(fs2 - fp2))) # calculating filter order
    alpha = (n - 1) / 2
    h_dn = np.array([((np.sin(fc2 * (i - alpha))) / (np.pi * (i - alpha)) - (np.sin(fc1 * (i - alpha))) / (np.pi * (i - alpha))) if i != alpha else (fc2 / np.pi - fc1 / np.pi)  for i in range(n)]) # determining the filter coefficients
    w_n = apply_window(n, window_type)
    return h_dn*w_n
