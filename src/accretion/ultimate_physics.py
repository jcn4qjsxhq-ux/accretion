"""Physics and accretion-model functions."""

from __future__ import annotations

try:
    from .ultimate_common import *
except ImportError:
    from ultimate_common import *


def flux_to_magnitude(flux, zeropoint):
    """Convert flux to magnitude using m = -2.5 * log10(flux/zp)"""
    return -2.5 * np.log10(flux / zeropoint)


def log_flux_to_magnitude(log_flux, zeropoint):
    """Convert log flux to magnitude"""
    flux = np.exp(log_flux)  # Convert from natural log
    return flux_to_magnitude(flux, zeropoint)


def wavelength_to_frequency(wavelength_micron):
    return c / (wavelength_micron * 1e-6)


def get_filter_wavelengths():
    return {
        'B': 0.44, 'V': 0.55, 'G': 0.625, 'R': 0.64, 'O':0.663,
        'i prime': 0.772, 'Ic': 0.783, 'I': 0.788, 'y': 1.05,
        'z prime': 1.083, 'J': 1.20, 'H': 1.60, 'K': 2.20,
        'W1': 3.4, 'L': 3.452, 'W2': 4.6, 'M': 4.8, 'N': 10.1, 'Q': 20.0
    }


def planck_function(freq, T):
    """Vectorized Planck function (freq in Hz, T in K)."""
    # Ensure arrays
    freq = np.atleast_1d(freq)
    T = np.atleast_1d(T)

    # Broadcast into 2D grid
    freq_grid, T_grid = np.meshgrid(freq, T, indexing="ij")

    x = constants.h * freq_grid / (constants.k * T_grid)
    result = np.empty_like(x)

    # Masks
    mask_large = x > 700
    mask_small = x < 1e-3
    mask_mid = ~(mask_large | mask_small)

    # Pre-factor (same shape as x)
    prefactor = (2 * constants.h * freq_grid**3) / constants.c**2

    # Apply formulas
    result[mask_large] = 1e-50
    result[mask_small] = prefactor[mask_small] * (1/x[mask_small] - 0.5 + x[mask_small]/12)
    result[mask_mid] = prefactor[mask_mid] / (np.exp(x[mask_mid]) - 1)

    return result


def planck_model_custom(frequencies, temperature, distance, av=0.0, R_bb=2 * 1.496e11):
    """
    Custom Planck model using your existing planck_function
    R_bb: emitting radius in metres. If None, defaults to dust sublimation
    radius estimate. Should be ~0.1–1 AU for a 1500 K dust component.
    """


    try:
        frequencies = np.asarray(frequencies)
        
        # Validate temperature (handle scalar or array)
        temp_array = np.atleast_1d(temperature)
        if np.any(temp_array <= 500) or np.any(temp_array > 20000):
            return np.full(len(frequencies), -50.0)
        
        # Use your existing vectorized planck function
        planck_intensity = planck_function(frequencies, temperature)
        
        # Handle the 2D output from planck_function
        if planck_intensity.ndim > 1:
            if planck_intensity.shape[1] > 1:
                planck_intensity = planck_intensity[:, 0]
            else:
                planck_intensity = planck_intensity.squeeze()
        
        # Ensure it's 1D
        planck_intensity = np.atleast_1d(planck_intensity)
        
        # Check for valid planck output BEFORE distance scaling
        # if np.any(planck_intensity < 1e-40) or np.any(~np.isfinite(planck_intensity)):
        #     print(f"Invalid Planck intensity: {planck_intensity}, T={temperature}, freq={frequencies}")
        #     return np.full(len(frequencies), -50.0)

        if np.any(~np.isfinite(planck_intensity)) or np.any(planck_intensity <= 0):
            print(f"EARLY EXIT 1: planck_intensity={planck_intensity}")
            return np.full(len(frequencies), -50.0)
        
        # Convert to flux density at distance
        # emitting_area = np.pi * (1e10)**2  # 10^10 m radius disk

        emitting_area = np.pi * R_bb**2
        solid_angle = emitting_area / (4 * np.pi * distance**2)
        flux_density = planck_intensity * solid_angle
        
        # Apply extinction if provided
        if av > 0:
            wavelength_micron = (constants.c / frequencies) * 1e6
            A_lambda = ccm89_extinction(wavelength_micron) * av
            flux_density *= 10**(-A_lambda / 2.5)
        
        # Check if flux is too small before taking log
        if np.any(flux_density <= 0) or np.any(~np.isfinite(flux_density)):
            print(f"EARLY EXIT 2: flux_density={flux_density}")
            return np.full(len(frequencies), -50.0)
            
        # Convert to log flux
        # log_flux = np.log(flux_density)
        # After computing flux_density in SI (W/m²/Hz):
        flux_density_jy = flux_density * 1e26  # Convert W/m²/Hz → Jy
        
        # Then apply extinction and take log of the Jy value
        log_flux = np.log(flux_density_jy)
        
        # Validate result - allow very negative values, they're just faint!
        # Changed threshold from -55 to -100 to allow faint extincted sources
        if np.any(~np.isfinite(log_flux)) or np.any(log_flux < -100):
            print(f"EARLY EXIT 3: log_flux={log_flux}")
            return np.full(len(frequencies), -50.0)
                
        return log_flux
    
    except Exception as e:
        print(f"Planck model error: {e}")
        import traceback
        traceback.print_exc()
        frequencies = np.asarray(frequencies)
        return np.full(len(frequencies), -50.0)


def ccm89_extinction(wavelength_micron, Rv=3.1):

    wav = np.asarray(wavelength_micron, dtype=float)
    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        x = 1.0 / wav  # inverse microns

    a = np.zeros_like(x)
    b = np.zeros_like(x)

    # mask for x < 0.3 (wavelength > 3.33 µm): extend the infrared law
    mask0 = x < 0.3
    if np.any(mask0):
        a[mask0] = 0.574 * x[mask0]**1.61
        b[mask0] = -0.527 * x[mask0]**1.61

    # mask for 0.3 <= x <= 1.1
    mask1 = (x >= 0.3) & (x <= 1.1)
    if np.any(mask1):
        a[mask1] = 0.574 * x[mask1]**1.61
        b[mask1] = -0.527 * x[mask1]**1.61

    # mask for 1.1 < x <= 3.3
    mask2 = (x > 1.1) & (x <= 3.3)
    if np.any(mask2):
        y = x[mask2] - 1.82
        a[mask2] = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3
                    + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6
                    + 0.32999*y**7)
        b[mask2] = (1.41338*y + 2.28305*y**2 + 1.07233*y**3
                    - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6
                    - 2.09002*y**7)
    
    # mask for 3.3 < x <= 8
    mask3 = (x > 3.3) & (x <= 8)
    if np.any(mask3):
        Fa = np.zeros_like(x)
        Fb = np.zeros_like(x)
        
        # Split mask3 into low and high ranges
        mask3_low = mask3 & (x < 5.9)   # 3.3 < x < 5.9
        mask3_high = mask3 & (x >= 5.9)   # 5.9 <= x <= 8
        
        if np.any(mask3_high):
            Fa[mask3_high] = -0.04473 * (x[mask3_high] - 5.9)**2 - 0.009779 * (x[mask3_high] - 5.9)**3
            Fb[mask3_high] = 0.2130 * (x[mask3_high] - 5.9)**2 + 0.1207 * (x[mask3_high] - 5.9)**3
        
        a[mask3] = (1.752 - 0.316*x[mask3] - 0.104/((x[mask3] - 4.67)**2 + 0.341) + Fa[mask3])
        b[mask3] = (-3.090 + 1.825*x[mask3] + 1.206/((x[mask3] - 4.62)**2 + 0.263) + Fb[mask3])

    # outside these ranges, a and b remain zero
    return (a + b / Rv)


def temperature_profile_vectorized(R_array, M, Mdot, R_star):
    n = R_array.shape[0]
    T = np.empty(n, dtype=np.float64)

    MM = M * Mdot
    for i in range(n):
        R = R_array[i]
        if R <= 0 or M <= 0 or Mdot <= 0 or R_star <= 0:
            T[i] = 1e-10
            continue

        factor = (3 * constants.G * MM) / (8 * np.pi * R**3 * constants.sigma)
        sqrt_term = np.sqrt(R_star / R)

        epps = 1e-10
        if sqrt_term >= 1.0 - epps:  # Avoid negative temperatures
            T[i] = 1e-10
            continue

        Ti = (factor * (1 - sqrt_term))**0.25

        # Maximum temperature constraint
        Tmax = 0.488 * (factor**0.25)
        critical_R = (49/36) * R_star

        if R < critical_R:
            T[i] = min(Ti, Tmax)
        else:
            T[i] = Ti if Ti > 0 else 1e-10

    return T


def accretion_model(nu, Mdot, M_star, R_star, Rout, Rin, distance=1, Av=0, incl=np.deg2rad(40),
                    n_radial_points=100):

    nu = np.atleast_1d(nu).astype(float)
    log_fluxy = np.zeros_like(nu, dtype=np.float64)

    # Validate inputs
    if (Mdot <= 0 or M_star <= 0 or R_star <= 0 or
        Rout <= 0 or Rin <= 0 or distance <= 0 or Rin >= Rout):
        print("all 0 cause input was invalid")
        return np.full_like(nu, -50)

    # Pre-compute radial grid
    try:
        R_grid = np.logspace(np.log10(Rin), np.log10(Rout), n_radial_points)
        dR = np.diff(R_grid)
        R_mid = 0.5 * (R_grid[1:] + R_grid[:-1])

        # Temperatures (same for all frequencies)
        T_grid = temperature_profile_vectorized(R_mid, M_star, Mdot, R_star)
        if np.any(T_grid <= 0) or np.any(~np.isfinite(T_grid)):
            raise ValueError("Invalid temperature grid (<=0 or non-finite)")
            return np.full_like(nu, -50)
    except:
        print("all 0 cause T grid was invalid")
        return np.full_like(nu, -50)

# Broadcast Planck function over (freqs × radii
    try:
        B_values = planck_function(nu, T_grid)  # both 1D
        
        if np.any(~np.isfinite(B_values)):
            raise ValueError("Non-finite B_values")
            print("all 0 cause B grid was invalid")
            return np.full_like(nu, -50)

        # Integrate over radius (axis=1)
        # integrand = 2 * np.pi**2 * B_values * R_mid[None, :]  # Old
        integrand = 4* np.pi**2 * B_values * R_mid[None, :]  # New *2


        # flux = np.sum(integrand * dR[None, :], axis=1) / (4 * np.pi * distance**2)  # Old
        flux = np.cos(incl) * np.sum(integrand * dR[None, :], axis=1) / (2 * np.pi * distance**2)  # New *2

        # Convert to Jy
        flux_jy = flux / 1e-26
        if Av > 0:
            wavelength_micron = (constants.c / nu) * 1e6
            A_lambda = ccm89_extinction(wavelength_micron) * Av
            flux_red = flux_jy * 10**(-A_lambda / 2.5)

        else:
            flux_red = flux_jy

        # Take log safely
        log_fluxy = np.where((flux_red > 0) & np.isfinite(flux_red),
                            np.log(flux_red), -50.0)

    except:
        flux_fluxy = np.full_like(nu, -50)
        print("all 0 cause calculation was invalid")
    return log_fluxy

__all__ = ['flux_to_magnitude', 'log_flux_to_magnitude', 'wavelength_to_frequency', 'get_filter_wavelengths', 'planck_function', 'planck_model_custom', 'ccm89_extinction', 'temperature_profile_vectorized', 'accretion_model']
