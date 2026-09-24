"""File import/export helpers from `Ultimate.ipynb`."""

from __future__ import annotations

try:
    from .ultimate_common import *
except ImportError:
    from ultimate_common import *


DEFAULT_IRTF_LXD_MASK_REGIONS = [
    (1.9e-6, 2.45e-6),
    (2.45e-6, 2.9e-6),
    (4.15e-6, 4.6e-6),
]

# ADD YOUR OWN NORMALIZATION FACTORS HERE IF YOU HAVE NEW IRTF SPECTRA TO LOAD
DEFAULT_IRTF_LXD_NORMALIZATION_FACTORS = {
    '20150722': 1.09216,
    '20230722': 2.66122,
    '20250331': 0.889563,
    '20250415': 2.32305,
    '20250615': 0.970957,
    '20250621': 0.970957,
}

DEFAULT_IRTF_SXD_MASK_REGIONS = [
    (0.68e-6, 0.74e-6),
    (1.3e-6,  1.4e-6),
    (1.85e-6, 1.95e-6),
    (2.45e-6, 2.9e-6),
]

# ADD YOUR OWN NORMALIZATION FACTORS HERE IF YOU HAVE NEW IRTF SPECTRA TO LOAD
DEFAULT_IRTF_SXD_NORMALIZATION_FACTORS = {
    '20230722': 0.928525,
    '20250331': 0.858249,
    '20250415': 0.942584,
    '20250615': 1.04895,
    '20250621': 1.04895,
    '20250705': 0.788377,
}


# Observation times associated with the reduced XSHOOTER filenames.  The files
# themselves contain only wavelength and flux columns, so the epoch metadata is
# supplied here from the observing log/photometry database.
DEFAULT_XSHOOTER_NIR_EPOCHS = {
    '201506': {'JD': 2457209.599, 'Date': '2015-07-06'},
    '202402': {'JD': 2460364.850, 'Date': '2024-02-24'},
    '202410': {'JD': 2460594.478, 'Date': '2024-10-10'},
}

DEFAULT_XSHOOTER_NIR_MASK_REGIONS = [
    # The raw NIR spectra are unstable throughout the deep 1.4 and 1.9 micron
    # atmospheric bands.  The previous 1.30--1.41 micron mask both discarded
    # useful J-band continuum and retained the noisy red edge of the 1.4 micron
    # band (clearly visible to about 1.48 micron in these reductions).
    (1.34e-6, 1.48e-6),
    (1.80e-6, 1.96e-6),
]


def assign_jd_day(df):
    """
    Add a day-level JD key without rounding the Julian Date.

    When a Date column is available, rows from the same calendar date share the
    median observed JD for that date. Otherwise rows keep their exact JD as the
    grouping key, preserving half-day JDs such as 2455271.5.
    """
    out = df.copy()
    out['JD'] = pd.to_numeric(out['JD'], errors='coerce')

    if 'Date' in out.columns:
        date_key = out['Date'].astype(str).str.strip()
        valid_date = date_key.ne('') & date_key.str.lower().ne('nan')
        out['_JD_group_key'] = np.where(valid_date, 'date:' + date_key, 'jd:' + out['JD'].astype(str))
    else:
        out['_JD_group_key'] = 'jd:' + out['JD'].astype(str)

    jd_day_map = out.groupby('_JD_group_key')['JD'].median()
    out['JD_day'] = out['_JD_group_key'].map(jd_day_map).astype(float)
    return out


def repair_flux_from_magnitude(df):
    """Recover positive fluxes when table precision rounded small values to 0."""
    if df is None or len(df) == 0 or not {'Flux', 'Mag', 'ZP'}.issubset(df.columns):
        return df

    out = df.copy()
    flux = pd.to_numeric(out['Flux'], errors='coerce')
    mag = pd.to_numeric(out['Mag'], errors='coerce')
    zp = pd.to_numeric(out['ZP'], errors='coerce')
    recover_flux = (flux <= 0) & np.isfinite(mag) & np.isfinite(zp) & (zp > 0)
    if recover_flux.any():
        flux.loc[recover_flux] = zp.loc[recover_flux] * 10 ** (-0.4 * mag.loc[recover_flux])
        out['Flux'] = flux

    if {'Fluxerr', 'Magerr'}.issubset(out.columns):
        fluxerr = pd.to_numeric(out['Fluxerr'], errors='coerce')
        magerr = pd.to_numeric(out['Magerr'], errors='coerce')
        recover_fluxerr = (fluxerr <= 0) & (flux > 0) & np.isfinite(magerr) & (magerr > 0)
        if recover_fluxerr.any():
            fluxerr.loc[recover_fluxerr] = flux.loc[recover_fluxerr] * np.log(10.0) * magerr.loc[recover_fluxerr] / 2.5
            out['Fluxerr'] = fluxerr

    return out


def average_daily_measurements(df):
    """
    Average multiple measurements on the same day with the same filter.
    Preserve the Julian Date used for plotting and matching.
    """
    df = repair_flux_from_magnitude(df)
    df = assign_jd_day(df)
    averaged_data = []

    for (jd_group, filter_name), group in df.groupby(['_JD_group_key', 'Filter']):
        jd_day = float(group['JD_day'].iloc[0])
        if len(group) == 1:
            row = group.iloc[0].to_dict()
            row['JD_day'] = jd_day
            row.pop('_JD_group_key', None)
            averaged_data.append(row)
            continue

        weights = 1.0 / (group['Magerr'].values**2 + 1e-10)
        avg_mag = np.average(group['Mag'].values, weights=weights)
        avg_magerr = 1.0 / np.sqrt(np.sum(weights))

        flux_weights = 1.0 / (group['Fluxerr'].values**2 + 1e-10)
        avg_flux = np.average(group['Flux'].values, weights=flux_weights)
        avg_fluxerr = 1.0 / np.sqrt(np.sum(flux_weights))

        averaged_row = {
            'JD': group['JD'].mean(),
            'JD_day': jd_day,
            'Date': group['Date'].iloc[0],
            'Filter': filter_name,
            'Lambda': group['Lambda'].iloc[0],
            'Mag': avg_mag,
            'Magerr': avg_magerr,
            'ZP': group['ZP'].mean(),
            'Flux': avg_flux,
            'Fluxerr': avg_fluxerr,
            'Flag': group['Flag'].iloc[0],
            'Interpolated': 'averaged',
        }

        if 'Mag_re' in group.columns:
            mag_re_vals = group['Mag_re'].values
            magerr_re_vals = group['Magerr_re'].values if 'Magerr_re' in group.columns else np.full_like(mag_re_vals, 0.1)
            mag_re_weights = 1.0 / (magerr_re_vals**2 + 1e-10)
            averaged_row['Mag_re'] = np.average(mag_re_vals, weights=mag_re_weights)
            averaged_row['Magerr_re'] = 1.0 / np.sqrt(np.sum(mag_re_weights))

        if 'Flux_re' in group.columns:
            flux_re_vals = group['Flux_re'].values
            fluxerr_re_vals = group['Fluxerr_re'].values if 'Fluxerr_re' in group.columns else np.full_like(flux_re_vals, 1e-12)
            flux_re_weights = 1.0 / (fluxerr_re_vals**2 + 1e-10)
            averaged_row['Flux_re'] = np.average(flux_re_vals, weights=flux_re_weights)
            averaged_row['Fluxerr_re'] = 1.0 / np.sqrt(np.sum(flux_re_weights))

        averaged_data.append(averaged_row)

    df_avg = pd.DataFrame(averaged_data)
    df_avg.sort_values(['JD_day', 'Lambda'], inplace=True)
    df_avg.reset_index(drop=True, inplace=True)

    print(f"After averaging: {len(df_avg)} data points (reduced from {len(df)})")
    return df_avg


def get_daily_data(df, required_filters, debug=False):
    """Extract days where all required filters have measurements."""
    df_avg = average_daily_measurements(df)
    daily_data = {}

    for jd_day, day_data in df_avg.groupby('JD_day'):
        available_filters = set(day_data['Filter'].unique())
        if set(required_filters).issubset(available_filters):
            daily_data[jd_day] = day_data
        elif debug:
            missing = set(required_filters) - available_filters
            print(f"JD {jd_day}: missing filters {sorted(missing)}")

    print(f'Number of days having all required filters {required_filters}: {len(daily_data)}')
    if daily_data:
        total_measurements = sum(len(data) for data in daily_data.values())
        print(f'Total measurements in selected days: {total_measurements}')
    return daily_data


def _read_av_molecular_results(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            parts = stripped.split()
            if len(parts) < 4:
                print(f"Skipping {path}:{line_number}: expected JD, Av, method, instrument")
                continue

            try:
                jd = float(parts[0])
                av = float(parts[1])
            except ValueError:
                print(f"Skipping {path}:{line_number}: JD/Av are not numeric")
                continue

            rows.append({
                'JD': jd,
                'Av': av,
                'method': ' '.join(parts[2:-1]).strip().lower(),
                'instrument': parts[-1],
            })

    if not rows:
        return pd.DataFrame(columns=['JD', 'Av', 'method', 'instrument'])

    return pd.DataFrame(rows).sort_values('JD').reset_index(drop=True)


def _parse_irtf_text_header(filepath):
    header = {}
    data_start = None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if stripped == '#END':
                data_start = line_number + 1
                break

            if not stripped.startswith('#') or '=' not in stripped:
                continue

            payload = stripped[1:]
            key, remainder = payload.split('=', 1)
            value = remainder.split('/', 1)[0].strip()
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1].replace("''", "'")
            header[key.strip()] = value

    if data_start is None:
        raise ValueError(f"Could not find '#END' marker in {filepath}")

    return header, data_start


def _load_irtf_spectra(mode, directory='IRTF', mask_regions=None,
                       normalization_factors=None, sample_stride=1,
                       include_nonpositive=False):
    """Shared loader for both LXD and SXD IRTF spectra."""
    mode = mode.lower()
    if mode not in ('lxd', 'sxd'):
        raise ValueError("mode must be 'lxd' or 'sxd'")
    if sample_stride < 1:
        raise ValueError('sample_stride must be >= 1')

    if mask_regions is None:
        mask_regions = (DEFAULT_IRTF_LXD_MASK_REGIONS if mode == 'lxd'
                        else DEFAULT_IRTF_SXD_MASK_REGIONS)
    mask_regions = list(mask_regions)

    base_norm = (DEFAULT_IRTF_LXD_NORMALIZATION_FACTORS if mode == 'lxd'
                 else DEFAULT_IRTF_SXD_NORMALIZATION_FACTORS)
    normalization_map = base_norm.copy()
    if normalization_factors:
        normalization_map.update({str(k): float(v) for k, v in normalization_factors.items()})

    filepaths = sorted(glob.glob(os.path.join(directory, f'*{mode}*.txt')))
    if not filepaths:
        raise FileNotFoundError(f'No IRTF {mode.upper()} spectra found in {directory}')

    frames = []
    for filepath in filepaths:
        header, data_start = _parse_irtf_text_header(filepath)
        spectrum = np.loadtxt(filepath, skiprows=data_start)
        if spectrum.ndim == 1:
            spectrum = spectrum[np.newaxis, :]
        if spectrum.shape[1] < 4:
            raise ValueError(f'Expected 4 spectrum columns in {filepath}, found {spectrum.shape[1]}')

        wavelength_um = spectrum[:, 0]
        flux_jy       = spectrum[:, 1]
        fluxerr_jy    = spectrum[:, 2]
        flag          = spectrum[:, 3].astype(int)

        date_obs = str(header.get('DATE_OBS', '')).strip()
        date_code = date_obs.replace('-', '')
        normalization_factor = normalization_map.get(date_code, 1.0)

        wavelength_m = wavelength_um * 1e-6
        frequency_hz = constants.c / wavelength_m

        valid_mask = (
            np.isfinite(wavelength_um)
            & np.isfinite(wavelength_m)
            & np.isfinite(frequency_hz)
            & np.isfinite(flux_jy)
            & np.isfinite(fluxerr_jy)
        )
        for region_start, region_end in mask_regions:
            valid_mask &= ~((wavelength_m >= region_start) & (wavelength_m <= region_end))
        if not include_nonpositive:
            valid_mask &= flux_jy > 0
        if sample_stride > 1:
            sampled_mask = np.zeros_like(valid_mask, dtype=bool)
            sampled_mask[np.flatnonzero(valid_mask)[::sample_stride]] = True
            valid_mask &= sampled_mask

        wavelength_um  = wavelength_um[valid_mask]
        wavelength_m   = wavelength_m[valid_mask]
        frequency_hz   = frequency_hz[valid_mask]
        flux_jy        = flux_jy[valid_mask]    / normalization_factor
        fluxerr_jy     = fluxerr_jy[valid_mask] / normalization_factor
        flag           = flag[valid_mask]

        mjd_obs = pd.to_numeric(pd.Series([header.get('MJD_OBS', np.nan)]), errors='coerce').iloc[0]
        jd_obs  = float(mjd_obs + 2400000.5) if np.isfinite(mjd_obs) else np.nan
        jd_day  = float(jd_obs) if np.isfinite(jd_obs) else None

        frame = pd.DataFrame({
            'JD':                  jd_obs,
            'JD_day':              jd_day,
            'Date':                date_obs,
            'DateCode':            date_code,
            'Filter':              mode.upper(),
            'Lambda':              wavelength_um,
            'Lambda_m':            wavelength_m,
            'Frequency':           frequency_hz,
            'Mag':                 np.nan,
            'Magerr':              np.nan,
            'ZP':                  np.nan,
            'Flux':                flux_jy,
            'Fluxerr':             fluxerr_jy,
            'Flag':                flag,
            'Interpolated':        'no',
            'Instrument':          str(header.get('INSTRUME', header.get('INSTR', 'IRTF'))).strip(),
            'Mode':                str(header.get('MODENAME', mode.upper())).strip(),
            'Filename':            os.path.basename(filepath),
            'NormalizationFactor': normalization_factor,
        })
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df.sort_values(['JD', 'Lambda_m'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    label = mode.upper()
    print(f"Successfully loaded {len(df)} spectral samples from {len(filepaths)} IRTF {label} files")
    print(f"Dates: {sorted(df['DateCode'].dropna().unique().tolist())}")
    print(f"Wavelength coverage: {df['Lambda'].min():.3f}–{df['Lambda'].max():.3f} μm")
    return df


def load_irtf_lxd_spectra(directory='IRTF', mask_regions=None,
                          normalization_factors=None, sample_stride=1,
                          include_nonpositive=False):
    """Load IRTF LXD spectra into a dataframe compatible with fitting utilities."""
    return _load_irtf_spectra(
        'lxd', directory=directory, mask_regions=mask_regions,
        normalization_factors=normalization_factors, sample_stride=sample_stride,
        include_nonpositive=include_nonpositive,
    )


def load_irtf_sxd_spectra(directory='IRTF', mask_regions=None,
                          normalization_factors=None, sample_stride=1,
                          include_nonpositive=False):
    """Load IRTF SXD spectra into a dataframe compatible with fitting utilities."""
    return _load_irtf_spectra(
        'sxd', directory=directory, mask_regions=mask_regions,
        normalization_factors=normalization_factors, sample_stride=sample_stride,
        include_nonpositive=include_nonpositive,
    )


def load_xshooter_nir_spectra(
    directory='Xshooter',
    wavelength_range=(1.0e-6, 2.0e-6),
    mask_regions=None,
    epoch_metadata=None,
    fractional_error=0.10,
    bin_width_micron=0.005,
    sigma_clip=4.0,
    sample_stride=1,
    include_nonpositive=False,
    convert_air_to_vacuum=True,
):
    r"""Load reduced XSHOOTER NIR spectra in the fitter's Jy convention.

    The input files contain air wavelength in micron and :math:`F_\lambda` in
    ``erg s^-1 cm^-2 Angstrom^-1``.  This loader optionally converts air to
    vacuum wavelength and always converts flux density using

    ``F_nu[Jy] = F_lambda * lambda[Angstrom]^2 / c[Angstrom/s] * 1e23``.

    The supplied spectra do not contain an uncertainty column.  Raw detector
    pixels are therefore robustly combined into 0.005 micron continuum bins by
    default.  Importantly, negative pixels participate in a bin before a
    non-positive *bin* is rejected; dropping negative raw pixels would bias a
    noisy spectrum high.  The per-bin uncertainty combines the robust scatter
    with a 10 per cent calibration floor.  The spectral fitter subsequently
    adds its standard 5 per cent systematic floor in quadrature.
    """
    if sample_stride < 1:
        raise ValueError('sample_stride must be >= 1')
    if fractional_error <= 0:
        raise ValueError('fractional_error must be positive')
    if bin_width_micron is not None and bin_width_micron <= 0:
        raise ValueError('bin_width_micron must be positive or None')
    if sigma_clip is not None and sigma_clip <= 0:
        raise ValueError('sigma_clip must be positive or None')

    mask_regions = list(
        DEFAULT_XSHOOTER_NIR_MASK_REGIONS if mask_regions is None else mask_regions
    )
    epoch_map = {key: dict(value) for key, value in DEFAULT_XSHOOTER_NIR_EPOCHS.items()}
    if epoch_metadata:
        for key, value in epoch_metadata.items():
            if isinstance(value, dict):
                epoch_map[str(key)] = dict(value)
            else:
                jd, date = value
                epoch_map[str(key)] = {'JD': float(jd), 'Date': str(date)}

    filepaths = sorted(glob.glob(os.path.join(directory, 'final_nir_*.txt')))
    if not filepaths:
        raise FileNotFoundError(f'No XSHOOTER NIR spectra found in {directory}')

    frames = []
    c_angstrom_per_second = constants.c * 1e10
    for filepath in filepaths:
        date_code = os.path.splitext(os.path.basename(filepath))[0].rsplit('_', 1)[-1]
        if date_code not in epoch_map:
            raise ValueError(
                f'Missing epoch metadata for {os.path.basename(filepath)}; '
                'pass epoch_metadata={date_code: {"JD": ..., "Date": ...}}'
            )

        spectrum = np.loadtxt(filepath, skiprows=1)
        if spectrum.ndim == 1:
            spectrum = spectrum[np.newaxis, :]
        if spectrum.shape[1] < 2:
            raise ValueError(f'Expected wavelength and F_lambda columns in {filepath}')

        wavelength_air_m = spectrum[:, 0].astype(float) * 1e-6
        flux_lambda = spectrum[:, 1].astype(float)
        wavelength_m = wavelength_air_m.copy()
        if convert_air_to_vacuum:
            inverse_micron = 1e4 / (wavelength_air_m * 1e10)
            refractive_index = (
                1.0
                + 0.00008336624212083
                + 0.02408926869968 / (130.1065924522 - inverse_micron**2)
                + 0.0001599740894897 / (38.92568793293 - inverse_micron**2)
            )
            wavelength_m = wavelength_air_m * refractive_index

        # F_lambda is tabulated per Angstrom at the input (air) wavelength.
        # Use that wavelength for the density conversion, while retaining the
        # vacuum wavelength for frequency/model evaluation.
        wavelength_angstrom = wavelength_air_m * 1e10
        frequency_hz = constants.c / wavelength_m
        flux_jy = flux_lambda * wavelength_angstrom**2 / c_angstrom_per_second * 1e23

        valid = (
            np.isfinite(wavelength_m)
            & np.isfinite(frequency_hz)
            & np.isfinite(flux_jy)
            & (wavelength_m >= float(wavelength_range[0]))
            & (wavelength_m <= float(wavelength_range[1]))
        )
        for region_start, region_end in mask_regions:
            valid &= ~((wavelength_m >= region_start) & (wavelength_m <= region_end))

        wavelength_m = wavelength_m[valid]
        frequency_hz = frequency_hz[valid]
        flux_jy = flux_jy[valid]

        if bin_width_micron is None:
            binned_wavelength_m = wavelength_m
            binned_frequency_hz = frequency_hz
            binned_flux_jy = flux_jy
            binned_scatter_jy = np.zeros_like(binned_flux_jy)
            samples_per_bin = np.ones(len(binned_flux_jy), dtype=int)
        else:
            bin_width_m = float(bin_width_micron) * 1e-6
            start_m = float(wavelength_range[0])
            bin_index = np.floor((wavelength_m - start_m) / bin_width_m).astype(int)
            binned_rows = []
            for index in np.unique(bin_index):
                in_bin = bin_index == index
                bin_wavelength = wavelength_m[in_bin]
                bin_flux = flux_jy[in_bin]
                if len(bin_flux) == 0:
                    continue

                median_flux = np.median(bin_flux)
                robust_scatter = 1.4826 * np.median(np.abs(bin_flux - median_flux))
                keep = np.ones(len(bin_flux), dtype=bool)
                if sigma_clip is not None and robust_scatter > 0:
                    keep = np.abs(bin_flux - median_flux) <= float(sigma_clip) * robust_scatter
                if not np.any(keep):
                    continue

                kept_flux = bin_flux[keep]
                kept_wavelength = bin_wavelength[keep]
                center_flux = float(np.mean(kept_flux))
                final_scatter = float(
                    1.4826 * np.median(np.abs(kept_flux - np.median(kept_flux)))
                )
                # XSHOOTER NIR spectra are sampled at roughly three detector
                # pixels per resolution element, so adjacent pixels are not
                # independent when estimating the error on a bin.
                effective_n = max(len(kept_flux) / 3.0, 1.0)
                statistical_error = final_scatter / np.sqrt(effective_n)
                calibration_error = float(fractional_error) * abs(center_flux)
                flux_error = np.hypot(statistical_error, calibration_error)
                center_wavelength = float(np.mean(kept_wavelength))
                binned_rows.append((
                    center_wavelength,
                    constants.c / center_wavelength,
                    center_flux,
                    final_scatter,
                    flux_error,
                    len(kept_flux),
                ))

            if binned_rows:
                binned = np.asarray(binned_rows, dtype=float)
                binned_wavelength_m = binned[:, 0]
                binned_frequency_hz = binned[:, 1]
                binned_flux_jy = binned[:, 2]
                binned_scatter_jy = binned[:, 3]
                fluxerr_jy = binned[:, 4]
                samples_per_bin = binned[:, 5].astype(int)
            else:
                binned_wavelength_m = np.array([], dtype=float)
                binned_frequency_hz = np.array([], dtype=float)
                binned_flux_jy = np.array([], dtype=float)
                binned_scatter_jy = np.array([], dtype=float)
                fluxerr_jy = np.array([], dtype=float)
                samples_per_bin = np.array([], dtype=int)

        if bin_width_micron is None:
            fluxerr_jy = np.abs(binned_flux_jy) * float(fractional_error)

        keep_bins = np.isfinite(binned_flux_jy) & np.isfinite(fluxerr_jy) & (fluxerr_jy > 0)
        if not include_nonpositive:
            keep_bins &= binned_flux_jy > 0
        if sample_stride > 1:
            sampled = np.zeros_like(keep_bins, dtype=bool)
            sampled[np.flatnonzero(keep_bins)[::sample_stride]] = True
            keep_bins &= sampled

        wavelength_m = binned_wavelength_m[keep_bins]
        frequency_hz = binned_frequency_hz[keep_bins]
        flux_jy = binned_flux_jy[keep_bins]
        fluxerr_jy = fluxerr_jy[keep_bins]
        binned_scatter_jy = binned_scatter_jy[keep_bins]
        samples_per_bin = samples_per_bin[keep_bins]

        metadata = epoch_map[date_code]
        jd = float(metadata['JD'])
        date = str(metadata['Date'])
        frames.append(pd.DataFrame({
            'JD': jd,
            'JD_day': jd,
            'Date': date,
            'DateCode': date_code,
            'Filter': 'XSHOOTER_NIR',
            'Lambda': wavelength_m * 1e6,
            'Lambda_m': wavelength_m,
            'Frequency': frequency_hz,
            'Mag': np.nan,
            'Magerr': np.nan,
            'ZP': np.nan,
            'Flux': flux_jy,
            'Fluxerr': fluxerr_jy,
            'Flag': 1,
            'Interpolated': 'no',
            'Instrument': 'VLT/XSHOOTER',
            'Mode': 'NIR',
            'Filename': os.path.basename(filepath),
            'InputFluxConvention': 'F_lambda [erg/s/cm^2/Angstrom]',
            'FractionalErrorAssigned': float(fractional_error),
            'BinWidthMicron': np.nan if bin_width_micron is None else float(bin_width_micron),
            'SamplesPerBin': samples_per_bin,
            'RobustScatterJy': binned_scatter_jy,
        }))

    df = pd.concat(frames, ignore_index=True)
    df.sort_values(['JD', 'Lambda_m'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'Successfully loaded {len(df)} samples from {len(filepaths)} XSHOOTER NIR files')
    print(f'Dates: {sorted(df["DateCode"].unique().tolist())}')
    print(f'Fitting wavelength coverage: {df["Lambda"].min():.3f}–{df["Lambda"].max():.3f} μm')
    return df


def calibrate_spectra_to_photometry(
    spectral_df,
    photometry_df,
    filters=('J', 'H'),
    max_time_difference_days=1.0,
    window_half_width_micron=0.03,
):
    """Apply a smooth absolute/colour correction from simultaneous photometry.

    A power law in wavelength is fitted through the requested photometric
    anchors for each spectral epoch.  Magnitude and zeropoint are preferred for
    the anchor flux because legacy tables round small Jy values in ``Flux``.
    The uncorrected values are retained in ``FluxUncalibrated`` and
    ``FluxerrUncalibrated``.
    """
    if spectral_df is None or len(spectral_df) == 0:
        return spectral_df
    if photometry_df is None or len(photometry_df) == 0:
        raise ValueError('photometry_df must contain calibration measurements')
    if window_half_width_micron <= 0:
        raise ValueError('window_half_width_micron must be positive')

    calibrated = spectral_df.copy()
    calibrated['FluxUncalibrated'] = pd.to_numeric(calibrated['Flux'], errors='coerce')
    calibrated['FluxerrUncalibrated'] = pd.to_numeric(calibrated['Fluxerr'], errors='coerce')
    calibrated['FluxCalibrationFactor'] = np.nan
    calibrated['PhotometricCalibration'] = ''

    photometry = repair_flux_from_magnitude(photometry_df)
    photometry = photometry.copy()
    photometry['JD'] = pd.to_numeric(photometry['JD'], errors='coerce')
    photometry['Lambda'] = pd.to_numeric(photometry['Lambda'], errors='coerce')

    for jd_day, epoch in calibrated.groupby('JD_day'):
        anchor_wavelengths = []
        anchor_scales = []
        anchor_labels = []
        for filter_name in filters:
            candidates = photometry[
                (photometry['Filter'].astype(str).str.lower() == str(filter_name).lower())
                & np.isfinite(photometry['JD'])
                & (np.abs(photometry['JD'] - float(jd_day)) <= float(max_time_difference_days))
            ].copy()
            if candidates.empty:
                continue
            candidates['_time_distance'] = np.abs(candidates['JD'] - float(jd_day))
            row = candidates.sort_values('_time_distance').iloc[0]
            wavelength_um = float(row['Lambda'])

            mag = pd.to_numeric(pd.Series([row.get('Mag', np.nan)]), errors='coerce').iloc[0]
            zp = pd.to_numeric(pd.Series([row.get('ZP', np.nan)]), errors='coerce').iloc[0]
            if np.isfinite(mag) and np.isfinite(zp) and zp > 0:
                photometric_flux = float(zp * 10 ** (-0.4 * mag))
            else:
                photometric_flux = float(row.get('Flux', np.nan))

            local = epoch[
                np.abs(pd.to_numeric(epoch['Lambda'], errors='coerce') - wavelength_um)
                <= float(window_half_width_micron)
            ]
            local_flux = pd.to_numeric(local['FluxUncalibrated'], errors='coerce')
            local_flux = local_flux[np.isfinite(local_flux) & (local_flux > 0)]
            if len(local_flux) == 0 or not np.isfinite(photometric_flux) or photometric_flux <= 0:
                continue
            spectral_flux = float(np.median(local_flux))
            if spectral_flux <= 0:
                continue

            anchor_wavelengths.append(wavelength_um)
            anchor_scales.append(photometric_flux / spectral_flux)
            anchor_labels.append(
                f'{filter_name}@{float(row["JD"]):.3f}:x{photometric_flux / spectral_flux:.4g}'
            )

        if not anchor_scales:
            raise ValueError(f'No usable photometric calibration anchors near JD {jd_day}')

        anchor_wavelengths = np.asarray(anchor_wavelengths, dtype=float)
        anchor_scales = np.asarray(anchor_scales, dtype=float)
        reference_wavelength = float(np.exp(np.mean(np.log(anchor_wavelengths))))
        if len(anchor_scales) >= 2 and np.ptp(np.log(anchor_wavelengths)) > 0:
            slope, intercept = np.polyfit(
                np.log(anchor_wavelengths / reference_wavelength),
                np.log(anchor_scales),
                1,
            )
        else:
            slope = 0.0
            intercept = float(np.mean(np.log(anchor_scales)))

        epoch_index = epoch.index
        wavelength = pd.to_numeric(calibrated.loc[epoch_index, 'Lambda'], errors='coerce')
        correction = np.exp(intercept) * (wavelength / reference_wavelength) ** slope
        calibrated.loc[epoch_index, 'Flux'] = (
            calibrated.loc[epoch_index, 'FluxUncalibrated'] * correction
        )
        calibrated.loc[epoch_index, 'Fluxerr'] = (
            calibrated.loc[epoch_index, 'FluxerrUncalibrated'] * correction
        )
        calibrated.loc[epoch_index, 'FluxCalibrationFactor'] = correction
        calibrated.loc[epoch_index, 'PhotometricCalibration'] = '; '.join(anchor_labels)

    return calibrated


def load_database_new(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.strip().split('\n')

    # Skip header lines (first 2 lines contain headers and separators)
    data_start = 2
    data = []

    for line_num, line in enumerate(lines[data_start:], data_start):
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        parts = line.split()

        # Parse the fixed format columns
        jd = float(parts[0])
        date = parts[1]
        lambda_val = float(parts[2])
        mag = float(parts[3])
        magerr = float(parts[4]) if parts[4] != '-99.000' else 0.001
        zp = float(parts[5])
        flux = float(parts[6])
        fluxerr = float(parts[7])
        interpolated = parts[8]  # 'yes' or 'no'
        flag = int(parts[9])

        if flux <= 0 and np.isfinite(mag) and np.isfinite(zp) and zp > 0:
            flux = zp * 10 ** (-0.4 * mag)
        if fluxerr <= 0 and flux > 0 and np.isfinite(magerr) and magerr > 0:
            fluxerr = flux * np.log(10.0) * magerr / 2.5

        # Map lambda to filter names (common astronomical filters)
        filter_mapping = {
            0.44: 'B', 0.55: 'V', 
            0.513: 'V', 0.518: 'V',
            0.625: 'G', 0.62: 'G', 
            0.64: 'R',
            0.663: 'O', 0.767: 'i prime',
            0.772: 'i prime', 0.783:'Ic', # O itt Ic
            0.788: 'I', 0.801: 'I',
            1.083: 'z prime', 1.05:'y',
            1.20: 'J',
            1.221: 'J', 1.235: 'J',
            1.24: 'J',1.25: 'J',
            1.60: 'H', 1.64: 'H',
            1.65: 'H', 1.662: 'H',
            2.144: 'K', 2.159: 'K',
            2.18: 'K', 2.20: 'K',
            3.4: 'W1',
            3.353: 'L',
            3.452: 'L',
            3.5: 'L', 3.60: 'L',
            3.77: 'L', 3.80: 'L', 
            4.6: 'W2', 4.603: 'W2',
            4.80: 'M', 10.1: 'N',
            20.0:'Q'
        }

        filter_name = filter_mapping.get(lambda_val, f'λ{lambda_val}')

        # Create data entry matching the old format structure
        data.append({
            'JD': jd,
            'Date': date,
            'Filter': filter_name,
            'Lambda': lambda_val,
            'Mag': mag,
            'Magerr': magerr,
            'ZP': zp,
            'Flux': flux,
            'Fluxerr': fluxerr,
            'Flag': flag,
            'Interpolated': interpolated  # New column from new format
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    df.sort_values('JD', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Successfully loaded {len(data)} data points")
    print(f"Column names: {df.columns.tolist()}")
    print(f"Available filters: {sorted(df['Filter'].unique())}")
    return df


def _to_builtin(value):
    """Convert numpy/pandas types to plain Python types for saving."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _repair_flux_from_magnitude(frame):
    """Backward-compatible alias for saved-result loaders."""
    return repair_flux_from_magnitude(frame)


def _preserve_loaded_jd_day(daily_df):
    """Preserve saved daily-data keys, inferring them only for legacy files."""
    if 'JD' not in daily_df.columns or len(daily_df) == 0:
        return daily_df, {}

    repaired = daily_df.copy()
    old_jd_day = pd.to_numeric(repaired['JD_day'], errors='coerce') if 'JD_day' in repaired.columns else None
    if 'JD_day' in repaired.columns:
        saved_jd_day = old_jd_day
        if saved_jd_day.notna().any():
            repaired['JD_day'] = saved_jd_day
            mapping_frame = pd.DataFrame({'old': saved_jd_day, 'new': saved_jd_day}).dropna()
            key_map = {
                float(old_value): float(old_value)
                for old_value in mapping_frame['old'].dropna().unique()
            }
            return repaired.dropna(subset=['JD_day']).copy(), key_map

    repaired['JD'] = pd.to_numeric(repaired['JD'], errors='coerce')
    if 'Date' in repaired.columns:
        date_key = repaired['Date'].astype(str).str.strip()
        valid_date = date_key.ne('') & date_key.str.lower().ne('nan')
        group_key = np.where(valid_date, 'date:' + date_key, 'jd:' + repaired['JD'].astype(str))
    else:
        group_key = 'jd:' + repaired['JD'].astype(str)

    jd_day_map = repaired.assign(_JD_group_key=group_key).groupby('_JD_group_key')['JD'].median()
    repaired['JD_day'] = pd.Series(group_key, index=repaired.index).map(jd_day_map).astype(float)

    key_map = {}
    if old_jd_day is not None:
        mapping_frame = pd.DataFrame({'old': old_jd_day, 'new': repaired['JD_day']}).dropna()
        for old_value, group in mapping_frame.groupby('old'):
            new_values = group['new'].dropna().unique()
            if len(new_values) == 1:
                key_map[float(old_value)] = float(new_values[0])

    return repaired, key_map


def _remap_daily_params_keys(daily_params, key_map):
    if not key_map:
        return daily_params
    remapped = {}
    for jd_key, params in daily_params.items():
        try:
            lookup_key = float(jd_key)
        except Exception:
            lookup_key = jd_key
        remapped[key_map.get(lookup_key, jd_key)] = params
    return remapped


def _project_root():
    """Return repository root from an installed or local src checkout."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _generated_datas_dir():
    output_dir = os.path.join(_project_root(), 'results', 'generated', 'datas')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generated_data_path(filepath, default_extension='.txt'):
    """Resolve generated table/text paths into results/generated/datas."""
    filepath = str(filepath)
    root, ext = os.path.splitext(filepath)
    if root == '':
        root = filepath
    extension = ext or default_extension

    if os.path.isabs(filepath):
        output_path = root + extension
    else:
        output_name = os.path.basename(root.rstrip('/\\')) + extension
        output_path = os.path.join(_generated_datas_dir(), output_name)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    return output_path


def _resolve_result_root(filepath, for_write=False):
    """Return the path stem used for a saved result bundle."""
    filepath = str(filepath)
    root, ext = os.path.splitext(filepath)
    if root == '':
        root = filepath

    if os.path.isabs(filepath):
        return root, ext

    if not for_write:
        exact_daily_path = f"{root}_daily_params.csv"
        exact_txt_path = root + (ext if ext in ('.txt', '.json') else '.txt')
        if os.path.exists(exact_daily_path) or os.path.exists(exact_txt_path):
            return root, ext

    output_name = os.path.basename(root.rstrip('/\\'))
    return os.path.join(_generated_datas_dir(), output_name), ext


def result_saver(results, filepath='fit_results', file_format='csv', include_daily_data=True):
    """
    Save fitting results so they can be reopened later without refitting.

    Parameters
    ----------
    results : dict
        Output from fitting functions (e.g. ultimate_fitting_regularized).
    filepath : str
        Base path of output files (without extension required).
    file_format : str
        'csv' or 'txt'.
    include_daily_data : bool
        If True and available, save `daily_data` too.

    Returns
    -------
    dict
        Paths of written files.
    """
    import json

    if not isinstance(results, dict):
        raise ValueError("results must be a dictionary")

    fmt = str(file_format).strip().lower()
    root, ext = _resolve_result_root(filepath, for_write=True)

    written_files = {}

    # ---- TXT mode: single JSON-like text file ----
    if fmt in ('txt', 'text', 'json'):
        out_file = root + (ext if ext in ('.txt', '.json') else '.txt')

        payload = {
            'success': bool(results.get('success', False)),
            'daily_params': _to_builtin(results.get('daily_params', {})),
            'global_params': _to_builtin(results.get('global_params', {})),
            'param_errors': _to_builtin(results.get('param_errors', {})),
            'fit_info': _to_builtin(results.get('fit_info', {})),
            'chi_squared': _to_builtin(results.get('chi_squared', np.nan)),
            'reduced_chi_squared': _to_builtin(results.get('reduced_chi_squared', np.nan)),
            'regularization_term': _to_builtin(results.get('regularization_term', np.nan)),
            'total_objective': _to_builtin(results.get('total_objective', np.nan)),
        }

        if include_daily_data and ('daily_data' in results):
            daily_data_rows = []
            for jd_day, day_df in results.get('daily_data', {}).items():
                if isinstance(day_df, pd.DataFrame):
                    tmp = day_df.copy()
                    if 'JD_day' not in tmp.columns:
                        tmp['JD_day'] = float(jd_day)
                    daily_data_rows.extend(tmp.to_dict(orient='records'))
            payload['daily_data_rows'] = _to_builtin(daily_data_rows)

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        written_files['results_txt'] = out_file
        print(f"Saved fitting results to {out_file}")
        return written_files

    # ---- CSV mode: multi-file bundle ----
    if fmt == 'csv':
        os.makedirs(os.path.dirname(root) or '.', exist_ok=True)

        # Daily params
        daily_rows = []
        for jd_day, params in results.get('daily_params', {}).items():
            row = {'JD_day': float(jd_day)}
            if isinstance(params, dict):
                row.update(_to_builtin(params))
            daily_rows.append(row)
        daily_df = pd.DataFrame(daily_rows).sort_values('JD_day') if daily_rows else pd.DataFrame(columns=['JD_day'])
        daily_path = f"{root}_daily_params.csv"
        daily_df.to_csv(daily_path, index=False)
        written_files['daily_params_csv'] = daily_path

        # Global params + errors
        global_rows = []
        gparams = results.get('global_params', {})
        gerrs = results.get('param_errors', {})
        for k, v in gparams.items():
            global_rows.append({
                'parameter': str(k),
                'value': _to_builtin(v),
                'error': _to_builtin(gerrs.get(k, np.nan))
            })
        global_df = pd.DataFrame(global_rows)
        global_path = f"{root}_global_params.csv"
        global_df.to_csv(global_path, index=False)
        written_files['global_params_csv'] = global_path

        # Fit info (key-value)
        fit_info_rows = []
        for k, v in results.get('fit_info', {}).items():
            vv = _to_builtin(v)
            if isinstance(vv, (dict, list)):
                vv = json.dumps(vv)
            fit_info_rows.append({'key': str(k), 'value': vv})
        for scalar_key in ('success', 'chi_squared', 'reduced_chi_squared', 'regularization_term', 'total_objective'):
            vv = _to_builtin(results.get(scalar_key, np.nan if scalar_key != 'success' else False))
            fit_info_rows.append({'key': scalar_key, 'value': vv})

        fit_info_df = pd.DataFrame(fit_info_rows)
        fit_info_path = f"{root}_fit_info.csv"
        fit_info_df.to_csv(fit_info_path, index=False)
        written_files['fit_info_csv'] = fit_info_path

        # Optional daily_data
        if include_daily_data and ('daily_data' in results):
            all_rows = []
            for jd_day, day_df in results.get('daily_data', {}).items():
                if isinstance(day_df, pd.DataFrame):
                    tmp = day_df.copy()
                    if 'JD_day' not in tmp.columns:
                        tmp['JD_day'] = float(jd_day)
                    all_rows.append(tmp)
            if all_rows:
                daily_data_df = pd.concat(all_rows, ignore_index=True)
                daily_data_path = f"{root}_daily_data.csv"
                daily_data_df.to_csv(daily_data_path, index=False)
                written_files['daily_data_csv'] = daily_data_path

        print("Saved fitting result bundle:")
        for k, v in written_files.items():
            print(f"  - {k}: {v}")
        return written_files

    raise ValueError("file_format must be 'csv' or 'txt'")


__all__ = [
    'assign_jd_day',
    'repair_flux_from_magnitude',
    'average_daily_measurements',
    'get_daily_data',
    'load_irtf_lxd_spectra',
    'load_irtf_sxd_spectra',
    'load_xshooter_nir_spectra',
    'calibrate_spectra_to_photometry',
    'load_database_new',
    'generated_data_path',
    'result_saver',
    'result_opener',
]


def result_opener(filepath='fit_results', file_format='csv', Debug=False):
    """
    Reopen fitting results saved by `result_saver`.

    Parameters
    ----------
    filepath : str
        Base path used in saving.
    file_format : str
        'csv' or 'txt'.
    Debug : bool
        If True, print debug information.

    Returns
    -------
    dict
        Reconstructed results dictionary.
    """
    import json
    import ast

    fmt = str(file_format).strip().lower()
    root, ext = _resolve_result_root(filepath, for_write=False)

    if fmt in ('txt', 'text', 'json'):
        in_file = root + (ext if ext in ('.txt', '.json') else '.txt')
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"Could not find: {in_file}")

        with open(in_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        results = {
            'success': bool(loaded.get('success', False)),
            'daily_params': {},
            'global_params': loaded.get('global_params', {}),
            'param_errors': loaded.get('param_errors', {}),
            'fit_info': loaded.get('fit_info', {}),
            'chi_squared': loaded.get('chi_squared', np.nan),
            'reduced_chi_squared': loaded.get('reduced_chi_squared', np.nan),
            'regularization_term': loaded.get('regularization_term', np.nan),
            'total_objective': loaded.get('total_objective', np.nan),
        }

        for jd_key, param_dict in loaded.get('daily_params', {}).items():
            try:
                jd = float(jd_key)
            except Exception:
                jd = jd_key
            results['daily_params'][jd] = param_dict

        if 'daily_data_rows' in loaded:
            ddf = pd.DataFrame(loaded['daily_data_rows'])
            if 'JD_day' in ddf.columns and len(ddf) > 0:
                ddf = _repair_flux_from_magnitude(ddf)
                ddf['JD_day'] = pd.to_numeric(ddf['JD_day'], errors='coerce')
                ddf = ddf.dropna(subset=['JD_day']).copy()
                ddf, key_map = _preserve_loaded_jd_day(ddf)
                results['daily_params'] = _remap_daily_params_keys(results['daily_params'], key_map)
                results['daily_data'] = {jd: grp.copy() for jd, grp in ddf.groupby('JD_day')}
        if (not np.isfinite(results.get('reduced_chi_squared', np.nan))) and ('dof' in results.get('fit_info', {})):
            dof = results['fit_info'].get('dof')
            if dof is not None and dof > 0 and np.isfinite(results.get('chi_squared', np.nan)):
                results['reduced_chi_squared'] = float(results['chi_squared']) / float(dof)

        print(f"Loaded fitting results from {in_file}")
        return results

    if fmt == 'csv':
        daily_path = f"{root}_daily_params.csv"
        global_path = f"{root}_global_params.csv"
        fit_info_path = f"{root}_fit_info.csv"
        daily_data_path = f"{root}_daily_data.csv"

        for required_file in (daily_path, global_path, fit_info_path):
            if not os.path.exists(required_file):
                raise FileNotFoundError(f"Missing required file: {required_file}")

        def _safe_read_csv(path, default_columns=None):
            if not os.path.exists(path):
                return pd.DataFrame(columns=default_columns or [])
            if os.path.getsize(path) == 0:
                return pd.DataFrame(columns=default_columns or [])
            # Treat whitespace-only files as empty to avoid EmptyDataError.
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(4096)
                if sample.strip() == '':
                    return pd.DataFrame(columns=default_columns or [])
            try:
                return pd.read_csv(path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                return pd.DataFrame(columns=default_columns or [])

        daily_df = _safe_read_csv(daily_path, default_columns=['JD_day'])
        global_df = _safe_read_csv(global_path, default_columns=['parameter', 'value', 'error'])
        fit_info_df = _safe_read_csv(fit_info_path, default_columns=['key', 'value'])

        results = {
            'success': False,
            'daily_params': {},
            'global_params': {},
            'param_errors': {},
            'fit_info': {},
            'chi_squared': np.nan,
            'reduced_chi_squared': np.nan,
            'regularization_term': np.nan,
            'total_objective': np.nan,
        }

        # daily params
        if 'JD_day' in daily_df.columns:
            for _, row in daily_df.iterrows():
                jd = float(row['JD_day'])
                params = {
                    k: row[k] for k in daily_df.columns
                    if k != 'JD_day' and pd.notna(row[k])
                }
                results['daily_params'][jd] = _to_builtin(params)

        # global params
        if len(global_df) > 0:
            for _, row in global_df.iterrows():
                p = str(row['parameter'])
                results['global_params'][p] = _to_builtin(row['value'])
                if 'error' in global_df.columns and pd.notna(row['error']):
                    results['param_errors'][p] = _to_builtin(row['error'])

        # fit info and scalar values
        if {'key', 'value'}.issubset(set(fit_info_df.columns)):
            for _, row in fit_info_df.iterrows():
                k = str(row['key'])
                v = row['value']

                # Try parse JSON/list/dict first, then python literals, else keep string
                parsed = v
                if isinstance(v, str):
                    vv = v.strip()
                    try:
                        parsed = json.loads(vv)
                    except Exception:
                        try:
                            parsed = ast.literal_eval(vv)
                        except Exception:
                            parsed = vv

                if k in ('success', 'chi_squared', 'reduced_chi_squared', 'regularization_term', 'total_objective'):
                    if k == 'success':
                        if isinstance(parsed, str):
                            results[k] = parsed.strip().lower() in ('true', '1', 'yes')
                        else:
                            results[k] = bool(parsed)
                    else:
                        try:
                            results[k] = float(parsed)
                        except Exception:
                            results[k] = np.nan
                else:
                    results['fit_info'][k] = _to_builtin(parsed)

        # optional daily_data
        if os.path.exists(daily_data_path):
            ddf = _safe_read_csv(daily_data_path)
            if 'JD_day' in ddf.columns and len(ddf) > 0:
                ddf = _repair_flux_from_magnitude(ddf)
                ddf['JD_day'] = pd.to_numeric(ddf['JD_day'], errors='coerce')
                ddf = ddf.dropna(subset=['JD_day']).copy()
                ddf, key_map = _preserve_loaded_jd_day(ddf)
                results['daily_params'] = _remap_daily_params_keys(results['daily_params'], key_map)
                results['daily_data'] = {jd: grp.copy() for jd, grp in ddf.groupby('JD_day')}
        if (not np.isfinite(results.get('reduced_chi_squared', np.nan))) and ('dof' in results.get('fit_info', {})):
            dof = results['fit_info'].get('dof')
            if dof is not None and dof > 0 and np.isfinite(results.get('chi_squared', np.nan)):
                results['reduced_chi_squared'] = float(results['chi_squared']) / float(dof)

        if Debug:
            print("Loaded fitting result bundle:")
            print(f"  - {daily_path}")
            print(f"  - {global_path}")
            print(f"  - {fit_info_path}")
        if os.path.exists(daily_data_path):
            print(f"  - {daily_data_path}")

        return results

    raise ValueError("file_format must be 'csv' or 'txt'")
