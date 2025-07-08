import glob
import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as const
import os # Import os for path manipulation

# Import custom functions from your modeling modules
from emissivity_model_functions import calc_emiss_and_SF_props, expectation_value_mass_weighted_linear
from PDF_model_functions import expectation_value_volume_weighted_linear, lognormal, power_law, combine_PL_LN

def process_single_model_output(filepath, df_grid):
    """
    Reads, processes, and calculates derived properties for a single model output file.

    This function performs several key steps:
    1. Reads raw output data from an individual model run.
    2. Cleans the data by handling NaN and infinite values.
    3. Extracts scalar properties and cross-references them with a main model grid.
    4. Calculates various derived quantities such as filling factors and alpha conversion factors.
    5. **Performs mass-weighted calculations:** A significant part of this function involves
       computing mass-weighted expectation values for various physical quantities
       (like heating/cooling rates, column densities, volume densities, and temperatures).
       These calculations use the `expectation_value_mass_weighted_linear` function,
       which averages properties by weighting them by the mass they represent within
       the cloud's density distribution. This ensures that the derived averages are
       physically representative of the entire molecular cloud.
    6. Organizes all extracted and calculated data into a single row DataFrame.

    Args:
        filepath (str): The path to the individual model output TSV file (e.g., 'output/modelX.tsv').
        df_grid (pd.DataFrame): The DataFrame containing the initial model grid properties
                                 (from 'model_grid.tsv') for cross-referencing.

    Returns:
        pd.DataFrame: A DataFrame row containing the processed data for the current model,
                      or None if an error occurs during processing.
    """
    try:
        # Read the individual model output file
        df_ = pd.read_csv(filepath, sep='\t')

        # Clean up the DataFrame: replace infinities/NaNs and drop all-NaN columns
        df_ = df_.replace([np.inf, -np.inf], np.nan)
        df_ = df_.dropna(axis=1, how='all')

        # Ensure single-element arrays are squeezed to scalar values for easier access
        # This is common when reading mixed data types or from certain Astropy table exports.
        df_ = df_.apply(np.squeeze)

        # Extract key scalar properties from the current model output
        # These are assumed to be consistent across all rows of df_ for a single model file
        dv = df_['sigma_v_3D'].values[0]
        # Convert surface density from cm^-2 to M_sun/pc^2 using Astropy units
        Sigma = (df_['N_surface'].values[0] * u.cm**-2 * (2.33 * (const.m_p + const.m_e))).to(u.M_sun / u.pc**2).value

        # Cross-reference with the main model grid to get Galaxy and original Sigma
        # This step assumes a direct match based on Sigma and dv, and that these values
        # are unique enough to identify a single row in df_grid.
        idx = np.argmin(np.abs(1 - df_grid['Sigma'] / Sigma))
        idy = np.argmin(np.abs(1 - df_grid['sigv'] / dv))

        gal = None # Initialize to None
        original_Sigma_from_grid = None # Initialize to None

        # If the indices match, it means a consistent model was found in the grid
        if idx == idy:
            gal = df_grid.loc[idx]['Galaxy']
            original_Sigma_from_grid = df_grid.loc[idx]['Sigma']
        else:
            print(f"Warning: Could not find a unique match in df_grid for model {filepath}. "
                  f"Sigma: {Sigma}, dv: {dv}. Skipping Galaxy and original Sigma assignment.")

        npoints = len(df_) # Number of data points (e.g., density bins) in this model's profile

        # Extract emissivities, radii, and other properties
        hcn_emiss = df_['emiss10_hcn_K_kms_cm-2'].values[0]
        co_emiss = df_['emiss10_co_K_kms_cm-2'].values[0]
        r_co_mass_wt = df_['r10_co_mass_wt'].values[0]
        r_hcn_mass_wt = df_['r10_hcn_mass_wt'].values[0]
        r_cloud_mass_wt = df_['r_cloud_mass_wt'].values[0]

        # Calculate filling factors (fractions of area/volume)
        ff_co = r_co_mass_wt**2 / r_cloud_mass_wt**2 # Filling factor for CO
        ff_hcn = r_hcn_mass_wt**2 / r_co_mass_wt**2 # Filling factor for HCN relative to CO

        # Calculate alpha conversion factors (inverse of emissivity, with a constant)
        # 1.36 and 6.3e19 are constants from the original script, likely related to
        # unit conversions or specific definitions of alpha.
        alpha_hcn = 1.36 / hcn_emiss / 6.3e19
        alpha_co = 1.36 / co_emiss / 6.3e19

        # Extract excitation temperatures and optical depths
        tex_hcn = df_['tex10_hcn_weight'].values[0]
        tex_co = df_['tex10_co_weight'].values[0]
        tau_hcn = df_['tau10_hcn_weight'].values[0]
        tau_co = df_['tau10_co_weight'].values[0]

        # Extract mean temperature and cloud properties
        temp_mean = df_['temp_mean'].values[0]
        Mc_Msun = df_['Mc_Msun'].values[0]
        rs_pc = df_['rs_pc'].values[0]
        kappa = df_['kappa'].values[0]

        # Extract PDF normalization constants and parameters
        N_norm = df_['N norm'].values[0]
        C_norm = df_['C norm'].values[0]
        sigma_s = df_['sigma_s'].values[0]
        s_0 = df_['s_0'].values[0]
        alpha_s = df_['alpha_s'].values[0]
        s_t = df_['s_t'].values[0]
        n_src = df_['n_src'].values[0]

        # Reconstruct large-scale PDF and related arrays for calculations if needed
        # These arrays are used for weighted averages later in the script.
        s_array_large = np.linspace(-20, 20, 500)
        LN_large = lognormal(N_norm, sigma_s, s_array_large, s_0)
        PL_large = power_law(N_norm, C_norm, alpha_s, s_array_large)
        PDF_large = combine_PL_LN(s_t, PL_large, LN_large, s_array_large)
        n_array_large = n_src * np.exp(s_array_large) / u.cm**3
        r_array_large = rs_pc * (n_array_large.value / n_src)**(-1 / kappa)

        # --- Unused/Commented-out calculations from original script ---
        # These blocks are kept as comments to preserve original intent/exploration.
        # pdf_n = np.copy(PDF_large / n_array_large)
        # pdf_trap = (pdf_n + np.roll(pdf_n, -1)) / 2
        # pdf_trap[-1] = np.nan
        # dn = np.abs(n_array_large - np.roll(n_array_large, -1))
        # dn[-1] = np.nan
        # Mtot = 4. / 3 * np.pi * np.nansum(pdf_trap * dn * 2.33 * (const.m_p + const.m_e)) * (rs_pc * u.pc)**3
        # Mtot = Mtot.to(u.M_sun).value

        # Mc_const = 4 * np.pi * Sigma * u.M_sun / u.pc**2 * (rs_pc * u.pc)**2
        # Mc_const = Mc_const.to(u.M_sun)
        # Mc_const = Mc_const.value
        # -------------------------------------------------------------

        # Extract arrays for weighted averages
        pdf_arr = df_['PDF'].values
        s_arr = df_['s_array'].values
        n_arr = 10**df_['log n_arr'].values
        N_arr = 10**df_['log N_arr'].values
        temp_arr = df_['temp'].values

        flux_hcn = df_['FLUX_Kkm/s_hcn'].values
        flux_co = df_['FLUX_Kkm/s_co'].values

        # Calculate emissivities for each density point
        emiss_hcn_arr = df_['FLUX_Kkm/s_hcn'].values / N_arr
        emiss_co_arr = df_['FLUX_Kkm/s_co'].values / N_arr

        # Handle NaN values in heating/cooling terms before summation
        gammacompress = np.nan_to_num(df_['gammacompress'].values)
        gammaturb = np.nan_to_num(df_['gammaturb'].values)
        gammaH23b = np.nan_to_num(df_['gammaH23b'].values)
        gammaH2dust = np.nan_to_num(df_['gammaH2dust'].values)
        gammacosmic = np.nan_to_num(df_['gammacosmic'].values)

        # Calculate mass-weighted sums of heating/cooling terms.
        # These calculations use the 'expectation_value_mass_weighted_linear' function
        # to compute averages of various physical quantities (like heating/cooling rates,
        # column densities, volume densities, and temperatures) where each value is
        # weighted by the mass it represents within the cloud's density distribution.
        # This provides a physically representative average for the entire cloud.
        gammacompress_sum = expectation_value_mass_weighted_linear(n_arr, gammacompress, pdf_arr / n_arr)
        gammaturb_sum = expectation_value_mass_weighted_linear(n_arr, gammaturb, pdf_arr / n_arr)
        gammaH23b_sum = expectation_value_mass_weighted_linear(n_arr, gammaH23b, pdf_arr / n_arr)
        gammaH2dust_sum = expectation_value_mass_weighted_linear(n_arr, gammaH2dust, pdf_arr / n_arr)
        gammacosmic_sum = expectation_value_mass_weighted_linear(n_arr, gammacosmic, pdf_arr / n_arr)

        # Calculate mass-weighted average N and n from PDF
        N_pdf_avg = expectation_value_mass_weighted_linear(n_arr, N_arr, pdf_arr / n_arr)
        n_pdf_avg = expectation_value_mass_weighted_linear(n_arr, n_arr, pdf_arr / n_arr)

        # Calculate mass-weighted average N and n, weighted by emissivity
        # These represent the effective column/volume densities traced by the emission
        N_hcn_emiss_wt = expectation_value_mass_weighted_linear(n_arr, N_arr, emiss_hcn_arr * pdf_arr / n_arr)
        N_co_emiss_wt = expectation_value_mass_weighted_linear(n_arr, N_arr, emiss_co_arr * pdf_arr / n_arr)
        n_hcn_emiss_wt = expectation_value_mass_weighted_linear(n_arr, n_arr, emiss_hcn_arr * pdf_arr / n_arr)
        n_co_emiss_wt = expectation_value_mass_weighted_linear(n_arr, n_arr, emiss_co_arr * pdf_arr / n_arr)

        # Calculate mass-weighted average temperatures, weighted by emissivity
        T_hcn_emiss_wt = expectation_value_mass_weighted_linear(n_arr, temp_arr, emiss_hcn_arr * pdf_arr / n_arr)
        T_co_emiss_wt = expectation_value_mass_weighted_linear(n_arr, temp_arr, emiss_co_arr * pdf_arr / n_arr)

        # Calculate mass-weighted average fluxes
        Ico_mass_avg = expectation_value_mass_weighted_linear(n_arr, flux_co, pdf_arr / n_arr)
        Ihcn_mass_avg = expectation_value_mass_weighted_linear(n_arr, flux_hcn, pdf_arr / n_arr)
        # Ihcop_mass_avg = expectation_value_mass_weighted_linear(n_arr, flux_hcop, pdf_arr / n_arr) # Commented in original

        # Calculate volume-weighted average fluxes
        Ico_vol_avg = expectation_value_volume_weighted_linear(n_arr, flux_co, pdf_arr / n_arr)
        Ihcn_vol_avg = expectation_value_volume_weighted_linear(n_arr, flux_hcn, pdf_arr / n_arr)
        # Ihcop_vol_avg = expectation_value_mass_weighted_linear(n_arr/n_arr, flux_hcop, pdf_arr/n_arr) # Commented in original

        # Create a dictionary for the current model's summarized data
        # This will become a single row in the final DataFrame
        model_data = {
            'dv': dv,
            'Sigma': Sigma,
            'Galaxy': gal,
            'Original_Sigma_from_Grid': original_Sigma_from_grid, # Keep track of cross-referenced Sigma
            'N_points': npoints,
            'model': int(filepath.split('/model')[-1].split('.')[0]), # Extract model number from filename
            'hcn_emiss': hcn_emiss,
            'co_emiss': co_emiss,
            'r_co_mass_wt': r_co_mass_wt,
            'r_hcn_mass_wt': r_hcn_mass_wt,
            'r_cloud_mass_wt': r_cloud_mass_wt,
            'ff_co': ff_co,
            'ff_hcn': ff_hcn,
            'alpha_hcn': alpha_hcn,
            'alpha_co': alpha_co,
            'tex_hcn': tex_hcn,
            'tex_co': tex_co,
            'tau_hcn': tau_hcn,
            'tau_co': tau_co,
            'temp_mean': temp_mean,
            'Mc_Msun': Mc_Msun,
            'rs_pc': rs_pc,
            'kappa': kappa,
            'N_norm': N_norm,
            'C_norm': C_norm,
            'sigma_s': sigma_s,
            's_0': s_0,
            'alpha_s': alpha_s,
            's_t': s_t,
            'n_src': n_src,
            'gammacompress_sum': gammacompress_sum,
            'gammaturb_sum': gammaturb_sum,
            'gammaH23b_sum': gammaH23b_sum,
            'gammaH2dust_sum': gammaH2dust_sum,
            'gammacosmic_sum': gammacosmic_sum,
            'N_pdf_avg': N_pdf_avg,
            'n_pdf_avg': n_pdf_avg,
            'N_hcn_emiss_wt': N_hcn_emiss_wt,
            'N_co_emiss_wt': N_co_emiss_wt,
            'n_hcn_emiss_wt': n_hcn_emiss_wt,
            'n_co_emiss_wt': n_co_emiss_wt,
            'T_hcn_emiss_wt': T_hcn_emiss_wt,
            'T_co_emiss_wt': T_co_emiss_wt,
            'I_hcn_mass_avg': Ihcn_mass_avg,
            'I_co_mass_avg': Ico_mass_avg,
            'I_hcn_vol_avg': Ihcn_vol_avg,
            'I_co_vol_avg': Ico_vol_avg,
            # Add other columns from the commented-out 'columns' list if they are actually used
            # in the input files and need to be carried through.
            # Example: 'f5p5': df_['f5p5'].values[0],
            # If these are not present in df_, they will cause KeyError.
        }

        # Convert dictionary to a DataFrame row
        return pd.DataFrame([model_data])

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

def apply_filtering_conditions(df):
    """
    Applies filtering conditions to the combined DataFrame based on physical criteria.

    Args:
        df (pd.DataFrame): The combined DataFrame of all model outputs.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Define filtering requirements based on the original script's logic
    # Ensure these columns exist in df before applying filters
    ff_req = (df['ff_co'] <= 1.) & (df['ff_co'] > 1e-2) & (df['ff_hcn'] > 1e-3)
    tex_req = (df['tex_co'] > 0) & (df['tex_hcn'] > 0) & \
              (df['tex_hcn'] <= 1.5 * df['temp_mean']) & (df['tex_co'] <= 1.5 * df['temp_mean'])
    alpha_req = (df['alpha_co'] < 10) # Original had '& (alpha_co > 0.1)' commented out
    tau_req = (df['tau_co'] > 1.) & (df['tau_co'] < 200)
    ratio_req = (df['hcn_emiss'] / df['co_emiss'] > 1e-3) # Original had '& (hcn/co < 1.)' commented out

    # Combine all conditions. The original script only used 'npoints > 300' and 'tex_req'
    # for the main concatenation, then dropped NaNs.
    # This function applies all defined 'req' conditions.
    # If the intent was to apply these as a post-filter, this is the place.
    # If they were meant to gate individual model processing, they should be in process_single_model_output.
    # Based on original 'if (npoints > 300) & tex_req:', I'll apply npoints filter here too.
    # The 'npoints > 300' seems to be a quality cut on the number of valid density points in the model.
    # It's applied during concatenation in the original, so we'll make it a filter here.

    # Apply filters
    # The original script's filter was: if (npoints > 300) & tex_req:
    # Let's ensure npoints is a column if it's used for filtering here.
    # Assuming 'npoints' is now a column in the df.
    filtered_df = df[
        (df['N_points'] > 300) &
        tex_req &
        alpha_req &
        tau_req &
        ratio_req
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    print(f"Initial models: {len(df)}. Models after filtering: {len(filtered_df)}")
    return filtered_df


def main():
    """
    Main function to orchestrate the reading, processing, combining, and filtering
    of all model output files.
    """
    print("Starting analysis of model outputs...")

    # --- Configuration ---
    # Path to the initial model grid TSV file
    model_grid_path = 'model_grid.tsv'
    # Directory containing individual model output files
    output_files_directory = 'output/'
    # Output filename for the combined and processed data
    combined_output_filename = 'combined_model_output.csv'

    # --- Step 1: Load the main model grid ---
    print(f"Loading initial model grid from {model_grid_path}...")
    try:
        df_grid = pd.read_csv(model_grid_path, sep='\t')
        print(f"Loaded {len(df_grid)} entries from model grid.")
    except FileNotFoundError:
        print(f"Error: {model_grid_path} not found. Please ensure it exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {model_grid_path}: {e}")
        sys.exit(1)

    # --- Step 2: Find all individual model output files ---
    # Use os.path.join for robust path handling across OS
    fils = glob.glob(os.path.join(output_files_directory, 'model*'))
    if not fils:
        print(f"No model output files found in {output_files_directory}. "
              "Please ensure 'input_measurements_grid.py' and 'run_model_grid.py' "
              "have been run and generated output files.")
        sys.exit(1)
    print(f"Found {len(fils)} individual model output files to process.")

    # --- Step 3: Process each individual model file ---
    df_all_list = [] # Use a list to collect DataFrames for efficient concatenation
    processed_count = 0
    total_files = len(fils)

    for i, f in enumerate(fils):
        processed_df_row = process_single_model_output(f, df_grid)
        if processed_df_row is not None:
            df_all_list.append(processed_df_row)
            processed_count += 1

        # Print progress update
        if (i + 1) % 100 == 0 or (i + 1) == total_files:
            print(f"Processed {i + 1}/{total_files} files.")

    if not df_all_list:
        print("No valid model output rows were processed. Exiting.")
        sys.exit(1)

    # Concatenate all processed individual model DataFrames into one
    print("Concatenating all processed model data...")
    df_combined = pd.concat(df_all_list, ignore_index=True)
    print(f"Combined DataFrame has {len(df_combined)} rows before final filtering.")

    # --- Step 4: Apply final filtering and cleanup ---
    # Drop rows with any NaN values that might have resulted from calculations
    # or initial data issues not caught by df_.dropna(axis=1, how='all')
    print("Applying final NaN drop...")
    df_combined = df_combined.dropna()
    print(f"DataFrame has {len(df_combined)} rows after dropping NaNs.")

    # Apply the specific filtering conditions
    print("Applying filtering conditions...")
    df_final = apply_filtering_conditions(df_combined)
    print(f"Final DataFrame has {len(df_final)} rows after all filtering.")


    # --- Step 5: Save the combined and filtered data ---
    try:
        df_final.to_csv(combined_output_filename, index=False)
        print(f"Combined and filtered model output saved to {combined_output_filename}")
    except Exception as e:
        print(f"Error saving combined output to {combined_output_filename}: {e}")

    print("Analysis complete.")

if __name__ == '__main__':
    main()
