# --- 3. Scientific Processing Engine ---
def process_scientific_ftir(file, s_val):
    try:
        # Load Data
        df = pd.read_csv(file, header=None, sep=None, engine='python', on_bad_lines='skip')
        df = df.apply(pd.to_numeric, errors='coerce').dropna().iloc[:, :2]
        df.columns = ['Wavenumber', 'Raw_Intensity']
        df = df.sort_values('Wavenumber', ascending=True)

        # 1. Convert Transmittance to Absorbance if needed
        if raw_data_format == "Transmittance (%)":
            df['Raw_Intensity'] = df['Raw_Intensity'].clip(lower=0.001)
            df['Intensity'] = 2 - np.log10(df['Raw_Intensity'])
        else:
            df['Intensity'] = df['Raw_Intensity']

        # 2. ATR Correction
        if apply_atr:
            df['Intensity'] = df['Intensity'] * (df['Wavenumber'] / 1000)

        # 3. Smoothing
        data_len = len(df)
        actual_window = 3 # Fallback
        if data_len > 5:
            actual_window = s_val if s_val < data_len else (data_len - 1 if (data_len - 1) % 2 != 0 else data_len - 2)
            if actual_window >= 3:
                df['Intensity'] = savgol_filter(df['Intensity'], actual_window, 3)

        # 4. Baseline Min-Subtraction & Normalization
        df['Intensity'] = df['Intensity'] - df['Intensity'].min()
        max_val = df['Intensity'].max()
        if max_val > 0:
            df['Intensity'] = df['Intensity'] / max_val

        # 5. 2nd Derivative Calculation (ALWAYS calculate in background)
        d_window = max(3, actual_window - 2)
        if d_window % 2 == 0:  # scipy requires an odd window length
            d_window += 1 
            
        df['2nd_Deriv'] = savgol_filter(df['Intensity'], d_window, 3, deriv=2)

        return df
    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        return None
