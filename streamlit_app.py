with tab2:
        if calc_deriv:
            st.markdown("**2nd Derivative Spectra:** Used to find hidden peaks. Minima (valleys) correspond to peak maxima in the original spectrum.")
            fig_deriv = go.Figure()
            for name in master['File'].unique():
                df = spectra[name]
                
                # Safety check to prevent KeyError on legacy data in memory
                if '2nd_Deriv' in df.columns:
                    fig_deriv.add_trace(go.Scatter(
                        x=df['Wavenumber'], y=df['2nd_Deriv'],
                        mode='lines', line=dict(width=1.5), name=name
                    ))
                else:
                    st.warning(f"⚠️ {name} is missing derivative data. Please click 'Clear Memory' and re-upload.")
            
            fig_deriv.update_layout(
                template="simple_white", height=600,
                xaxis=dict(title="<b>Wavenumber (cm⁻¹)</b>", range=[2000, 600], **FTIR_STYLE),
                yaxis=dict(title="<b>d²A/dν²</b>", **FTIR_STYLE)
            )
            fig_deriv.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_deriv, use_container_width=True)
        else:
            st.info("Check 'Calculate 2nd Derivative' in the sidebar to view peak deconvolution.")
