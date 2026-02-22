import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from matplotlib.lines import Line2D
from scipy.stats import pearsonr


def compute_summary_stats(da, name):
    """Compute key summary statistics for a DataArray."""
    print(f"\nComputing stats for {name}...")
    
    # These operations are dask-friendly
    mean_val    = float(da.mean().compute())
    max_val     = float(da.max().compute())
    
    # Quantiles — xarray supports this with dask in recent versions
    # If this is slow or errors, see the fallback approach below
    try:
        median_val = float(da.quantile(0.50).compute())
        p95_val    = float(da.quantile(0.95).compute())
        p99_val    = float(da.quantile(0.99).compute())
    except Exception:
        # Fallback: load into memory (only if you have enough RAM ~4-8GB)
        print(f"  Quantile via dask failed; loading into memory...")
        vals = da.values.ravel()
        vals = vals[~np.isnan(vals)]
        median_val = np.percentile(vals, 50)
        p95_val    = np.percentile(vals, 95)
        p99_val    = np.percentile(vals, 99)
    
    stats_dict = {
        'Mean': mean_val,
        'Median': median_val,
        '95th Percentile': p95_val,
        '99th Percentile': p99_val,
        'Maximum': max_val,
    }
    return stats_dict

def plot_spatial_map(data, lat, lon, title, cmap, vmin, vmax, cbar_label,
                     figsize=(10, 7), extend='both', contour_levels=None,
                     contour_colors=None, contour_styles=None, save_path=None):
    """
    Plot a 2D field on a geographic map of South Asia using cartopy.
    
    Parameters
    ----------
    data : 2D array (lat × lon)
    lat, lon : 1D coordinate arrays
    title : str
    cmap : colormap name or object
    vmin, vmax : color scale limits
    cbar_label : str for colorbar
    contour_levels : list of float — optional contour lines to overlay
    contour_colors : list of str — colors for each contour level
    contour_styles : list of str — linestyles for each contour level
    save_path : str or None — if set, saves the figure
    
    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': ccrs.PlateCarree()})
    
    mesh = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                         cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    # Optional contour lines (e.g., risk thresholds)
    if contour_levels is not None:
        if contour_colors is None:
            contour_colors = ['black'] * len(contour_levels)
        if contour_styles is None:
            contour_styles = ['--'] * len(contour_levels)
        for lvl, clr, sty in zip(contour_levels, contour_colors, contour_styles):
            ax.contour(lon, lat, data, levels=[lvl], colors=clr,
                       linewidths=1.5, linestyles=sty, transform=ccrs.PlateCarree())
    
    # Map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray', alpha=0.3)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_extent([float(lon.min()), float(lon.max()),
                   float(lat.min()), float(lat.max())], crs=ccrs.PlateCarree())
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.08,
                        shrink=0.8, extend=extend)
    cbar.set_label(cbar_label)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig, ax

def regrid_elevation_to_merra(elev_lats, elev_lons, elev_vals,
                               merra_lats, merra_lons,
                               dlat=0.25, dlon=0.3125):
    """
    Regrid high-resolution elevation data to the coarser MERRA-2 grid
    by averaging all elevation pixels within each MERRA-2 cell.
    
    Parameters
    ----------
    elev_lats, elev_lons : 1D arrays from ETOPO1
    elev_vals : 2D array (lat × lon) of elevation values
    merra_lats, merra_lons : 1D arrays from MERRA-2
    dlat : float — half of MERRA-2 latitude spacing (default 0.25°)
    dlon : float — half of MERRA-2 longitude spacing (default 0.3125°)
    
    Returns
    -------
    elev_on_merra : 2D array (merra_lat × merra_lon)
    """
    elev_on_merra = np.full((len(merra_lats), len(merra_lons)), np.nan)
    
    for i, mlat in enumerate(merra_lats):
        lat_mask = (elev_lats >= mlat - dlat) & (elev_lats < mlat + dlat)
        for j, mlon in enumerate(merra_lons):
            lon_mask = (elev_lons >= mlon - dlon) & (elev_lons < mlon + dlon)
            subset = elev_vals[np.ix_(lat_mask, lon_mask)]
            if subset.size > 0:
                elev_on_merra[i, j] = np.nanmean(subset)
    
    return elev_on_merra

def plot_elevation_scatter(elev_flat, mean_t2m_flat, mean_t2mwet_flat,
                           p95_t2mwet_flat, save_path=None):
    """
    Create 3-panel scatter plot: elevation vs each temperature metric.
    Returns Pearson correlations as a dict.
    """
    # Remove NaNs
    valid = ~(np.isnan(elev_flat) | np.isnan(mean_t2m_flat) | np.isnan(p95_t2mwet_flat))
    elev_v = elev_flat[valid]
    t2m_v = mean_t2m_flat[valid]
    t2mwet_v = mean_t2mwet_flat[valid]
    p95_v = p95_t2mwet_flat[valid]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Elevation vs Mean T2M
    ax = axes[0]
    sc = ax.scatter(elev_v, t2m_v, c=t2m_v, cmap='RdYlBu_r',
                    s=15, alpha=0.7, edgecolors='none')
    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Annual Mean T2M (°C)')
    ax.set_title('Elevation vs Mean Air Temp')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    plt.colorbar(sc, ax=ax, label='°C', shrink=0.8)
    
    # Elevation vs Mean T2MWET
    ax = axes[1]
    sc = ax.scatter(elev_v, t2mwet_v, c=t2mwet_v, cmap='RdYlBu_r',
                    s=15, alpha=0.7, edgecolors='none')
    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Annual Mean T2MWET (°C)')
    ax.set_title('Elevation vs Mean Wet-Bulb Temp')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    plt.colorbar(sc, ax=ax, label='°C', shrink=0.8)
    
    # Elevation vs P95 T2MWET
    ax = axes[2]
    sc = ax.scatter(elev_v, p95_v, c=p95_v, cmap='YlOrRd',
                    s=15, alpha=0.7, edgecolors='none')
    ax.set_xlabel('Elevation (m)')
    ax.set_ylabel('Annual P95 T2MWET (°C)')
    ax.set_title('Elevation vs P95 Wet-Bulb Temp')
    ax.axhline(25, color='orange', lw=1.5, ls='--', label='25°C Moderate')
    ax.axhline(28, color='red', lw=1.5, ls='--', label='28°C High')
    ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label='°C', shrink=0.8)
    
    plt.suptitle('Relationship Between Elevation and Heat Stress Metrics',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    # Correlations
    r_t2m, _ = pearsonr(elev_v, t2m_v)
    r_t2mwet, _ = pearsonr(elev_v, t2mwet_v)
    r_p95, _ = pearsonr(elev_v, p95_v)
    
    correlations = {
        'Mean T2M': r_t2m,
        'Mean T2MWET': r_t2mwet,
        'P95 T2MWET': r_p95,
    }
    return correlations, (elev_v, t2m_v, t2mwet_v, p95_v)

def plot_combined_panel(annual_mean_t2m, annual_mean_t2mwet, annual_p95_t2mwet,
                        ds_lat, ds_lon, elevation, elev_lat, elev_lon,
                        save_path=None):
    """
    Create a 2×2 panel combining all four Q2 maps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    panels = [
        (annual_mean_t2m, ds_lat, ds_lon,
         'Annual Mean T2M (°C)', 'RdYlBu_r', -10, 35),
        (annual_mean_t2mwet, ds_lat, ds_lon,
         'Annual Mean T2MWET (°C)', 'RdYlBu_r', -15, 28),
        (annual_p95_t2mwet, ds_lat, ds_lon,
         'Annual P95 T2MWET (°C)', 'YlOrRd', 5, 30),
        (elevation, elev_lat, elev_lon,
         'Elevation (m)', 'terrain', -500, 6000),
    ]
    
    for idx, (data, lats, lons, title, cmap, vmin, vmax) in enumerate(panels):
        ax = axes.flat[idx]
        mesh = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        ax.set_extent([60, 100, 5, 35], crs=ccrs.PlateCarree())
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        if idx % 2 != 0:
            gl.left_labels = False
        if idx < 2:
            gl.bottom_labels = False
        
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.06,
                            shrink=0.85, extend='both')
        cbar.set_label(title)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.suptitle('Spatial Structure of Heat Stress and Topographic Context — South Asia, 2024',
                 fontsize=15, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig

def print_q2_summary(annual_p95_t2mwet, annual_mean_t2m, elev_on_merra,
                     ds_lat, ds_lon, elev_flat, p95_flat):
    """
    Print quantitative summary statistics for Q2 interpretation.
    """
    p95_data = annual_p95_t2mwet
    
    cells_above_25 = np.sum(p95_data >= 25)
    cells_above_28 = np.sum(p95_data >= 28)
    total_cells = np.sum(~np.isnan(p95_data))
    
    print(f"\nGrid cells where annual P95 T2MWET:")
    print(f"  >= 25°C (Moderate): {cells_above_25} / {total_cells} "
          f"({100*cells_above_25/total_cells:.1f}%)")
    print(f"  >= 28°C (High):     {cells_above_28} / {total_cells} "
          f"({100*cells_above_28/total_cells:.1f}%)")
    
    # Hottest grid cell
    max_idx = np.unravel_index(np.nanargmax(p95_data), p95_data.shape)
    print(f"\nHottest P95 T2MWET grid cell:")
    print(f"  Location: {ds_lat[max_idx[0]]:.1f}°N, {ds_lon[max_idx[1]]:.1f}°E")
    print(f"  P95 T2MWET: {p95_data[max_idx]:.2f}°C")
    print(f"  Mean T2M:   {annual_mean_t2m[max_idx]:.2f}°C")
    print(f"  Elevation:  {elev_on_merra[max_idx]:.0f} m")
    
    # Low vs high elevation
    valid = ~np.isnan(elev_flat) & ~np.isnan(p95_flat)
    ev = elev_flat[valid]
    pv = p95_flat[valid]
    
    low_mask = ev < 100
    high_mask = ev > 2000
    
    if np.any(low_mask):
        print(f"\nLow-elevation cells (<100m): n={np.sum(low_mask)}")
        print(f"  Mean P95 T2MWET: {np.mean(pv[low_mask]):.2f}°C")
        print(f"  Max P95 T2MWET:  {np.max(pv[low_mask]):.2f}°C")
    
    if np.any(high_mask):
        print(f"\nHigh-elevation cells (>2000m): n={np.sum(high_mask)}")
        print(f"  Mean P95 T2MWET: {np.mean(pv[high_mask]):.2f}°C")
        print(f"  Max P95 T2MWET:  {np.max(pv[high_mask]):.2f}°C")

def load_and_aggregate_population(pop_path, merra_lats, merra_lons,
                                   dlat=0.25, dlon=0.3125):
    """
    Load GPWv4 population density raster and aggregate to the MERRA-2 grid.
    
    Approach:
    1. Read population density (people/km²) from GeoTIFF
    2. For each MERRA-2 cell, identify all population pixels within the cell boundaries
    3. Compute the mean density within the cell, then multiply by the cell's area
       to get total population count per MERRA-2 cell
    
    This is equivalent to summing (density × pixel_area) for all fine pixels in each
    coarse cell, which preserves total population.
    
    Parameters
    ----------
    pop_path : str — path to the GPWv4 GeoTIFF
    merra_lats, merra_lons : 1D arrays from MERRA-2 grid
    dlat : float — half of MERRA-2 latitude spacing (0.25°)
    dlon : float — half of MERRA-2 longitude spacing (0.3125°)
    
    Returns
    -------
    pop_on_merra : 2D array (merra_lat × merra_lon) — total population per cell
    pop_density_on_merra : 2D array — mean density per cell (people/km²)
    """
    import rasterio
    
    print("Loading population density raster...")
    with rasterio.open(pop_path) as src:
        pop_data = src.read(1).astype(np.float64)
        pop_transform = src.transform
        pop_res = src.res  # (lat_res, lon_res) in degrees
    
    # Mask invalid values (nodata = -9999, also catch large negatives)
    pop_data[pop_data < 0] = 0.0
    pop_data[pop_data > 1e6] = 0.0  # cap unreasonable values
    
    # Build lat/lon arrays for the population raster
    nrows, ncols = pop_data.shape
    # GeoTIFF convention: top-left origin, rows go south
    pop_lons = np.array([pop_transform[2] + (j + 0.5) * pop_transform[0] for j in range(ncols)])
    pop_lats = np.array([pop_transform[5] + (i + 0.5) * pop_transform[4] for i in range(nrows)])
    
    # Pixel area in km² (varies with latitude)
    # Approximate: dx = lon_res * 111.32 * cos(lat), dy = lat_res * 110.574
    lat_res_deg = abs(pop_res[0])
    lon_res_deg = abs(pop_res[1])
    
    pop_on_merra = np.full((len(merra_lats), len(merra_lons)), 0.0)
    pop_density_on_merra = np.full((len(merra_lats), len(merra_lons)), np.nan)
    
    print("Aggregating population to MERRA-2 grid...")
    for i, mlat in enumerate(merra_lats):
        # Row indices in pop raster for this MERRA-2 cell
        lat_lo, lat_hi = mlat - dlat, mlat + dlat
        row_mask = (pop_lats >= lat_lo) & (pop_lats < lat_hi)
        
        if not np.any(row_mask):
            continue
            
        row_indices = np.where(row_mask)[0]
        
        for j, mlon in enumerate(merra_lons):
            lon_lo, lon_hi = mlon - dlon, mlon + dlon
            col_mask = (pop_lons >= lon_lo) & (pop_lons < lon_hi)
            
            if not np.any(col_mask):
                continue
            
            col_indices = np.where(col_mask)[0]
            
            # Extract density values for pixels in this MERRA-2 cell
            subset = pop_data[np.ix_(row_indices, col_indices)]
            
            if subset.size == 0:
                continue
            
            # Compute total population:
            # Sum of (density_per_pixel × area_per_pixel)
            # Pixel area varies with latitude
            center_lat = mlat
            dy_km = lat_res_deg * 110.574  # km per degree latitude
            dx_km = lon_res_deg * 111.32 * np.cos(np.radians(center_lat))
            pixel_area_km2 = dx_km * dy_km
            
            total_pop = np.sum(subset) * pixel_area_km2
            mean_density = np.mean(subset)
            
            pop_on_merra[i, j] = total_pop
            pop_density_on_merra[i, j] = mean_density
    
    total_pop_region = np.sum(pop_on_merra)
    print(f"  Total population on MERRA-2 grid: {total_pop_region/1e9:.2f} billion")
    print(f"  Max cell population: {np.max(pop_on_merra)/1e6:.2f} million")
    print(f"  Non-zero cells: {np.sum(pop_on_merra > 0)}")
    
    return pop_on_merra, pop_density_on_merra

def compute_person_hours(t2mwet_c, pop_on_merra, high_thresh=28,
                          peak_month=None):
    """
    Compute person-hours of High heat-risk exposure (T2MWET ≥ 28°C).
    
    Parameters
    ----------
    t2mwet_c : xarray DataArray (time, lat, lon) in °C
    pop_on_merra : 2D array (lat × lon) — population per MERRA-2 cell
    high_thresh : float — High risk threshold (default 28°C)
    peak_month : int or None — if set, also compute person-hours for this month
    
    Returns
    -------
    dict with keys:
        'annual_hours': 2D array — total hours ≥28°C per cell (full year)
        'annual_person_hours': 2D array — person-hours per cell (full year)
        'peak_hours': 2D array — hours ≥28°C for peak month (if peak_month set)
        'peak_person_hours': 2D array — person-hours for peak month
        'total_annual_ph': float — grand total person-hours (year)
        'total_peak_ph': float — grand total person-hours (peak month)
    """
    print(f"Computing hours with T2MWET ≥ {high_thresh}°C...")
    
    # Annual: count hours above threshold at each grid cell
    high_mask = (t2mwet_c >= high_thresh)
    annual_hours = high_mask.sum(dim='time').values.astype(float)
    
    print(f"  Annual hours range: {annual_hours.min():.0f} to {annual_hours.max():.0f}")
    print(f"  Grid cells with any High-risk hours: {np.sum(annual_hours > 0)}")
    
    # Person-hours = population × hours
    annual_person_hours = pop_on_merra * annual_hours
    total_annual_ph = np.sum(annual_person_hours)
    
    print(f"  Total annual person-hours (High risk): {total_annual_ph:.2e}")
    
    result = {
        'annual_hours': annual_hours,
        'annual_person_hours': annual_person_hours,
        'total_annual_ph': total_annual_ph,
    }
    
    # Peak month computation
    if peak_month is not None:
        print(f"\nComputing for peak month ({peak_month})...")
        month_mask = t2mwet_c.time.dt.month == peak_month
        t2mwet_month = t2mwet_c.sel(time=month_mask)
        
        peak_hours = (t2mwet_month >= high_thresh).sum(dim='time').values.astype(float)
        peak_person_hours = pop_on_merra * peak_hours
        total_peak_ph = np.sum(peak_person_hours)
        
        print(f"  Peak month hours range: {peak_hours.min():.0f} to {peak_hours.max():.0f}")
        print(f"  Total peak-month person-hours: {total_peak_ph:.2e}")
        
        result['peak_hours'] = peak_hours
        result['peak_person_hours'] = peak_person_hours
        result['total_peak_ph'] = total_peak_ph
    
    return result

def plot_person_hours_maps(ph_result, pop_on_merra, lats, lons,
                           peak_month_name=None, save_path=None):
    """
    Visualize person-hours and supporting data in a 2×2 panel:
    - Population per grid cell
    - Annual hours ≥28°C
    - Annual person-hours
    - Peak-month person-hours (if available)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    panels = [
        (np.log10(np.clip(pop_on_merra, 1, None)),
         'Population per Grid Cell (log₁₀)',
         'YlOrRd', 0, 8, 'log₁₀(population)'),
        (ph_result['annual_hours'],
         'Annual Hours with T2MWET ≥ 28°C',
         'hot_r', 0, None, 'Hours'),
        (np.log10(np.clip(ph_result['annual_person_hours'], 1, None)),
         'Annual Person-Hours of High Risk (log₁₀)',
         'magma_r', 0, None, 'log₁₀(person-hours)'),
        None,  # placeholder for peak month
    ]
    
    if 'peak_person_hours' in ph_result and peak_month_name:
        panels[3] = (
            np.log10(np.clip(ph_result['peak_person_hours'], 1, None)),
            f'Person-Hours of High Risk — {peak_month_name} (log₁₀)',
            'magma_r', 0, None, 'log₁₀(person-hours)')
    else:
        # Show raw hours for peak month or population density
        panels[3] = (
            np.log10(np.clip(ph_result.get('peak_person_hours',
                                            ph_result['annual_person_hours']), 1, None)),
            'Peak Month Person-Hours (log₁₀)',
            'magma_r', 0, None, 'log₁₀(person-hours)')
    
    for idx, (data, title, cmap, vmin, vmax, clabel) in enumerate(panels):
        ax = axes.flat[idx]
        
        if vmax is None:
            vmax = np.nanmax(data[data > 0]) if np.any(data > 0) else 1
        
        mesh = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        ax.set_extent([60, 100, 5, 35], crs=ccrs.PlateCarree())
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        if idx % 2 != 0:
            gl.left_labels = False
        if idx < 2:
            gl.bottom_labels = False
        
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                            pad=0.06, shrink=0.85, extend='max')
        cbar.set_label(clabel)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.suptitle('Population Exposure to High Heat Risk (T2MWET ≥ 28°C) — South Asia, 2024',
                 fontsize=15, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def identify_top_exposure_cells(ph_result, pop_on_merra, lats, lons, n=10):
    """
    Identify and print the grid cells contributing most to total person-hours.
    
    Returns a DataFrame of the top-N cells.
    """
    annual_ph = ph_result['annual_person_hours']
    annual_hrs = ph_result['annual_hours']
    
    records = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            if annual_ph[i, j] > 0:
                records.append({
                    'lat': lats[i],
                    'lon': lons[j],
                    'population': pop_on_merra[i, j],
                    'hours_ge28': annual_hrs[i, j],
                    'person_hours': annual_ph[i, j],
                })
    
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No grid cells with person-hours > 0")
        return df
    
    df = df.sort_values('person_hours', ascending=False).reset_index(drop=True)
    
    # Add cumulative percentage
    total_ph = df['person_hours'].sum()
    df['cumulative_pct'] = 100 * df['person_hours'].cumsum() / total_ph
    
    print(f"\n{'='*80}")
    print(f"Top {n} Grid Cells by Annual Person-Hours of High Heat Risk (≥28°C)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Lat':>6} {'Lon':>8} {'Population':>14} "
          f"{'Hours≥28°C':>12} {'Person-Hours':>16} {'Cumul %':>10}")
    print(f"{'-'*80}")
    
    for idx, row in df.head(n).iterrows():
        print(f"{idx+1:<6} {row['lat']:>6.1f} {row['lon']:>8.1f} "
              f"{row['population']:>14,.0f} {row['hours_ge28']:>12.0f} "
              f"{row['person_hours']:>16,.0f} {row['cumulative_pct']:>9.1f}%")
    
    print(f"{'-'*80}")
    print(f"Total person-hours (all cells): {total_ph:,.0f}")
    print(f"Cells with any exposure: {len(df)}")
    print(f"Top {n} cells account for: {df.head(n)['cumulative_pct'].iloc[-1]:.1f}% of total")
    print(f"{'='*80}")
    
    return df

def plot_top_cells_bar(df_top, n=10, save_path=None):
    """
    Horizontal bar chart of top-N cells by person-hours, colored by population.
    """
    top = df_top.head(n).copy()
    top['label'] = [f"{row['lat']:.1f}°N, {row['lon']:.1f}°E" 
                    for _, row in top.iterrows()]
    top = top.iloc[::-1]  # reverse for horizontal bars (top at top)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.YlOrRd(top['population'] / top['population'].max())
    bars = ax.barh(top['label'], top['person_hours'], color=colors, edgecolor='gray', lw=0.5)
    
    ax.set_xlabel('Annual Person-Hours of High Heat Risk (≥28°C)')
    ax.set_title(f'Top {n} Grid Cells by Person-Hours Exposure — 2024', fontweight='bold')
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
    
    # Add population annotation
    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height()/2,
                f'pop: {row["population"]/1e6:.1f}M, {row["hours_ge28"]:.0f}h',
                va='center', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()