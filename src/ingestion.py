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

# =====================================================================
# Q2 ADDITIONS — paste these into your ingestion.py
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from scipy.stats import pearsonr


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


# =====================================================================
# Q3 ADDITIONS — paste these into your ingestion.py
# =====================================================================
# Additional imports needed at top of ingestion.py (if not already there):
#   import pandas as pd
#   import xarray as xr
#   import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import calendar


# ---- Q3 PART A: Regional Seasonal Analysis ----

def compute_daily_metrics(t2mwet_c, moderate_thresh=25, high_thresh=28, extreme_thresh=30):
    """
    Compute daily heat-stress metrics from hourly wet-bulb temperature.
    
    Parameters
    ----------
    t2mwet_c : xarray DataArray (time, lat, lon) in °C
    
    Returns
    -------
    dict of xarray DataArrays with keys:
        'daily_mean', 'daily_max',
        'hours_moderate', 'hours_high', 'hours_extreme',
        'hours_moderate_plus', 'hours_high_plus'
    """
    print("Computing daily mean T2MWET...")
    daily_mean = t2mwet_c.resample(time='1D').mean()
    
    print("Computing daily max T2MWET...")
    daily_max = t2mwet_c.resample(time='1D').max()
    
    print("Computing daily hours in Moderate band (25-28°C)...")
    moderate_mask = (t2mwet_c >= moderate_thresh) & (t2mwet_c < high_thresh)
    hours_moderate = moderate_mask.resample(time='1D').sum()
    
    print("Computing daily hours in High band (28-30°C)...")
    high_mask = (t2mwet_c >= high_thresh) & (t2mwet_c < extreme_thresh)
    hours_high = high_mask.resample(time='1D').sum()
    
    print("Computing daily hours in Extreme band (≥30°C)...")
    extreme_mask = (t2mwet_c >= extreme_thresh)
    hours_extreme = extreme_mask.resample(time='1D').sum()
    
    # Cumulative thresholds (useful for reporting)
    hours_moderate_plus = (t2mwet_c >= moderate_thresh).resample(time='1D').sum()
    hours_high_plus = (t2mwet_c >= high_thresh).resample(time='1D').sum()
    
    print("Done — daily metrics computed.")
    
    return {
        'daily_mean': daily_mean,
        'daily_max': daily_max,
        'hours_moderate': hours_moderate,
        'hours_high': hours_high,
        'hours_extreme': hours_extreme,
        'hours_moderate_plus': hours_moderate_plus,
        'hours_high_plus': hours_high_plus,
    }


def compute_monthly_summary(daily_metrics):
    """
    Aggregate daily metrics to monthly summaries (spatial mean across region).
    
    Parameters
    ----------
    daily_metrics : dict from compute_daily_metrics()
    
    Returns
    -------
    pandas DataFrame with monthly rows and metric columns
    """
    # Spatial mean across all grid cells, then group by month
    records = []
    
    for month in range(1, 13):
        month_mask = daily_metrics['daily_mean'].time.dt.month == month
        
        # Mean across space and time for this month
        mean_twet = float(daily_metrics['daily_mean'].sel(time=month_mask).mean())
        max_twet_mean = float(daily_metrics['daily_max'].sel(time=month_mask).mean())
        max_twet_peak = float(daily_metrics['daily_max'].sel(time=month_mask).max())
        
        # Mean daily hours (averaged over space and days in month)
        hrs_mod = float(daily_metrics['hours_moderate_plus'].sel(time=month_mask).mean())
        hrs_high = float(daily_metrics['hours_high_plus'].sel(time=month_mask).mean())
        
        records.append({
            'Month': calendar.month_abbr[month],
            'Month_num': month,
            'Mean Daily T2MWET (°C)': mean_twet,
            'Mean Daily Max T2MWET (°C)': max_twet_mean,
            'Peak Daily Max T2MWET (°C)': max_twet_peak,
            'Mean Daily Hours ≥25°C': hrs_mod,
            'Mean Daily Hours ≥28°C': hrs_high,
        })
    
    return pd.DataFrame(records)


def plot_monthly_boxplots(daily_metrics, save_path=None):
    """
    Monthly boxplots of region-wide daily mean and daily max T2MWET,
    with risk threshold lines.
    """
    # Spatial average per day
    daily_mean_spatial = daily_metrics['daily_mean'].mean(dim=['lat', 'lon'])
    daily_max_spatial = daily_metrics['daily_max'].mean(dim=['lat', 'lon'])
    
    months = daily_mean_spatial.time.dt.month.values
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Daily Mean T2MWET ---
    ax = axes[0]
    data_by_month = [daily_mean_spatial.values[months == m] for m in range(1, 13)]
    bp = ax.boxplot(data_by_month, labels=[calendar.month_abbr[m] for m in range(1, 13)],
                    patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor='#a8d5e2', alpha=0.7),
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(markersize=3))
    ax.axhline(25, color='orange', ls='--', lw=1.5, label='25°C Moderate')
    ax.axhline(28, color='red', ls='--', lw=1.5, label='28°C High')
    ax.set_ylabel('Region-Mean Daily Mean T2MWET (°C)')
    ax.set_title('Daily Mean Wet-Bulb Temperature by Month')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    
    # --- Daily Max T2MWET ---
    ax = axes[1]
    data_by_month = [daily_max_spatial.values[months == m] for m in range(1, 13)]
    bp = ax.boxplot(data_by_month, labels=[calendar.month_abbr[m] for m in range(1, 13)],
                    patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor='#f4a582', alpha=0.7),
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(markersize=3))
    ax.axhline(25, color='orange', ls='--', lw=1.5, label='25°C Moderate')
    ax.axhline(28, color='red', ls='--', lw=1.5, label='28°C High')
    ax.set_ylabel('Region-Mean Daily Max T2MWET (°C)')
    ax.set_title('Daily Max Wet-Bulb Temperature by Month')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Seasonal Distribution of Wet-Bulb Temperature Metrics — South Asia, 2024',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_monthly_risk_hours(daily_metrics, save_path=None):
    """
    Bar chart showing mean daily hours at Moderate+ and High+ risk by month.
    """
    # Spatial average of hours per day, grouped by month
    hrs_mod = daily_metrics['hours_moderate_plus'].mean(dim=['lat', 'lon'])
    hrs_high = daily_metrics['hours_high_plus'].mean(dim=['lat', 'lon'])
    
    months_arr = hrs_mod.time.dt.month.values
    
    monthly_mod = [float(hrs_mod.values[months_arr == m].mean()) for m in range(1, 13)]
    monthly_high = [float(hrs_high.values[months_arr == m].mean()) for m in range(1, 13)]
    
    month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
    x = np.arange(12)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, monthly_mod, width, label='Hours ≥25°C (Moderate+)',
                   color='#FFA500', alpha=0.8)
    bars2 = ax.bar(x + width/2, monthly_high, width, label='Hours ≥28°C (High+)',
                   color='#FF4500', alpha=0.8)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Daily Hours in Risk Band (region avg)')
    ax.set_title('Mean Daily Hours of Heat-Risk Exposure by Month — South Asia, 2024')
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


# ---- Q3 PART B: City-Level Analysis ----

# Four representative cities spanning different geographies
CITIES = {
    'Delhi':   {'lat': 28.61, 'lon': 77.21, 'desc': 'Inland, Indo-Gangetic Plain'},
    'Mumbai':  {'lat': 19.08, 'lon': 72.88, 'desc': 'Coastal, Arabian Sea'},
    'Dhaka':   {'lat': 23.81, 'lon': 90.41, 'desc': 'Riverine delta, Bay of Bengal'},
    'Karachi': {'lat': 24.86, 'lon': 67.01, 'desc': 'Coastal arid, Arabian Sea'},
}


def extract_city_timeseries(ds_t2m_c, ds_t2mwet_c, cities=None):
    """
    Extract nearest-grid-cell time series for each city.
    
    Parameters
    ----------
    ds_t2m_c : xarray DataArray of T2M in °C
    ds_t2mwet_c : xarray DataArray of T2MWET in °C
    cities : dict {name: {lat, lon, desc}} or None for defaults
    
    Returns
    -------
    dict {city_name: {'t2m': Series, 't2mwet': Series, 'lat_actual': float, 
                       'lon_actual': float, 'desc': str}}
    """
    if cities is None:
        cities = CITIES
    
    city_data = {}
    for name, info in cities.items():
        t2m_ts = ds_t2m_c.sel(lat=info['lat'], lon=info['lon'], method='nearest')
        t2mwet_ts = ds_t2mwet_c.sel(lat=info['lat'], lon=info['lon'], method='nearest')
        
        actual_lat = float(t2m_ts.lat)
        actual_lon = float(t2m_ts.lon)
        
        city_data[name] = {
            't2m': t2m_ts,
            't2mwet': t2mwet_ts,
            'lat_actual': actual_lat,
            'lon_actual': actual_lon,
            'desc': info.get('desc', ''),
        }
        print(f"  {name}: requested ({info['lat']:.2f}°N, {info['lon']:.2f}°E) "
              f"→ nearest ({actual_lat:.1f}°N, {actual_lon:.2f}°E)")
    
    return city_data


def plot_city_timeseries(city_data, save_path=None):
    """
    Create a 4-panel figure showing daily mean T2M and T2MWET for each city,
    with risk threshold bands highlighted.
    
    Parameters
    ----------
    city_data : dict from extract_city_timeseries()
    """
    cities = list(city_data.keys())
    n_cities = len(cities)
    
    fig, axes = plt.subplots(n_cities, 1, figsize=(14, 3.5 * n_cities), sharex=True)
    if n_cities == 1:
        axes = [axes]
    
    for i, (city, data) in enumerate(city_data.items()):
        ax = axes[i]
        
        # Resample to daily mean for smoother time series
        t2m_daily = data['t2m'].resample(time='1D').mean()
        t2mwet_daily = data['t2mwet'].resample(time='1D').mean()
        
        times = pd.to_datetime(t2m_daily.time.values)
        
        # Plot air temp and wet-bulb
        ax.plot(times, t2m_daily.values, color='#e07b54', alpha=0.7,
                lw=0.8, label='T2M (daily mean)')
        ax.plot(times, t2mwet_daily.values, color='#5b8fbc', alpha=0.9,
                lw=1.0, label='T2MWET (daily mean)')
        
        # Risk threshold shading across full time axis
        ax.axhspan(25, 28, alpha=0.12, color='orange', label='Moderate (25-28°C)')
        ax.axhspan(28, 30, alpha=0.12, color='red', label='High (28-30°C)')
        ax.axhspan(30, 50, alpha=0.12, color='darkred', label='Extreme (≥30°C)')
        
        # Threshold lines
        ax.axhline(25, color='orange', ls=':', lw=0.8, alpha=0.7)
        ax.axhline(28, color='red', ls=':', lw=0.8, alpha=0.7)
        
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'{city} ({data["desc"]}) — '
                     f'{data["lat_actual"]:.1f}°N, {data["lon_actual"]:.1f}°E',
                     fontsize=12, fontweight='bold')
        
        if i == 0:
            ax.legend(fontsize=8, loc='upper right', ncol=3)
        
        ax.set_ylim(bottom=min(0, float(t2mwet_daily.min()) - 2))
        ax.grid(axis='y', alpha=0.2)
    
    # Format x-axis
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].set_xlabel('Month (2024)')
    
    plt.suptitle('City-Level Seasonal Temperature and Wet-Bulb Profiles — 2024',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_city_risk_hours(city_data, moderate_thresh=25, high_thresh=28, save_path=None):
    """
    For each city, compute and plot monthly hours in Moderate+ and High+ risk.
    4-panel bar charts.
    """
    cities = list(city_data.keys())
    n_cities = len(cities)
    
    fig, axes = plt.subplots(n_cities, 1, figsize=(12, 3 * n_cities), sharex=True)
    if n_cities == 1:
        axes = [axes]
    
    month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
    x = np.arange(12)
    width = 0.35
    
    for i, (city, data) in enumerate(city_data.items()):
        ax = axes[i]
        
        t2mwet = data['t2mwet']
        months = t2mwet.time.dt.month.values
        
        # Total hours in each risk band per month
        mod_plus = (t2mwet.values >= moderate_thresh)
        high_plus = (t2mwet.values >= high_thresh)
        
        monthly_mod = [mod_plus[months == m].sum() for m in range(1, 13)]
        monthly_high = [high_plus[months == m].sum() for m in range(1, 13)]
        
        ax.bar(x - width/2, monthly_mod, width, label='Hours ≥25°C',
               color='#FFA500', alpha=0.8)
        ax.bar(x + width/2, monthly_high, width, label='Hours ≥28°C',
               color='#FF4500', alpha=0.8)
        
        ax.set_ylabel('Total Hours')
        ax.set_title(f'{city}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        
        if i == 0:
            ax.legend(fontsize=9)
    
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(month_labels)
    axes[-1].set_xlabel('Month (2024)')
    
    plt.suptitle('Monthly Hours of Heat-Risk Exposure by City — 2024',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def print_city_summary(city_data, moderate_thresh=25, high_thresh=28):
    """
    Print summary statistics for each city.
    """
    print(f"\n{'='*70}")
    print(f"{'City':<12} {'Mean T2M':>10} {'Mean TWET':>10} {'P95 TWET':>10} "
          f"{'Max TWET':>10} {'Hrs≥25°C':>10} {'Hrs≥28°C':>10}")
    print(f"{'='*70}")
    
    for city, data in city_data.items():
        t2m = data['t2m'].values
        tw = data['t2mwet'].values
        
        mean_t2m = np.nanmean(t2m)
        mean_tw = np.nanmean(tw)
        p95_tw = np.nanpercentile(tw, 95)
        max_tw = np.nanmax(tw)
        hrs_mod = np.sum(tw >= moderate_thresh)
        hrs_high = np.sum(tw >= high_thresh)
        
        print(f"{city:<12} {mean_t2m:>9.1f}° {mean_tw:>9.1f}° {p95_tw:>9.1f}° "
              f"{max_tw:>9.1f}° {hrs_mod:>10,} {hrs_high:>10,}")
    
    print(f"{'='*70}")