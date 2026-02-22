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