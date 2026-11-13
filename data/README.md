# Input data

The three tables are not included, because the LUCAS terms of use do not allow
redistribution. Put them in this folder with these names:

| File | Source |
|---|---|
| `LUCAS-SOIL-2018.csv` | LUCAS 2018 topsoil data, [ESDAC](https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data) (free after accepting the terms of use) |
| `AEF_LUCAS_all_2018.csv` | `gee/01_extract_predictors.js` in Google Earth Engine |
| `TrendsEarth_SDG1531_LUCAS_points.csv` | `gee/02_extract_trendsearth.js` in Google Earth Engine |

For the Earth Engine scripts, upload the LUCAS 2018 point locations (with the
`POINTID` column) as an asset in your own project and change `LUCAS_ASSET` at
the top of each script.

`europe_countries.geojson` holds country outlines for the maps: Natural Earth 1:50m
admin-0 countries (public domain), clipped in the map projection (EPSG:3035) to a
rectangle slightly larger than the map frame, so that no cut line is visible.
