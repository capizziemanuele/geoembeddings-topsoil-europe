// Sample the Trends.Earth SDG 15.3.1 layers (Zvoleff et al. 2025, v1.2)
// at the LUCAS 2018 points. Writes TrendsEarth_SDG1531_LUCAS_points.csv.

var LUCAS_ASSET = 'projects/YOUR_PROJECT/assets/LUCAS_2018';
var points = ee.FeatureCollection(LUCAS_ASSET).select(['POINTID']);

var sdg = ee.Image('projects/gef-land-degradation/assets/TrendsEarth_SDG15-3-1_2000-2023_Trends-Earth');
print('Bands in the asset', sdg.bandNames());

// Note: the soil carbon band is called "degradation" but holds a
// percentage change (continuous values), not a class.
var layers = sdg.select([
  'land_productivity_dynamics_using_trends_earth_for_2008_2023',
  'land_cover_degradation_for_2015_2023',
  'soil_organic_carbon_degradation_for_2015_2023',
  'sdg_indicator_15_3_1_status_in_2023_compared_to_2000_2015_baseline'
]).rename(['LPD_2008_2023', 'LC_DEGRADATION_2015_2023',
           'SOC_PCT_CHANGE_2015_2023', 'STATUS_2023']);

var samples = layers.sampleRegions({collection: points, properties: ['POINTID'],
                                    scale: 250, tileScale: 4, geometries: false})
  .filter(ee.Filter.notNull(['STATUS_2023']));

print('Sampled points', samples.size());

Export.table.toDrive({
  collection: samples,
  description: 'TrendsEarth_SDG1531_LUCAS_points',
  fileNamePrefix: 'TrendsEarth_SDG1531_LUCAS_points',
  fileFormat: 'CSV'
});
