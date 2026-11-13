// Sample AlphaEarth embeddings, Sentinel-2 indices and SoilGrids 2.0
// at the LUCAS 2018 points. Writes AEF_LUCAS_all_2018.csv to Google Drive.
//
// Before running: upload the LUCAS 2018 points (with a POINTID column)
// as an Earth Engine asset and change LUCAS_ASSET below.

var LUCAS_ASSET = 'projects/YOUR_PROJECT/assets/LUCAS_2018';
var YEAR = 2018;
var SCALE = 10;          // metres
var CS_THRESH = 0.50;    // Cloud Score+ threshold
var CLOUDY_MAX = 70;     // maximum cloudy pixel percentage per scene

var points = ee.FeatureCollection(LUCAS_ASSET).select(['POINTID']);
var region = points.geometry().bounds();
var start = YEAR + '-01-01';
var end = (YEAR + 1) + '-01-01';

// AlphaEarth annual embeddings: 64 bands, A00 to A63
var embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterBounds(region)
  .filterDate(start, end)
  .mosaic();

// Sentinel-2 annual median, cloud masked with Cloud Score+
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(start, end)
  .filterBounds(region)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_MAX))
  .linkCollection(csPlus, ['cs_cdf'])
  .map(function (img) {
    return img.updateMask(img.select('cs_cdf').gte(CS_THRESH))
              .select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']);
  })
  .median()
  .divide(10000);

var b = {B: s2.select('B2'), G: s2.select('B3'), R: s2.select('B4'), RE: s2.select('B5'),
         N: s2.select('B8'), S1: s2.select('B11'), S2: s2.select('B12')};

// The eleven spectral indices (formulas in Supplementary Table S1)
var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');
var indices = ee.Image.cat([
  ndvi,
  s2.expression('2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1)', b).clamp(-1, 2).rename('EVI'),
  s2.expression('1.5 * (N - R) / (N + R + 0.5)', b).rename('SAVI'),
  ndvi.multiply(b.N).rename('NIRv'),
  s2.normalizedDifference(['B8', 'B5']).rename('NDRE'),
  s2.normalizedDifference(['B3', 'B8']).rename('NDWI'),
  s2.normalizedDifference(['B8', 'B11']).rename('NDMI'),
  s2.normalizedDifference(['B11', 'B12']).rename('NBR2'),
  s2.expression('((S1 + R) - (N + B)) / ((S1 + R) + (N + B))', b).rename('BSI'),
  s2.expression('(R - B) / (R + B)', b).rename('SOIL_CI'),
  s2.expression('S1 / S2', b).rename('SOIL_SWIRR')
]);

// SoilGrids 2.0 mean for 0-20 cm, to match the LUCAS sample:
// (5 * [0-5] + 10 * [5-15] + 5 * [15-30]) / 20
// Only the upper 5 cm of the 15-30 cm layer falls inside 0-20 cm.
// Carbon stock exists only for 0-30 cm and is used as it is.
function soilgrids020(prop, name) {
  var img = ee.Image('projects/soilgrids-isric/' + prop + '_mean');
  if (prop === 'ocs') {
    return img.select('ocs_0-30cm_mean').rename(name);
  }
  return img.select(prop + '_0-5cm_mean').multiply(5)
    .add(img.select(prop + '_5-15cm_mean').multiply(10))
    .add(img.select(prop + '_15-30cm_mean').multiply(5))
    .divide(20)
    .rename(name);
}

// Native units: SOC dg/kg, N cg/kg, pH x10, texture g/kg, CEC mmol(c)/kg,
// bulk density cg/cm3, coarse fragments cm3/dm3, OCD hg/m3, OCS t/ha
var soilgrids = ee.Image.cat([
  soilgrids020('soc', 'SG_SOC'), soilgrids020('nitrogen', 'SG_N'),
  soilgrids020('phh2o', 'SG_PH'), soilgrids020('clay', 'SG_CLAY'),
  soilgrids020('sand', 'SG_SAND'), soilgrids020('silt', 'SG_SILT'),
  soilgrids020('cec', 'SG_CEC'), soilgrids020('bdod', 'SG_BDOD'),
  soilgrids020('cfvo', 'SG_CFVO'), soilgrids020('ocd', 'SG_OCD'),
  soilgrids020('ocs', 'SG_OCS')
]);

var samples = embeddings.addBands(indices).addBands(soilgrids)
  .sampleRegions({collection: points, properties: ['POINTID'], scale: SCALE,
                  tileScale: 4, geometries: false})
  .filter(ee.Filter.notNull(['A00', 'NDVI', 'SG_SOC']));

print('First row', samples.first());

Export.table.toDrive({
  collection: samples,
  description: 'AEF_LUCAS_all_' + YEAR,
  fileNamePrefix: 'AEF_LUCAS_all_' + YEAR,
  fileFormat: 'CSV'
});
