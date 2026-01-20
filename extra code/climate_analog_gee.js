// ====================
// 0. Region
// ====================
var baridoab = ee.FeatureCollection(
  "projects/ee-hamza-rafique/assets/shapefiles/bariDoab"
);
Map.addLayer(baridoab, {color: 'red'}, 'Bari Doab');

// ====================
// 1. Aridity Index
// ====================
var ai = ee.Image(
  "projects/sat-io/open-datasets/global_ai/global_ai_yearly"
).multiply(0.0001);

// ====================
// 2. ERA5 base collection
// ====================
var era5 = ee.ImageCollection("ECMWF/ERA5/MONTHLY")
  .filterDate("1981-01-01", "2010-12-31");

// --------------------
// Helpers
// --------------------
function addTempC(img) {
  return img.addBands(
    img.select("mean_2m_air_temperature")
      .subtract(273.15)
      .rename("T_C")
  );
}

function addPrecipMM(img) {
  return img.addBands(
    img.select("total_precipitation")
      .multiply(1000)
      .rename("P_mm")
  );
}

function addVPD(img) {
  var T = img.select("mean_2m_air_temperature").subtract(273.15);
  var Td = img.select("dewpoint_2m_temperature").subtract(273.15);
  var es = T.expression(
    "0.6108 * exp((17.27 * T) / (T + 237.3))", {T: T}
  );
  var ea = Td.expression(
    "0.6108 * exp((17.27 * Td) / (Td + 237.3))", {Td: Td}
  );
  return img.addBands(es.subtract(ea).rename("VPD_kPa"));
}

function addWind(img) {
  var u = img.select("u_component_of_wind_10m");
  var v = img.select("v_component_of_wind_10m");
  return img.addBands(u.pow(2).add(v.pow(2)).sqrt().rename("Wind_ms"));
}

// ====================
// 3. Yearly derived variables
// ====================
var years = ee.List.sequence(1981, 2010);

var yearly = ee.ImageCollection.fromImages(
  years.map(function(y) {
    var yr = ee.Number(y);

    var ic = era5
      .filter(ee.Filter.calendarRange(yr, yr, "year"))
      .map(addTempC)
      .map(addPrecipMM)
      .map(addVPD)
      .map(addWind);

    // --- Temp annual range ---
    var tmax = ic.select("T_C").max();
    var tmin = ic.select("T_C").min();
    var tempRange = tmax.subtract(tmin)
      .rename("Temp_Annual_Range_C");

    // --- Precip warmest quarter ---
    var monthlyP = ic.select("P_mm");

    var p3 = ee.ImageCollection([
      monthlyP.filter(ee.Filter.calendarRange(1,3,"month")).sum(),
      monthlyP.filter(ee.Filter.calendarRange(2,4,"month")).sum(),
      monthlyP.filter(ee.Filter.calendarRange(3,5,"month")).sum(),
      monthlyP.filter(ee.Filter.calendarRange(4,6,"month")).sum(),
      monthlyP.filter(ee.Filter.calendarRange(5,7,"month")).sum(),
      monthlyP.filter(ee.Filter.calendarRange(6,8,"month")).sum(),
      monthlyP.filter(ee.Filter.calendarRange(7,9,"month")).sum(),
      monthlyP.filter(ee.Filter.calendarRange(8,10,"month")).sum(),
      monthlyP.filter(ee.Filter.calendarRange(9,11,"month")).sum(),
      monthlyP.filter(ee.Filter.calendarRange(10,12,"month")).sum()
    ]);

    var precipWQ = p3.max()
      .rename("Precip_WarmestQ_mm");

    // --- Annual means ---
    var vpdMean  = ic.select("VPD_kPa").mean();
    var windMean = ic.select("Wind_ms").mean();

    return ee.Image.cat([
      tempRange,
      precipWQ,
      vpdMean,
      windMean
    ]).set("year", yr);
  })
);

// ====================
// 4. Means and σ
// ====================
var meanImg = yearly.mean();
var sigmaImg = yearly.reduce(ee.Reducer.stdDev())
  .rename([
    "Temp_Annual_Range_C_sigma",
    "Precip_WarmestQ_mm_sigma",
    "VPD_kPa_sigma",
    "Wind_ms_sigma"
  ]);

// ====================
// 5. Sites
// ====================
var sites = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([-116.7356, 43.1439]).buffer(5000), {Site: "US-Rls"}),
  ee.Feature(ee.Geometry.Point([-116.7486, 43.0645]).buffer(5000), {Site: "US-Rms"}),
  ee.Feature(ee.Geometry.Point([-116.7231, 43.1207]).buffer(5000), {Site: "US-Rwf"}),
  ee.Feature(ee.Geometry.Point([-110.8395, 31.9083]).buffer(5000), {Site: "US-SRC"})
]);

function extract(feature) {
  var g = feature.geometry();

  return feature
    .set(ai.reduceRegion(ee.Reducer.mean(), g, 1000))
    .set(meanImg.reduceRegion(ee.Reducer.mean(), g, 27830))
    .set(sigmaImg.reduceRegion(ee.Reducer.mean(), g, 27830));
}

// Bari Doab (no σ used later, but safe to include)
var bari = ee.Feature(baridoab.geometry(), {Site: "Bari_Doab"});

var out = ee.FeatureCollection([bari])
  .merge(sites)
  .map(extract);

// ====================
// 6. Export
// ====================
Export.table.toDrive({
  collection: out,
  description: "BariDoab_vs_FluxSites_Climate_v6",
  fileFormat: "CSV"
});
