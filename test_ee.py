import ee
ee.Initialize(project='earth-engine-legacy-project')

pt = ee.Geometry.Point([8.52, 12.00])
box = pt.buffer(3200).bounds()

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(box)
      .filterDate('2019-11-01', '2020-03-31')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .median())

bands = ['B2', 'B3', 'B4', 'B8']
result = s2.select(bands).reduceRegion(ee.Reducer.mean(), box, 10).getInfo()
print(result)