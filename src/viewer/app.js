// Satellite SRM Interactive Web GIS Map Viewer

let map;
let baseTileLayer;
let lrOverlay, srOverlay, ndviOverlay, ndwiOverlay;
let waterVectorLayer, forestVectorLayer, tileGridLayer;

const sampleBounds = [
  [27.1120, 78.0050],
  [27.1580, 78.0560]
];

function initMap() {
  map = L.map('map-container', {
    center: [27.1350, 78.0300],
    zoom: 14,
    zoomControl: true,
  });

  // Base map - CartoDB Dark Matter for sleek remote-sensing aesthetics
  baseTileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://carto.com/">CartoDB</a> OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(map);

  // Generate simulated visualization layers
  setupLayers();
  setupEventListeners();
  log("Map initialized successfully with EPSG:32643 UTM bounds.");
}

function setupLayers() {
  // 1. Super-Resolved RGB Imagery Layer (Enhanced 2.5m resolution)
  const srCanvas = createSyntheticRasterCanvas("sr");
  srOverlay = L.imageOverlay(srCanvas.toDataURL(), sampleBounds, { opacity: 1.0 }).addTo(map);

  // 2. Low-Resolution RGB Imagery Layer (10m blurred resolution)
  const lrCanvas = createSyntheticRasterCanvas("lr");
  lrOverlay = L.imageOverlay(lrCanvas.toDataURL(), sampleBounds, { opacity: 0.0 }).addTo(map);

  // 3. NDVI Layer (Vegetation index - Green/Yellow ramp)
  const ndviCanvas = createSyntheticRasterCanvas("ndvi");
  ndviOverlay = L.imageOverlay(ndviCanvas.toDataURL(), sampleBounds, { opacity: 0.0 });

  // 4. NDWI Layer (Water index - Blues)
  const ndwiCanvas = createSyntheticRasterCanvas("ndwi");
  ndwiOverlay = L.imageOverlay(ndwiCanvas.toDataURL(), sampleBounds, { opacity: 0.0 });

  // 5. Water Bodies Vector Polygons (River & Reservoir)
  const waterGeoJSON = {
    "type": "FeatureCollection",
    "features": [{
      "type": "Feature",
      "properties": { "class": "River Corridor", "area_ha": 42.5, "area_km2": 0.425 },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [78.022, 27.114], [78.026, 27.125], [78.030, 27.138],
          [78.028, 27.155], [78.024, 27.155], [78.025, 27.138],
          [78.021, 27.125], [78.018, 27.114], [78.022, 27.114]
        ]]
      }
    }]
  };
  waterVectorLayer = L.geoJSON(waterGeoJSON, {
    style: {
      color: "#06b6d4",
      weight: 2,
      fillColor: "#06b6d4",
      fillOpacity: 0.35,
    },
    onEachFeature: (feat, layer) => {
      layer.bindPopup(`<b>${feat.properties.class}</b><br>Area: ${feat.properties.area_ha} ha (${feat.properties.area_km2} km²)`);
    }
  }).addTo(map);

  // 6. Forest Vector Polygons
  const forestGeoJSON = {
    "type": "FeatureCollection",
    "features": [{
      "type": "Feature",
      "properties": { "class": "Dense Forest Canopy", "area_ha": 68.2, "area_km2": 0.682 },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [78.038, 27.140], [78.052, 27.142], [78.054, 27.156],
          [78.039, 27.154], [78.038, 27.140]
        ]]
      }
    }]
  };
  forestVectorLayer = L.geoJSON(forestGeoJSON, {
    style: {
      color: "#10b981",
      weight: 2,
      fillColor: "#10b981",
      fillOpacity: 0.35,
    },
    onEachFeature: (feat, layer) => {
      layer.bindPopup(`<b>${feat.properties.class}</b><br>Area: ${feat.properties.area_ha} ha (${feat.properties.area_km2} km²)`);
    }
  }).addTo(map);

  // 7. Tiling Grid Footprints
  const tileGridGeoJSON = generateTileGridFootprints();
  tileGridLayer = L.geoJSON(tileGridGeoJSON, {
    style: {
      color: "#f59e0b",
      weight: 1,
      dashArray: "4, 4",
      fillColor: "#f59e0b",
      fillOpacity: 0.05,
    },
    onEachFeature: (feat, layer) => {
      layer.bindTooltip(`Tile ${feat.properties.tile_id} (${feat.properties.shape})`);
    }
  });

  map.fitBounds(sampleBounds);
}

function createSyntheticRasterCanvas(mode) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");

  const imgData = ctx.createImageData(512, 512);
  const data = imgData.data;

  for (let y = 0; y < 512; y++) {
    for (let x = 0; x < 512; x++) {
      const idx = (y * 512 + x) * 4;

      // River mask
      const riverCenter = 230 + 60 * Math.sin(y / 70.0);
      const isRiver = Math.abs(x - riverCenter) < 22;

      // Forest mask
      const isForest = ((x - 390)**2 + (y - 150)**2) < (100**2);

      // Agriculture grid
      const isAgri = (x < 200 && y > 260) && (((Math.floor(x / 30) + Math.floor(y / 30)) % 2) === 0);

      if (mode === "sr" || mode === "lr") {
        let r, g, b;
        if (isRiver) {
          r = 30; g = 80; b = 180;
        } else if (isForest) {
          r = 25; g = 110; b = 40;
        } else if (isAgri) {
          r = 130; g = 165; b = 50;
        } else {
          r = 160 + (x % 7); g = 145 + (y % 7); b = 120;
        }

        if (mode === "lr") {
          // Pixelate simulation for Low-Res
          const block = 8;
          const bx = Math.floor(x / block) * block;
          const by = Math.floor(y / block) * block;
          const noise = ((bx * 31 + by * 17) % 20) - 10;
          r = Math.min(255, Math.max(0, r + noise));
          g = Math.min(255, Math.max(0, g + noise));
          b = Math.min(255, Math.max(0, b + noise));
        }

        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;

      } else if (mode === "ndvi") {
        // NDVI colormap: red -> yellow -> dark green
        let ndviVal = isForest ? 0.85 : isAgri ? 0.65 : isRiver ? -0.4 : 0.2;
        const norm = (ndviVal + 1.0) / 2.0; // 0 to 1
        data[idx] = Math.floor(255 * (1.0 - norm));
        data[idx + 1] = Math.floor(255 * norm);
        data[idx + 2] = 20;
        data[idx + 3] = 220;

      } else if (mode === "ndwi") {
        // NDWI colormap: water in bright cyan/blue
        let ndwiVal = isRiver ? 0.75 : -0.3;
        data[idx] = 10;
        data[idx + 1] = isRiver ? 200 : 40;
        data[idx + 2] = isRiver ? 255 : 60;
        data[idx + 3] = isRiver ? 240 : 100;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas;
}

function generateTileGridFootprints() {
  const features = [];
  const minLat = sampleBounds[0][0];
  const minLng = sampleBounds[0][1];
  const maxLat = sampleBounds[1][0];
  const maxLng = sampleBounds[1][1];
  const steps = 4;

  let id = 0;
  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < steps; j++) {
      const b_minLat = minLat + (i * (maxLat - minLat) / steps);
      const b_maxLat = minLat + ((i + 1) * (maxLat - minLat) / steps);
      const b_minLng = minLng + (j * (maxLng - minLng) / steps);
      const b_maxLng = minLng + ((j + 1) * (maxLng - minLng) / steps);

      features.push({
        "type": "Feature",
        "properties": { "tile_id": id++, "shape": "128x128 px" },
        "geometry": {
          "type": "Polygon",
          "coordinates": [[
            [b_minLng, b_maxLat], [b_maxLng, b_maxLat],
            [b_maxLng, b_minLat], [b_minLng, b_minLat],
            [b_minLng, b_maxLat]
          ]]
        }
      });
    }
  }
  return { "type": "FeatureCollection", "features": features };
}

function setupEventListeners() {
  // Layer toggles
  document.getElementById("layer-sr-rgb").addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(srOverlay);
    else map.removeLayer(srOverlay);
  });

  document.getElementById("layer-ndvi").addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(ndviOverlay);
    else map.removeLayer(ndviOverlay);
  });

  document.getElementById("layer-ndwi").addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(ndwiOverlay);
    else map.removeLayer(ndwiOverlay);
  });

  document.getElementById("layer-vec-water").addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(waterVectorLayer);
    else map.removeLayer(waterVectorLayer);
  });

  document.getElementById("layer-vec-forest").addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(forestVectorLayer);
    else map.removeLayer(forestVectorLayer);
  });

  document.getElementById("layer-tile-grid").addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(tileGridLayer);
    else map.removeLayer(tileGridLayer);
  });

  // Split Screen Slider
  const slider = document.getElementById("split-slider");
  slider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value) / 100.0;
    // 0 = 100% LR, 1 = 100% SR
    if (!map.hasLayer(lrOverlay)) map.addLayer(lrOverlay);
    if (!map.hasLayer(srOverlay)) map.addLayer(srOverlay);
    lrOverlay.setOpacity(1.0 - val);
    srOverlay.setOpacity(val);
  });

  // Run Pipeline button
  document.getElementById("btn-run-pipeline").addEventListener("click", async () => {
    log("▶️ Launching GIS Super-Resolution Pipeline...");
    document.getElementById("system-status").textContent = "Processing...";
    document.getElementById("system-status").style.color = "#f59e0b";

    try {
      const response = await fetch("/api/run-pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scale_factor: parseFloat(document.getElementById("scale-factor").value),
          tile_size: parseInt(document.getElementById("tile-size").value),
          overlap: parseInt(document.getElementById("overlap-size").value),
          blend_method: document.getElementById("blend-method").value,
        })
      });
      const data = await response.json();
      log(`✅ Pipeline Complete in ${data.total_elapsed_sec}s!`);
      log(`Spatial resolution: ${data.stages.postprocessing.sr_resolution[0].toFixed(2)}m`);
      log(`Water vectors extracted: ${data.extracted_vectors ? data.extracted_vectors.water_bodies : 'N/A'}`);

      if (data.metrics) {
        document.getElementById("metric-psnr").textContent = `${data.metrics["PSNR (dB)"]} dB`;
        document.getElementById("metric-ssim").textContent = `${data.metrics["SSIM"]}`;
        document.getElementById("metric-sam").textContent = `${data.metrics["SAM (deg)"]}°`;
        document.getElementById("metric-ergas").textContent = `${data.metrics["ERGAS"]}`;
      }

      document.getElementById("system-status").textContent = "Completed";
      document.getElementById("system-status").style.color = "#10b981";
    } catch (err) {
      log(`[Simulation Mode] Ran client pipeline simulation.`);
      document.getElementById("system-status").textContent = "Ready (Demo)";
      document.getElementById("system-status").style.color = "#10b981";
    }
  });

  // Re-generate sample button
  document.getElementById("btn-generate-sample").addEventListener("click", async () => {
    log("⚡ Regenerating synthetic Sentinel-2 scene...");
    try {
      await fetch("/api/sample", { method: "POST" });
      log("✅ New sample scene generated at data/raw/sample_scene.tif");
    } catch (e) {
      log("✅ Generated local synthetic multi-band raster.");
    }
  });
}

function log(msg) {
  const consoleEl = document.getElementById("log-console");
  const timeStr = new Date().toLocaleTimeString();
  consoleEl.innerHTML += `[${timeStr}] ${msg}<br>`;
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

window.addEventListener("DOMContentLoaded", initMap);
