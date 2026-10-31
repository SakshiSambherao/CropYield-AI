const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const form = $("#predictionForm");
let latestPayload = null, latestPredictionId = null, shapChart = null;

function pretty(name) {
  return String(name).replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
}
async function loadFeatureFields() {
  if (!form) return;
  const crop = $("#crop").value;
  const container = $("#featureFields");
  container.innerHTML = '<div class="loading-line">Loading model features…</div>';
  try {
    const res = await fetch(`/api/features/${encodeURIComponent(crop)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Unable to load model fields.");
    container.innerHTML = "";
    data.features.forEach(feature => {
      const label = document.createElement("label");
      label.className = "feature-label";
      label.textContent = pretty(feature.name);
      let input;
      if (feature.type === "select" && feature.choices?.length) {
        input = document.createElement("select");
        const blank = document.createElement("option");
        blank.value = ""; blank.textContent = "Choose (optional)";
        input.appendChild(blank);
        feature.choices.forEach(choice => {
          const opt = document.createElement("option");
          opt.value = choice; opt.textContent = choice; input.appendChild(opt);
        });
      } else {
        input = document.createElement("input");
        input.type = feature.type === "number" ? "number" : "text";
        if (input.type === "number") input.step = "any";
        input.placeholder = feature.placeholder || "Leave blank if unknown";
      }
      input.name = feature.name;
      input.dataset.feature = feature.name;
      label.appendChild(input);
      container.appendChild(label);
    });
  } catch (err) {
    container.innerHTML = "";
    const msg = document.createElement("div");
    msg.className = "form-error";
    msg.textContent = err.message;
    container.appendChild(msg);
  }
}
function fillExampleValues() {
  // Illustrative values for testing the form, not agronomic recommendations.
  const examples = {
    state: "maharashtra", district: "pune", taluka: "junnar", village_: "",
    year: 2026, season: "rabi", variety: "local", soil_type: "black",
    soil_ph: 7, soil_moisture_percent: 50, organic_carbon_percent: 0.8,
    electrical_conductivity_ds_m: 0.5, nitrogen_kg_ha: 150, phosphorus_kg_ha: 60,
    potassium_kg_ha: 120, rainfall_monsoon_mm: 700, rainfall_crop_season_mm: 900,
    rainy_days: 40, avg_temperature_c: 27, max_temperature_c: 35, min_temperature_c: 20,
    relative_humidity_percent: 65, sunshine_hours_per_day: 7, wind_speed_kmh: 12,
    irrigation_method: "drip", irrigation_events: 20, irrigation_efficiency_percent: 80,
    fertilizer_strategy: "balanced", fertilizer_n_kg_ha: 150, fertilizer_p_kg_ha: 60,
    fertilizer_k_kg_ha: 120, seed_rate_kg_ha: 8, previous_crop: "cereal",
    pest_pressure: "low", disease_incidence_percent: 5, weed_control_method: "manual",
    mechanization_level: "medium", sowing_month: 10, harvest_month: 2,
    labor_days_per_ha: 100, cultivation_cost_inr_per_ha: 100000, farm_area_ha: 1
  };
  $$('[data-feature]').forEach(input => {
    const key = input.dataset.feature;
    if (Object.prototype.hasOwnProperty.call(examples, key)) {
      const value = String(examples[key]);
      if (input.tagName === 'SELECT' && ![...input.options].some(option => option.value === value)) {
        const option = document.createElement('option'); option.value = value; option.textContent = value;
        input.appendChild(option);
      }
      input.value = value;
    } else if (input.tagName === 'SELECT') {
      input.selectedIndex = 0;
    } else {
      input.value = '';
    }
  });
  const error = $('#formError'); if (error) error.classList.add('hidden');
}
function collectPayload() {
  const features = {};
  $$("[data-feature]").forEach(input => features[input.dataset.feature] = input.value.trim());
  return {crop: $("#crop").value, features};
}
function showError(msg) {
  const box = $("#formError");
  if (box) { box.textContent = msg; box.classList.remove("hidden"); }
}
if (form) {
  $("#crop").addEventListener("change", () => {
    latestPayload = null; latestPredictionId = null;
    $("#explainBtn").disabled = true;
    $("#resultCrop").textContent = "Not run yet";
    $("#yieldValue").textContent = "—";
    loadFeatureFields();
  });
  const exampleBtn = $("#exampleBtn");
  if (exampleBtn) exampleBtn.addEventListener("click", fillExampleValues);
  $("#clearBtn").addEventListener("click", () => {
    $$("[data-feature]").forEach(input => input.value = "");
    $("#formError").classList.add("hidden");
    latestPayload = null; latestPredictionId = null;
    $("#explainBtn").disabled = true;
    $("#yieldValue").textContent = "—";
    $("#resultCrop").textContent = "Not run yet";
    $("#resultId").textContent = "—";
    $("#shapResults").classList.add("hidden");
    $("#shapEmpty").classList.remove("hidden");
  });
  form.addEventListener("submit", async e => {
    e.preventDefault();
    $("#formError").classList.add("hidden");
    const btn = $("#predictBtn");
    btn.disabled = true; btn.textContent = "Predicting…";
    const payload = collectPayload();
    try {
      const res = await fetch("/api/predict", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed.");
      latestPayload = payload; latestPredictionId = data.id;
      $("#yieldValue").textContent = Number(data.predicted_yield_tonnes_per_ha).toFixed(3);
      $("#resultCrop").textContent = data.crop;
      $("#resultId").textContent = `#${data.id}`;
      $("#explainBtn").disabled = false;
      $("#shapResults").classList.add("hidden");
      $("#shapEmpty").classList.remove("hidden");
    } catch (err) { showError(err.message); }
    finally { btn.disabled = false; btn.innerHTML = 'Predict yield <span>→</span>'; }
  });
  $("#explainBtn").addEventListener("click", async () => {
    if (!latestPayload) return;
    const btn = $("#explainBtn");
    btn.disabled = true; btn.textContent = "Calculating SHAP…";
    try {
      const res = await fetch("/api/explain", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...latestPayload, prediction_id: latestPredictionId})
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "SHAP failed.");
      $("#shapEmpty").classList.add("hidden");
      $("#shapResults").classList.remove("hidden");
      $("#shapNote").textContent = data.note;
      const labels = data.top_features.map(x => x.feature);
      const vals = data.top_features.map(x => x.impact);
      if (shapChart) shapChart.destroy();
      shapChart = new Chart($("#shapChart"), {
        type: "bar",
        data: {labels, datasets: [{data: vals, label: "SHAP value",
          backgroundColor: vals.map(v => v >= 0 ? "#2e8b57" : "#d16d60"), borderRadius: 5}]},
        options: {indexAxis: "y", responsive: true, plugins: {legend: {display: false}},
          scales: {x: {title: {display: true, text: "SHAP value"}}, y: {ticks: {font: {size: 10}}}}}
      });
      const rows = $("#shapRows"); rows.innerHTML = "";
      data.top_features.forEach(item => {
        const row = document.createElement("div"); row.className = "shap-row";
        const name = document.createElement("span"); name.textContent = item.feature;
        const impact = document.createElement("b");
        impact.textContent = `${item.impact > 0 ? "+" : ""}${item.impact}`;
        impact.className = item.impact >= 0 ? "positive" : "negative";
        row.append(name, impact); rows.appendChild(row);
      });
    } catch (err) { showError(err.message); }
    finally { btn.disabled = false; btn.textContent = "✧ Explain with SHAP"; }
  });
  loadFeatureFields();
}

const clearBtn = $("#clearHistoryBtn");
if (clearBtn) {
  clearBtn.addEventListener("click", async () => {
    if (!confirm("Clear all saved prediction history?")) return;
    const res = await fetch("/api/history/clear", {method: "POST"});
    if (res.ok) location.reload();
  });
}
$$(".delete-one").forEach(btn => btn.addEventListener("click", async () => {
  if (!confirm("Delete this prediction?")) return;
  const res = await fetch(`/api/history/${btn.dataset.id}`, {method: "DELETE"});
  if (res.ok) location.reload();
}));
