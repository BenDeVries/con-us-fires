/* Six synchronized county maps. All data and geometry are served with the tutorial. */
(async function () {
  "use strict";
  const root = document.getElementById("validation-maps");
  if (!root) return;
  const base = new URL("maps/", document.currentScript.src);
  const status = root.querySelector(".map-status");
  const play = root.querySelector(".map-play");
  const slider = root.querySelector(".map-time");
  const lead = root.querySelector(".map-horizon");
  const canvases = [...root.querySelectorAll("canvas")];
  let frame = 0, values, running = null, request = 0, paths, metadata, selected;
  const cache = new Map();
  const colors = (stops) => Array.from({length: 1025}, (_, index) => {
    const position = index / 1024 * (stops.length - 1);
    const low = Math.min(Math.floor(position), stops.length - 2);
    const t = position - low;
    return `rgb(${stops[low].map((v, k) => Math.round(v + t * (stops[low + 1][k] - v))).join(",")})`;
  });
  const probability = colors([[247, 250, 249], [121, 193, 165], [0, 79, 64]]);
  const magnitude = colors([[250, 249, 247], [253, 232, 172], [247, 148, 72], [193, 62, 32], [83, 21, 28]]);
  const fractionColor = (v) => magnitude[Math.round(Math.log1p(v / 1e-5) / Math.log1p(1e5) * 1024)];
  const percent = (v) => (100 * v).toLocaleString(undefined, {maximumSignificantDigits: 4}) + "%";
  const stop = () => { clearInterval(running); running = null; play.textContent = "Play"; play.setAttribute("aria-pressed", "false"); };
  const at = (field, county) => values[(frame * 5 + field) * paths.length + county];
  function draw() {
    if (!values) return;
    const fields = [0, 1, 2, 2, 3, 4];
    canvases.forEach((canvas, panel) => {
      const context = canvas.getContext("2d");
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.lineWidth = 0.22;
      context.strokeStyle = "#7c858580";
      paths.forEach((path, county) => {
        const value = at(fields[panel], county);
        context.fillStyle = panel === 2 ? (value > 0 ? "#004f40" : "#f7faf9") :
          panel % 2 === 0 ? probability[Math.round(value * 1024)] : fractionColor(value);
        context.fill(path, "evenodd");
        context.stroke(path);
      });
    });
    slider.value = frame;
    slider.setAttribute("aria-valuetext", selected.dates[frame]);
    status.textContent = `Target ${selected.dates[frame]} · origin ${selected.origins[frame]} · ${selected.horizon}-month lead · ${frame + 1} of ${metadata.shape[0]}`;
  }
  async function chooseLead() {
    stop();
    const sequence = ++request;
    const chosen = metadata.horizons[Number(lead.value) - 1];
    values = null;
    play.disabled = slider.disabled = true;
    status.textContent = `Loading ${chosen.horizon}-month forecasts…`;
    try {
      if (!cache.has(chosen.horizon)) {
        const response = await fetch(new URL(chosen.file, base));
        if (!response.ok) throw new Error(`Forecast data: HTTP ${response.status}`);
        const buffer = await response.arrayBuffer();
        if (buffer.byteLength !== metadata.shape.reduce((a, b) => a * b) * 4) throw new Error("Incomplete forecast data");
        cache.set(chosen.horizon, new Float32Array(buffer));
      }
      if (sequence !== request) return;
      selected = chosen;
      values = cache.get(chosen.horizon);
      play.disabled = slider.disabled = false;
      draw();
    } catch (error) {
      if (sequence === request) status.textContent = `Maps could not load: ${error.message}. Reload to try again.`;
    }
  }
  try {
    const read = async (file) => { const response = await fetch(new URL(file, base)); if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); };
    metadata = await read("index.json");
    const geometry = await read(metadata.geometry);
    paths = geometry.paths.map((path) => new Path2D(path));
    if (paths.length !== metadata.shape[2]) throw new Error("County geometry does not match forecasts");
    canvases.forEach((canvas) => { canvas.width = geometry.width; canvas.height = geometry.height; });
    slider.max = metadata.shape[0] - 1;
    lead.addEventListener("change", chooseLead);
    slider.addEventListener("input", () => { stop(); frame = Number(slider.value); draw(); });
    play.addEventListener("click", () => {
      if (running) { stop(); return; }
      play.textContent = "Pause";
      play.setAttribute("aria-pressed", "true");
      running = setInterval(() => { frame = (frame + 1) % metadata.shape[0]; draw(); }, 750);
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
    const detail = root.querySelector(".map-detail");
    canvases.forEach((canvas) => canvas.addEventListener("pointermove", (event) => {
      if (!values) return;
      const box = canvas.getBoundingClientRect();
      const x = (event.clientX - box.left) * canvas.width / box.width;
      const y = (event.clientY - box.top) * canvas.height / box.height;
      const context = canvas.getContext("2d");
      const county = paths.findIndex((path) => context.isPointInPath(path, x, y, "evenodd"));
      if (county < 0) { detail.textContent = "Point to a county to inspect its values."; return; }
      detail.textContent = `FIPS ${geometry.fips[county]} · observed ${percent(at(2, county))} (${at(2, county) > 0 ? "present" : "absent"}) · XGBoostLSS: p ${percent(at(0, county))}, conditional mean ${percent(at(1, county))} · GCN–LSTM: p ${percent(at(3, county))}, conditional mean ${percent(at(4, county))}`;
    }));
    await chooseLead();
  } catch (error) { status.textContent = `Maps could not load: ${error.message}.`; }
})();
