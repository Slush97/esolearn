// scry-cv workbench — all application logic

const { invoke } = window.__TAURI__.core;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let operations = [];       // OpInfo[] from list_operations
let currentOp = null;      // OpSpec being configured in the params panel
let pipeline = null;       // PipelineInfo from last result
let activeIndex = 0;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  operations = await invoke("list_operations");
  renderSidebar();

  // Start with a checkerboard
  await setSource({ kind: "Checkerboard", cell_size: 16, width: 256, height: 256 });
}

// ---------------------------------------------------------------------------
// Backend calls
// ---------------------------------------------------------------------------

async function setSource(op) {
  try {
    const result = await invoke("set_source", { op });
    handleResult(result);
    currentOp = null;
    renderParams();
  } catch (e) {
    showError(e);
  }
}

async function addStep(op) {
  try {
    const result = await invoke("add_step", { op });
    handleResult(result);
  } catch (e) {
    showError(e);
  }
}

async function updateStep(index, op) {
  try {
    const result = await invoke("update_step", { index, op });
    handleResult(result);
  } catch (e) {
    showError(e);
  }
}

async function removeStep(index) {
  try {
    const result = await invoke("remove_step", { index });
    handleResult(result);
  } catch (e) {
    showError(e);
  }
}

async function viewStep(index) {
  try {
    const result = await invoke("get_step", { index });
    handleResult(result);
  } catch (e) {
    showError(e);
  }
}

// ---------------------------------------------------------------------------
// Result handling
// ---------------------------------------------------------------------------

function handleResult(result) {
  renderImage(result.pixels, result.width, result.height);
  renderOverlay(result.overlay, result.width, result.height);
  pipeline = result.pipeline;
  activeIndex = result.pipeline.active_index;
  renderBreadcrumbs();

  const info = document.getElementById("image-info");
  info.textContent = `${result.width} x ${result.height}`;
}

function renderImage(pixelBytes, width, height) {
  const canvas = document.getElementById("img-canvas");
  const overlay = document.getElementById("overlay-canvas");

  // Scale canvas for display (fit in available space)
  const area = document.getElementById("canvas-area");
  const maxW = area.clientWidth - 40;
  const maxH = area.clientHeight - 40;
  const scale = Math.min(maxW / width, maxH / height, 4);
  const dispW = Math.floor(width * scale);
  const dispH = Math.floor(height * scale);

  canvas.width = width;
  canvas.height = height;
  canvas.style.width = dispW + "px";
  canvas.style.height = dispH + "px";

  overlay.width = width;
  overlay.height = height;
  overlay.style.width = dispW + "px";
  overlay.style.height = dispH + "px";

  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(new Uint8ClampedArray(pixelBytes), width, height);
  ctx.putImageData(imageData, 0, 0);
}

// ---------------------------------------------------------------------------
// Overlay rendering
// ---------------------------------------------------------------------------

function renderOverlay(overlay, width, height) {
  const canvas = document.getElementById("overlay-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!overlay) return;

  switch (overlay.type) {
    case "Lines":
      renderLines(ctx, overlay.lines, width, height);
      break;
    case "Circles":
      renderCircles(ctx, overlay.circles);
      break;
    case "Keypoints":
      renderKeypoints(ctx, overlay.points);
      break;
    case "Components":
      renderComponents(ctx, overlay);
      break;
    case "ContourPaths":
      renderContours(ctx, overlay.contours);
      break;
  }
}

function renderLines(ctx, lines, w, h) {
  const diag = Math.sqrt(w * w + h * h);
  ctx.strokeStyle = "rgba(233, 69, 96, 0.7)";
  ctx.lineWidth = 1;

  for (const line of lines.slice(0, 50)) {
    const cosT = Math.cos(line.theta);
    const sinT = Math.sin(line.theta);
    const x0 = cosT * line.rho;
    const y0 = sinT * line.rho;

    ctx.beginPath();
    ctx.moveTo(x0 + diag * (-sinT), y0 + diag * cosT);
    ctx.lineTo(x0 - diag * (-sinT), y0 - diag * cosT);
    ctx.stroke();
  }
}

function renderCircles(ctx, circles) {
  ctx.strokeStyle = "rgba(46, 213, 115, 0.8)";
  ctx.lineWidth = 1.5;

  for (const c of circles.slice(0, 50)) {
    ctx.beginPath();
    ctx.arc(c.cx, c.cy, c.radius, 0, Math.PI * 2);
    ctx.stroke();

    // Center dot
    ctx.fillStyle = "rgba(46, 213, 115, 0.9)";
    ctx.beginPath();
    ctx.arc(c.cx, c.cy, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderKeypoints(ctx, points) {
  ctx.strokeStyle = "rgba(72, 219, 251, 0.8)";
  ctx.lineWidth = 1;

  for (const kp of points.slice(0, 500)) {
    const r = Math.max(kp.size / 2, 2);
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, r, 0, Math.PI * 2);
    ctx.stroke();

    // Orientation tick
    if (kp.angle >= 0) {
      ctx.beginPath();
      ctx.moveTo(kp.x, kp.y);
      ctx.lineTo(kp.x + r * Math.cos(kp.angle), kp.y + r * Math.sin(kp.angle));
      ctx.stroke();
    }
  }
}

function renderComponents(ctx, overlay) {
  if (overlay.num_labels === 0) return;

  // Generate distinct colors per label
  const colors = [];
  for (let i = 0; i <= overlay.num_labels; i++) {
    const hue = (i * 137.5) % 360; // golden angle
    colors.push(i === 0 ? [0, 0, 0, 0] : hslToRgba(hue, 70, 55, 120));
  }

  const imageData = ctx.createImageData(overlay.width, overlay.height);
  const data = imageData.data;
  for (let i = 0; i < overlay.labels.length; i++) {
    const label = overlay.labels[i];
    const c = colors[label] || [0, 0, 0, 0];
    data[i * 4] = c[0];
    data[i * 4 + 1] = c[1];
    data[i * 4 + 2] = c[2];
    data[i * 4 + 3] = c[3];
  }
  ctx.putImageData(imageData, 0, 0);
}

function renderContours(ctx, contours) {
  for (let ci = 0; ci < contours.length && ci < 100; ci++) {
    const pts = contours[ci];
    if (pts.length < 2) continue;

    const hue = (ci * 137.5) % 360;
    ctx.strokeStyle = `hsla(${hue}, 80%, 60%, 0.8)`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0], pts[i][1]);
    }
    ctx.closePath();
    ctx.stroke();
  }
}

function hslToRgba(h, s, l, a) {
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12;
  const f = n => l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255), a];
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

function renderSidebar() {
  const container = document.getElementById("op-list");
  container.innerHTML = "";

  // Group by category
  const groups = {};
  for (const op of operations) {
    if (!groups[op.category]) groups[op.category] = [];
    groups[op.category].push(op);
  }

  for (const [category, ops] of Object.entries(groups)) {
    const div = document.createElement("div");
    div.className = "op-category";

    const label = document.createElement("div");
    label.className = "op-category-label";
    label.textContent = category;
    div.appendChild(label);

    for (const op of ops) {
      const btn = document.createElement("button");
      btn.className = "op-btn " + (op.category === "Source" ? "source" : "process");
      btn.textContent = op.label;
      btn.addEventListener("click", () => selectOperation(op));
      div.appendChild(btn);
    }

    container.appendChild(div);
  }
}

// ---------------------------------------------------------------------------
// Parameter panel
// ---------------------------------------------------------------------------

function selectOperation(opInfo) {
  currentOp = JSON.parse(JSON.stringify(opInfo.default_op));
  renderParams();

  const applyBtn = document.getElementById("apply-btn");
  applyBtn.style.display = "block";
  applyBtn.onclick = () => {
    if (currentOp.kind === "LoadFile") {
      // For LoadFile, we'd need a file dialog — skip for now
      showError("File loading not yet implemented in UI");
      return;
    }
    if (isSource(currentOp)) {
      setSource(currentOp);
    } else {
      addStep(currentOp);
    }
  };
}

function isSource(op) {
  return ["SolidColor", "Checkerboard", "Gradient", "Rectangle", "GaussianBlob", "LoadFile"].includes(op.kind);
}

function renderParams() {
  const container = document.getElementById("param-controls");
  container.innerHTML = "";

  if (!currentOp) {
    document.getElementById("apply-btn").style.display = "none";
    return;
  }

  const op = currentOp;

  switch (op.kind) {
    case "SolidColor":
      addSlider(container, "value", "Value", op.value, 0, 1, 0.01);
      addNumber(container, "width", "Width", op.width, 16, 1024);
      addNumber(container, "height", "Height", op.height, 16, 1024);
      break;
    case "Checkerboard":
      addNumber(container, "cell_size", "Cell Size", op.cell_size, 2, 128);
      addNumber(container, "width", "Width", op.width, 16, 1024);
      addNumber(container, "height", "Height", op.height, 16, 1024);
      break;
    case "Gradient":
      addNumber(container, "width", "Width", op.width, 16, 1024);
      addNumber(container, "height", "Height", op.height, 16, 1024);
      break;
    case "Rectangle":
      addNumber(container, "width", "Width", op.width, 16, 1024);
      addNumber(container, "height", "Height", op.height, 16, 1024);
      addNumber(container, "rx", "Rect X", op.rx, 0, 1024);
      addNumber(container, "ry", "Rect Y", op.ry, 0, 1024);
      addNumber(container, "rw", "Rect W", op.rw, 1, 1024);
      addNumber(container, "rh", "Rect H", op.rh, 1, 1024);
      break;
    case "GaussianBlob":
      addSlider(container, "sigma", "Sigma", op.sigma, 5, 100, 1);
      addNumber(container, "width", "Width", op.width, 16, 1024);
      addNumber(container, "height", "Height", op.height, 16, 1024);
      break;
    case "GaussianBlur":
      addSlider(container, "sigma", "Sigma", op.sigma, 0.5, 10, 0.1);
      break;
    case "Bilateral":
      addSlider(container, "sigma_space", "Space Sigma", op.sigma_space, 0.5, 10, 0.1);
      addSlider(container, "sigma_color", "Color Sigma", op.sigma_color, 0.01, 1, 0.01);
      break;
    case "Median":
      addNumber(container, "radius", "Radius", op.radius, 1, 10);
      break;
    case "BoxBlur":
      addNumber(container, "radius", "Radius", op.radius, 1, 20);
      break;
    case "Sobel":
      // No parameters
      addNote(container, "Computes gradient magnitude (Sobel X + Y)");
      break;
    case "Canny":
      addSlider(container, "low", "Low Threshold", op.low, 0, 0.5, 0.01);
      addSlider(container, "high", "High Threshold", op.high, 0.01, 1, 0.01);
      break;
    case "HoughLines":
      addSlider(container, "rho_res", "Rho Res", op.rho_res, 0.5, 5, 0.1);
      addSlider(container, "theta_res", "Theta Res", op.theta_res, 0.005, 0.1, 0.001);
      addNumber(container, "threshold", "Threshold", op.threshold, 1, 200);
      break;
    case "HoughCircles":
      addNumber(container, "center_threshold", "Center Thresh", op.center_threshold, 1, 100);
      addNumber(container, "radius_threshold", "Radius Thresh", op.radius_threshold, 1, 100);
      addNumber(container, "min_radius", "Min Radius", op.min_radius, 1, 200);
      addNumber(container, "max_radius", "Max Radius", op.max_radius, 5, 500);
      addSlider(container, "min_dist", "Min Distance", op.min_dist, 1, 100, 1);
      break;
    case "OrbDetect":
      addNumber(container, "n_features", "N Features", op.n_features, 10, 2000);
      addSlider(container, "fast_threshold", "FAST Threshold", op.fast_threshold, 0.01, 0.3, 0.01);
      break;
    case "ConnectedComponents":
      addSelect(container, "connectivity", "Connectivity", op.connectivity, [
        { value: 4, label: "4-connected" },
        { value: 8, label: "8-connected" },
      ]);
      break;
    case "Contours":
      addNote(container, "Finds contour boundaries in binary/edge image");
      break;
    case "Erode":
    case "Dilate":
    case "MorphOpen":
    case "MorphClose":
      addSelect(container, "shape", "Shape", op.shape, [
        { value: "rect", label: "Rectangle" },
        { value: "cross", label: "Cross" },
        { value: "ellipse", label: "Ellipse" },
      ]);
      addNumber(container, "ksize", "Kernel Size", op.ksize, 3, 15);
      break;
  }
}

// ---------------------------------------------------------------------------
// UI helpers for param controls
// ---------------------------------------------------------------------------

function addSlider(container, key, label, value, min, max, step) {
  const row = document.createElement("div");
  row.className = "param-row";

  const lbl = document.createElement("label");
  const valSpan = document.createElement("span");
  valSpan.className = "param-value";
  valSpan.textContent = Number(value).toFixed(step < 0.1 ? 3 : step < 1 ? 1 : 0);
  lbl.textContent = label + " ";
  lbl.appendChild(valSpan);

  const input = document.createElement("input");
  input.type = "range";
  input.min = min;
  input.max = max;
  input.step = step;
  input.value = value;
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    currentOp[key] = v;
    valSpan.textContent = v.toFixed(step < 0.1 ? 3 : step < 1 ? 1 : 0);
  });

  row.appendChild(lbl);
  row.appendChild(input);
  container.appendChild(row);
}

function addNumber(container, key, label, value, min, max) {
  const row = document.createElement("div");
  row.className = "param-row";

  const lbl = document.createElement("label");
  lbl.textContent = label;

  const input = document.createElement("input");
  input.type = "number";
  input.min = min;
  input.max = max;
  input.value = value;
  input.addEventListener("change", () => {
    currentOp[key] = parseInt(input.value, 10);
  });

  row.appendChild(lbl);
  row.appendChild(input);
  container.appendChild(row);
}

function addSelect(container, key, label, value, options) {
  const row = document.createElement("div");
  row.className = "param-row";

  const lbl = document.createElement("label");
  lbl.textContent = label;

  const sel = document.createElement("select");
  for (const opt of options) {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.label;
    if (String(opt.value) === String(value)) o.selected = true;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => {
    const v = sel.value;
    currentOp[key] = isNaN(v) ? v : parseInt(v, 10);
  });

  row.appendChild(lbl);
  row.appendChild(sel);
  container.appendChild(row);
}

function addNote(container, text) {
  const row = document.createElement("div");
  row.className = "param-row";
  row.style.color = "#666";
  row.style.fontStyle = "italic";
  row.textContent = text;
  container.appendChild(row);
}

// ---------------------------------------------------------------------------
// Breadcrumbs
// ---------------------------------------------------------------------------

function renderBreadcrumbs() {
  const container = document.getElementById("crumb-list");
  container.innerHTML = "";

  if (!pipeline) return;

  for (let i = 0; i < pipeline.steps.length; i++) {
    if (i > 0) {
      const arrow = document.createElement("span");
      arrow.className = "crumb-arrow";
      arrow.textContent = "\u2192";
      container.appendChild(arrow);
    }

    const crumb = document.createElement("span");
    crumb.className = "crumb" + (i === activeIndex ? " active" : "");
    crumb.textContent = pipeline.steps[i].label;

    crumb.addEventListener("click", () => viewStep(i));

    // Remove button for non-source steps
    if (i > 0) {
      const rm = document.createElement("span");
      rm.className = "remove-btn";
      rm.textContent = "\u00d7";
      rm.addEventListener("click", (e) => {
        e.stopPropagation();
        removeStep(i);
      });
      crumb.appendChild(rm);
    }

    container.appendChild(crumb);
  }
}

// ---------------------------------------------------------------------------
// Error display
// ---------------------------------------------------------------------------

function showError(msg) {
  console.error("scry-app:", msg);
  const info = document.getElementById("image-info");
  info.textContent = "Error: " + msg;
  info.style.color = "#e94560";
  setTimeout(() => { info.style.color = "#666"; }, 3000);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", init);
