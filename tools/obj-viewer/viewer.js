import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VertexNormalsHelper } from "three/addons/helpers/VertexNormalsHelper.js";

const viewport = document.getElementById("viewport");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(2, 1.5, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0a0a12);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.3;
controls.maxDistance = 20;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x8888aa, 0.6));

const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight1.position.set(3, 5, 4);
scene.add(dirLight1);

const dirLight2 = new THREE.DirectionalLight(0x6688cc, 0.4);
dirLight2.position.set(-3, 2, -4);
scene.add(dirLight2);

scene.add(new THREE.GridHelper(10, 20, 0x222233, 0x161622));

const axisHelper = new THREE.AxesHelper(0.5);
axisHelper.position.set(-4.9, 0.001, -4.9);
scene.add(axisHelper);

let currentMesh = null;
let normalHelpers = [];
let wireframeActive = false;
let normalsActive = false;
let autoRotate = false;
let latestDownloadBytes = null;
let latestDownloadName = "mesh.obj";
let latestDownloadMime = "text/plain;charset=utf-8";
let currentPair = {
  uid: "",
  generated: null,
  gt: null,
};

const DEFAULT_MATERIAL = new THREE.MeshStandardMaterial({
  color: 0x7c8cf8,
  roughness: 0.45,
  metalness: 0.1,
  flatShading: false,
  side: THREE.DoubleSide,
});

const WIREFRAME_MATERIAL = new THREE.MeshStandardMaterial({
  color: 0x7c8cf8,
  roughness: 0.45,
  metalness: 0.1,
  wireframe: true,
  side: THREE.DoubleSide,
});

const objLoader = new OBJLoader();
const gltfLoader = new GLTFLoader();

function onResize() {
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

window.addEventListener("resize", onResize);
new ResizeObserver(onResize).observe(viewport);
onResize();

function animate() {
  requestAnimationFrame(animate);
  if (autoRotate && currentMesh) {
    currentMesh.rotation.y += 0.005;
  }
  controls.update();
  renderer.render(scene, camera);
}
animate();

function clearNormals() {
  for (const helper of normalHelpers) {
    scene.remove(helper);
    helper.dispose();
  }
  normalHelpers = [];
}

function clearMesh() {
  clearNormals();
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach((mat) => mat.dispose?.());
        } else {
          child.material.dispose?.();
        }
      }
    });
    currentMesh = null;
  }
}

function meshMaterial() {
  return wireframeActive ? WIREFRAME_MATERIAL.clone() : DEFAULT_MATERIAL.clone();
}

function applyDisplayMaterials(obj) {
  obj.traverse((child) => {
    if (!child.isMesh) return;
    child.material = meshMaterial();
    child.geometry.computeVertexNormals?.();
    child.geometry.computeBoundingBox?.();
  });
}

function updateMeshInfo(obj) {
  let totalVerts = 0;
  let totalFaces = 0;

  obj.traverse((child) => {
    if (!child.isMesh || !child.geometry) return;
    const geo = child.geometry;
    totalVerts += geo.attributes.position ? geo.attributes.position.count : 0;
    totalFaces += geo.index
      ? geo.index.count / 3
      : geo.attributes.position
        ? geo.attributes.position.count / 3
        : 0;
  });

  document.getElementById("info-verts").textContent = totalVerts.toLocaleString();
  document.getElementById("info-faces").textContent = Math.round(totalFaces).toLocaleString();
  document.getElementById("mesh-info").classList.add("visible");
  document.getElementById("empty-state").style.display = "none";
}

function frameObject(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);

  if (maxDim > 1e-6) {
    const scale = 2.0 / maxDim;
    obj.scale.setScalar(scale);
    obj.position.sub(center.multiplyScalar(scale));
  }

  obj.updateMatrixWorld(true);

  const fitBox = new THREE.Box3().setFromObject(obj);
  const fitCenter = fitBox.getCenter(new THREE.Vector3());
  const fitSize = fitBox.getSize(new THREE.Vector3());
  const fitMaxDim = Math.max(fitSize.x, fitSize.y, fitSize.z);
  const dist = fitMaxDim / (2 * Math.tan((camera.fov * Math.PI) / 360));

  controls.target.copy(fitCenter);
  camera.position.set(
    fitCenter.x + dist * 0.8,
    fitCenter.y + dist * 0.5,
    fitCenter.z + dist * 0.8
  );
  controls.update();
}

function showNormals() {
  clearNormals();
  if (!currentMesh || !normalsActive) return;
  currentMesh.traverse((child) => {
    if (child.isMesh) {
      const helper = new VertexNormalsHelper(child, 0.05, 0x44ff88);
      normalHelpers.push(helper);
      scene.add(helper);
    }
  });
}

async function parseMeshPayload(payload) {
  if (payload.format === "obj" && payload.textData) {
    return objLoader.parse(payload.textData);
  }

  if ((payload.format === "glb" || payload.format === "gltf") && payload.dataB64) {
    const bytes = Uint8Array.from(atob(payload.dataB64), (c) => c.charCodeAt(0));
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return await new Promise((resolve, reject) => {
      gltfLoader.parse(buffer, "", (gltf) => resolve(gltf.scene), reject);
    });
  }

  throw new Error(`Unsupported mesh format: ${payload.format || "unknown"}`);
}

async function loadMeshPayload(payload, label = "Mesh") {
  clearMesh();
  const obj = await parseMeshPayload(payload);
  applyDisplayMaterials(obj);
  scene.add(obj);
  currentMesh = obj;
  if (normalsActive) showNormals();
  updateMeshInfo(obj);
  frameObject(obj);
  logToConsole(`${label} loaded successfully.`, "success");
}

function setLatestDownload(bytes, fileName, mimeType) {
  latestDownloadBytes = bytes;
  latestDownloadName = fileName || "mesh.obj";
  latestDownloadMime = mimeType || "application/octet-stream";
  document.getElementById("download-result-btn").disabled = !latestDownloadBytes;
}

function setLatestDownloadText(text, fileName) {
  setLatestDownload(new TextEncoder().encode(text), fileName, "text/plain;charset=utf-8");
}

function triggerDownloadBytes(bytes, fileName, mimeType) {
  const blob = new Blob([bytes], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

const consoleOutput = document.getElementById("console-output");

function logToConsole(msg, type = "") {
  const line = document.createElement("div");
  if (type) line.className = type;
  line.textContent = msg;
  consoleOutput.appendChild(line);
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function clearConsole() {
  consoleOutput.innerHTML = "";
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
  });
});

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

document.getElementById("browse-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) loadMeshFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadMeshFile(fileInput.files[0]);
});

viewport.addEventListener("dragover", (e) => e.preventDefault());
viewport.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) loadMeshFile(file);
});

async function loadMeshFile(file) {
  clearConsole();
  logToConsole(`Loading file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, "info");
  const format = file.name.split(".").pop().toLowerCase();

  try {
    if (format === "obj") {
      const text = await file.text();
      await loadMeshPayload({ format, textData: text }, file.name);
      setLatestDownloadText(text, file.name);
      return;
    }

    const buffer = await file.arrayBuffer();
    const dataB64 = bytesToBase64(new Uint8Array(buffer));
    await loadMeshPayload({ format, dataB64 }, file.name);
    setLatestDownload(new Uint8Array(buffer), file.name, "application/octet-stream");
  } catch (err) {
    logToConsole(`Error loading mesh: ${err.message}`, "error");
  }
}

document.getElementById("load-obj-btn").addEventListener("click", async () => {
  const text = document.getElementById("obj-textarea").value.trim();
  if (!text) return;
  clearConsole();
  logToConsole("Loading pasted OBJ data…", "info");
  try {
    await loadMeshPayload({ format: "obj", textData: text }, "Pasted OBJ");
    setLatestDownloadText(text, "pasted.obj");
  } catch (err) {
    logToConsole(`Error parsing OBJ: ${err.message}`, "error");
  }
});

const codeTextarea = document.getElementById("code-textarea");
const executeBtn = document.getElementById("execute-btn");
const downloadResultBtn = document.getElementById("download-result-btn");
const loadingOverlay = document.getElementById("loading-overlay");

codeTextarea.value = `import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_monkey_add(size=1, location=(0, 0, 0))
obj = bpy.context.active_object
mod = obj.modifiers.new(name='Subsurf', type='SUBSURF')
mod.levels = 2
bpy.ops.object.shade_smooth()
`;

codeTextarea.addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    e.preventDefault();
    const start = codeTextarea.selectionStart;
    const end = codeTextarea.selectionEnd;
    codeTextarea.value =
      codeTextarea.value.substring(0, start) +
      "    " +
      codeTextarea.value.substring(end);
    codeTextarea.selectionStart = codeTextarea.selectionEnd = start + 4;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    executeCode();
  }
});

executeBtn.addEventListener("click", executeCode);

document.getElementById("clear-code-btn").addEventListener("click", () => {
  codeTextarea.value = "";
  codeTextarea.focus();
});

downloadResultBtn.addEventListener("click", () => {
  if (latestDownloadBytes) {
    triggerDownloadBytes(latestDownloadBytes, latestDownloadName, latestDownloadMime);
  }
});

async function executeCode() {
  const code = codeTextarea.value.trim();
  if (!code) return;

  clearConsole();
  logToConsole("Sending code to Blender…", "info");
  executeBtn.disabled = true;
  loadingOverlay.classList.add("visible");

  try {
    const resp = await fetch("/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    const data = await resp.json();

    if (data.stdout) {
      for (const line of data.stdout.split("\n").filter((l) => l.trim()).slice(-20)) {
        logToConsole(line);
      }
    }
    if (data.stderr) {
      for (const line of data.stderr.split("\n").filter((l) => l.trim()).slice(-10)) {
        logToConsole(line, "error");
      }
    }

    if (data.success && data.objData) {
      await loadMeshPayload({ format: "obj", textData: data.objData }, "Generated mesh");
      setLatestDownloadText(
        data.objData,
        data.savedPath ? data.savedPath.split("/").pop() : "generated.obj"
      );
      if (data.savedPath) {
        logToConsole(`Saved to local cache: ${data.savedPath}`, "info");
      }
      await refreshFileList();
    } else {
      logToConsole(`Execution failed: ${data.error || "Unknown error"}`, "error");
    }
  } catch (err) {
    logToConsole(`Network error: ${err.message}`, "error");
  } finally {
    executeBtn.disabled = false;
    loadingOverlay.classList.remove("visible");
  }
}

async function loadLibraryObject(path, name) {
  clearConsole();
  logToConsole(`Loading local mesh: ${path}`, "info");
  try {
    const resp = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
    const data = await resp.json();
    if (!data.success) {
      throw new Error(data.error || "Could not load mesh");
    }

    await loadMeshPayload(data, name || data.name || "Library mesh");

    if (data.format === "obj" && data.textData) {
      setLatestDownloadText(data.textData, data.name || name || "library.obj");
    } else if (data.dataB64) {
      const bytes = Uint8Array.from(atob(data.dataB64), (c) => c.charCodeAt(0));
      setLatestDownload(bytes, data.name || name || "library.bin", "application/octet-stream");
    }
  } catch (err) {
    logToConsole(`Error loading mesh: ${err.message}`, "error");
  }
}

async function refreshFileList() {
  const rootEl = document.getElementById("library-root");
  const listEl = document.getElementById("file-list");

  try {
    const resp = await fetch("/api/files");
    const data = await resp.json();
    rootEl.textContent = data.root || "No local mesh directory configured.";
    listEl.innerHTML = "";

    if (!data.files || data.files.length === 0) {
      const empty = document.createElement("div");
      empty.className = "file-path";
      empty.textContent = data.root
        ? "No mesh files found."
        : "Remote fetches will populate the local cache here.";
      listEl.appendChild(empty);
      return;
    }

    for (const file of data.files) {
      const item = document.createElement("div");
      item.className = "file-item";

      const info = document.createElement("div");
      info.className = "file-info";
      info.innerHTML = `
        <div class="file-name">${file.name}</div>
        <div class="file-path">${file.path}</div>
      `;

      const actions = document.createElement("div");
      actions.className = "file-actions";

      const loadBtn = document.createElement("button");
      loadBtn.className = "btn";
      loadBtn.textContent = "Load";
      loadBtn.addEventListener("click", () => loadLibraryObject(file.path, file.name));

      const downloadBtn = document.createElement("a");
      downloadBtn.className = "btn";
      downloadBtn.textContent = "Download";
      downloadBtn.href = `/api/download?path=${encodeURIComponent(file.path)}`;

      actions.append(loadBtn, downloadBtn);
      item.append(info, actions);
      listEl.appendChild(item);
    }
  } catch {
    rootEl.textContent = "Could not load local mesh library.";
    listEl.innerHTML = "";
  }
}

function setPairButtonsEnabled(pair) {
  currentPair = pair;
  document.getElementById("load-generated-btn").disabled = !pair.generated?.localPath;
  document.getElementById("load-gt-btn").disabled = !pair.gt?.localPath;
  document.getElementById("download-generated-btn").disabled = !pair.generated?.localPath;
  document.getElementById("download-gt-btn").disabled = !pair.gt?.localPath;
}

async function fetchPairByUid() {
  const uid = document.getElementById("uid-input").value.trim();
  const pairStatus = document.getElementById("pair-status");
  if (!uid) return;

  pairStatus.textContent = "Fetching generated + GT pair into local cache…";
  clearConsole();
  logToConsole(`Fetching pair for UID ${uid}…`, "info");

  try {
    const resp = await fetch(`/api/fetch/pair?uid=${encodeURIComponent(uid)}`);
    const data = await resp.json();
    if (!data.success) {
      throw new Error(data.error || "Pair fetch failed");
    }

    setPairButtonsEnabled({
      uid,
      generated: data.generated,
      gt: data.gt,
    });

    pairStatus.textContent =
      `Generated: ${data.status.generated_available ? "available" : "missing"} | ` +
      `GT: ${data.status.gt_available ? data.status.gt_format : "missing"}`;

    if (data.generated?.localPath) {
      logToConsole(`Cached generated mesh: ${data.generated.localPath}`, "info");
    }
    if (data.gt?.localPath) {
      logToConsole(`Cached GT mesh: ${data.gt.localPath}`, "info");
    }

    if (data.generated?.localPath) {
      await loadLibraryObject(data.generated.localPath, data.generated.filename);
    } else if (data.gt?.localPath) {
      await loadLibraryObject(data.gt.localPath, data.gt.filename);
    }
    await refreshFileList();
  } catch (err) {
    pairStatus.textContent = err.message;
    logToConsole(`Remote fetch error: ${err.message}`, "error");
    setPairButtonsEnabled({ uid, generated: null, gt: null });
  }
}

document.getElementById("fetch-pair-btn").addEventListener("click", fetchPairByUid);
document.getElementById("uid-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    fetchPairByUid();
  }
});

document.getElementById("load-generated-btn").addEventListener("click", async () => {
  if (currentPair.generated?.localPath) {
    await loadLibraryObject(currentPair.generated.localPath, currentPair.generated.filename);
  }
});

document.getElementById("load-gt-btn").addEventListener("click", async () => {
  if (currentPair.gt?.localPath) {
    await loadLibraryObject(currentPair.gt.localPath, currentPair.gt.filename);
  }
});

document.getElementById("download-generated-btn").addEventListener("click", () => {
  if (currentPair.generated?.localPath) {
    window.location.href = `/api/download?path=${encodeURIComponent(currentPair.generated.localPath)}`;
  }
});

document.getElementById("download-gt-btn").addEventListener("click", () => {
  if (currentPair.gt?.localPath) {
    window.location.href = `/api/download?path=${encodeURIComponent(currentPair.gt.localPath)}`;
  }
});

document.getElementById("btn-wireframe").addEventListener("click", function () {
  wireframeActive = !wireframeActive;
  this.classList.toggle("active", wireframeActive);
  if (currentMesh) {
    applyDisplayMaterials(currentMesh);
    if (normalsActive) showNormals();
  }
});

document.getElementById("btn-normals").addEventListener("click", function () {
  normalsActive = !normalsActive;
  this.classList.toggle("active", normalsActive);
  showNormals();
});

document.getElementById("btn-rotate").addEventListener("click", function () {
  autoRotate = !autoRotate;
  this.classList.toggle("active", autoRotate);
});

document.getElementById("btn-reset").addEventListener("click", () => {
  controls.target.set(0, 0, 0);
  camera.position.set(2, 1.5, 2);
  controls.update();
});

const resizer = document.getElementById("resizer");
const leftPanel = document.getElementById("left-panel");
let isResizing = false;

resizer.addEventListener("mousedown", (e) => {
  isResizing = true;
  document.body.style.cursor = "col-resize";
  document.body.style.userSelect = "none";
  e.preventDefault();
});

document.addEventListener("mousemove", (e) => {
  if (!isResizing) return;
  const newWidth = Math.max(280, Math.min(e.clientX, window.innerWidth - 300));
  leftPanel.style.width = `${newWidth}px`;
  onResize();
});

document.addEventListener("mouseup", () => {
  if (!isResizing) {
    return;
  }
  isResizing = false;
  document.body.style.cursor = "";
  document.body.style.userSelect = "";
  onResize();
});

async function checkBlenderStatus() {
  try {
    const [blenderResp, remoteResp] = await Promise.all([
      fetch("/api/blender-status"),
      fetch("/api/remote-status"),
    ]);
    const blender = await blenderResp.json();
    const remote = await remoteResp.json();

    const statusEl = document.getElementById("blender-status");
    if (blender.available) {
      statusEl.classList.add("available");
      statusEl.querySelector("span").textContent = "Blender ready";
    } else {
      statusEl.querySelector("span").textContent = "Blender not found";
    }

    const remoteStatusEl = document.getElementById("remote-status");
    if (remote.enabled) {
      remoteStatusEl.textContent =
        `Remote fetch ready. Cache: ${remote.cacheRoot} (${(remote.cacheMaxBytes / 1024 ** 3).toFixed(1)} GB max).`;
    } else {
      remoteStatusEl.textContent =
        "Reward API fetch is not configured. Set OBJ_VIEWER_REWARD_API or LLM3D_MODAL__ENDPOINT.";
    }
  } catch {
    document.getElementById("blender-status").querySelector("span").textContent = "Status unknown";
    document.getElementById("remote-status").textContent = "Remote fetch status unavailable.";
  }
}

document.getElementById("refresh-files-btn").addEventListener("click", refreshFileList);
setPairButtonsEnabled({ uid: "", generated: null, gt: null });
checkBlenderStatus();
refreshFileList();
