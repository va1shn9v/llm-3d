import { tmpdir } from "os";
import { basename, extname, join, relative, resolve, sep } from "path";
import { existsSync } from "fs";
import {
  copyFile,
  mkdir,
  readFile,
  readdir,
  rm,
  stat,
  unlink,
  utimes,
  writeFile,
} from "fs/promises";
import { randomUUID } from "crypto";

const PORT = parseInt(process.env.PORT || "3333");
const BLENDER_PATH = await findBlender();
const BROWSE_ROOT = await resolveBrowseRoot();
const CACHE_ROOT = await resolveCacheRoot(BROWSE_ROOT);
const LIBRARY_ROOT = BROWSE_ROOT || CACHE_ROOT;
const CACHE_MAX_BYTES = parseCacheMaxBytes();
const REMOTE_ENDPOINT = normalizeEndpoint(
  process.env.OBJ_VIEWER_REWARD_API || process.env.LLM3D_MODAL__ENDPOINT || ""
);
const REMOTE_TOKEN = process.env.OBJ_VIEWER_REWARD_TOKEN || process.env.LLM3D_MODAL__AUTH_TOKEN || "";
const VIEWABLE_EXTENSIONS = new Set([".obj", ".glb", ".gltf"]);

if (BLENDER_PATH) {
  console.log(`[blender] Found at: ${BLENDER_PATH}`);
} else {
  console.warn(
    "[blender] Not found — Python code execution disabled. " +
      "Install Blender and ensure it's in PATH, or set BLENDER_PATH env var."
  );
}

if (!REMOTE_ENDPOINT) {
  console.warn(
    "[remote] Reward API endpoint not configured. " +
      "Set OBJ_VIEWER_REWARD_API or LLM3D_MODAL__ENDPOINT to enable on-demand fetch."
  );
}

await mkdir(CACHE_ROOT, { recursive: true });

async function findBlender(): Promise<string | null> {
  if (process.env.BLENDER_PATH) {
    try {
      const proc = Bun.spawn([process.env.BLENDER_PATH, "--version"], {
        stdout: "pipe",
        stderr: "pipe",
      });
      if ((await proc.exited) === 0) return process.env.BLENDER_PATH;
    } catch {}
  }

  const candidates = [
    "blender",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/Applications/Blender.app/Contents/MacOS/blender",
    "/opt/homebrew/bin/blender",
    "/usr/local/bin/blender",
    "/opt/blender/blender",
  ];

  for (const bin of candidates) {
    try {
      const proc = Bun.spawn([bin, "--version"], {
        stdout: "pipe",
        stderr: "pipe",
      });
      if ((await proc.exited) === 0) return bin;
    } catch {}
  }
  return null;
}

function normalizeEndpoint(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

async function resolveBrowseRoot(): Promise<string | null> {
  const argRoot = process.argv.slice(2).find((arg) => arg && !arg.startsWith("-"));
  const configuredRoot = process.env.OBJ_VIEWER_DIR || argRoot;
  if (!configuredRoot) return null;

  const root = resolve(process.cwd(), configuredRoot);
  try {
    const info = await stat(root);
    return info.isDirectory() ? root : null;
  } catch {
    return null;
  }
}

async function resolveCacheRoot(browseRoot: string | null): Promise<string> {
  const configured = process.env.OBJ_VIEWER_CACHE_DIR;
  if (configured) {
    return resolve(process.cwd(), configured);
  }
  if (browseRoot) {
    return join(browseRoot, ".obj-viewer-cache");
  }
  return resolve(process.cwd(), ".obj-viewer-cache");
}

function parseCacheMaxBytes(): number {
  const rawGb = parseFloat(process.env.OBJ_VIEWER_CACHE_MAX_GB || "10");
  if (!Number.isFinite(rawGb) || rawGb <= 0) {
    return 10 * 1024 ** 3;
  }
  return Math.floor(rawGb * 1024 ** 3);
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function isViewableFile(path: string): boolean {
  return VIEWABLE_EXTENSIONS.has(extname(path).toLowerCase());
}

function resolveLibraryPath(relPath: string): string | null {
  if (!LIBRARY_ROOT || !relPath) return null;
  const normalized = resolve(LIBRARY_ROOT, relPath);
  const rel = relative(LIBRARY_ROOT, normalized);
  if (rel.startsWith("..") || rel.includes(`..${sep}`)) {
    return null;
  }
  return normalized;
}

async function listLibraryFiles(root: string): Promise<Array<{
  name: string;
  path: string;
  size: number;
  mtimeMs: number;
}>> {
  const entries: Array<{ name: string; path: string; size: number; mtimeMs: number }> = [];

  async function walk(dir: string) {
    for (const dirent of await readdir(dir, { withFileTypes: true })) {
      const fullPath = join(dir, dirent.name);
      if (dirent.isDirectory()) {
        await walk(fullPath);
        continue;
      }
      if (!dirent.isFile() || !isViewableFile(fullPath)) {
        continue;
      }
      const info = await stat(fullPath);
      entries.push({
        name: basename(fullPath),
        path: relative(root, fullPath),
        size: info.size,
        mtimeMs: info.mtimeMs,
      });
    }
  }

  await walk(root);
  entries.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return entries;
}

async function listCacheFiles(root: string): Promise<Array<{
  fullPath: string;
  size: number;
  mtimeMs: number;
}>> {
  const entries: Array<{ fullPath: string; size: number; mtimeMs: number }> = [];

  async function walk(dir: string) {
    for (const dirent of await readdir(dir, { withFileTypes: true })) {
      const fullPath = join(dir, dirent.name);
      if (dirent.isDirectory()) {
        await walk(fullPath);
        continue;
      }
      const info = await stat(fullPath);
      entries.push({
        fullPath,
        size: info.size,
        mtimeMs: info.mtimeMs,
      });
    }
  }

  await walk(root);
  return entries;
}

async function enforceCacheBudget(): Promise<void> {
  const files = await listCacheFiles(CACHE_ROOT);
  let total = files.reduce((sum, file) => sum + file.size, 0);
  if (total <= CACHE_MAX_BYTES) {
    return;
  }

  files.sort((a, b) => a.mtimeMs - b.mtimeMs);
  for (const file of files) {
    if (total <= CACHE_MAX_BYTES) {
      break;
    }
    await unlink(file.fullPath).catch(() => {});
    total -= file.size;
  }
}

async function persistToCache(
  kind: "generated" | "gt",
  uid: string,
  filename: string,
  data: Uint8Array,
): Promise<string | null> {
  const ext = extname(filename).toLowerCase() || (kind === "generated" ? ".obj" : ".bin");
  const dir = join(CACHE_ROOT, kind);
  await mkdir(dir, { recursive: true });
  const filePath = join(dir, `${uid}${ext}`);
  await writeFile(filePath, data);
  const now = new Date();
  await utimes(filePath, now, now).catch(() => {});
  await enforceCacheBudget();

  if (!LIBRARY_ROOT) {
    return null;
  }
  const rel = relative(LIBRARY_ROOT, filePath);
  return rel.startsWith("..") ? null : rel;
}

function remotePath(kind: "generated" | "gt" | "pair", uid: string): string {
  if (kind === "pair") return `/artifacts/pair/${uid}`;
  return `/artifacts/${kind}/${uid}`;
}

function buildRemoteUrl(path: string): string {
  if (!REMOTE_ENDPOINT) {
    throw new Error("Reward API endpoint is not configured.");
  }
  const url = new URL(`${REMOTE_ENDPOINT}${path}`);
  if (REMOTE_TOKEN) {
    url.searchParams.set("token", REMOTE_TOKEN);
  }
  return url.toString();
}

function parseContentDispositionFilename(header: string | null, fallback: string): string {
  if (!header) return fallback;
  const match = header.match(/filename="?([^"]+)"?/i);
  return match?.[1] || fallback;
}

async function fetchRemoteBytes(kind: "generated" | "gt", uid: string): Promise<{
  bytes: Uint8Array;
  filename: string;
}> {
  const resp = await fetch(buildRemoteUrl(remotePath(kind, uid)));
  if (!resp.ok) {
    const message = await resp.text().catch(() => "");
    throw new Error(message || `Remote ${kind} fetch failed with status ${resp.status}`);
  }
  const bytes = new Uint8Array(await resp.arrayBuffer());
  const fallbackName = kind === "generated" ? `${uid}.obj` : `${uid}.bin`;
  const filename = parseContentDispositionFilename(resp.headers.get("content-disposition"), fallbackName);
  return { bytes, filename };
}

async function fetchRemotePairStatus(uid: string): Promise<any> {
  const resp = await fetch(buildRemoteUrl(remotePath("pair", uid)));
  if (!resp.ok) {
    const message = await resp.text().catch(() => "");
    throw new Error(message || `Remote pair lookup failed with status ${resp.status}`);
  }
  return await resp.json();
}

async function ensureRemoteArtifact(kind: "generated" | "gt", uid: string): Promise<{
  uid: string;
  kind: "generated" | "gt";
  available: boolean;
  localPath: string | null;
  filename: string | null;
  error?: string;
}> {
  try {
    const { bytes, filename } = await fetchRemoteBytes(kind, uid);
    const localPath = await persistToCache(kind, uid, filename, bytes);
    return { uid, kind, available: true, localPath, filename };
  } catch (error: any) {
    return {
      uid,
      kind,
      available: false,
      localPath: null,
      filename: null,
      error: error?.message || String(error),
    };
  }
}

// Blender 4.x compatibility replacements (ported from blender_worker.py)
const CODE_REPLACEMENTS: [string, string][] = [
  ["'BLENDER_EEVEE'", "'BLENDER_EEVEE_NEXT'"],
  ['"BLENDER_EEVEE"', '"BLENDER_EEVEE_NEXT"'],
  ["bpy.ops.export_scene.obj(", "bpy.ops.wm.obj_export("],
  ['.inputs["Specular"]', '.inputs["Specular IOR Level"]'],
  [".inputs['Specular']", ".inputs['Specular IOR Level']"],
  ['.inputs["Clearcoat"]', '.inputs["Coat Weight"]'],
  [".inputs['Clearcoat']", ".inputs['Coat Weight']"],
  ['.inputs["Sheen"]', '.inputs["Sheen Weight"]'],
  [".inputs['Sheen']", ".inputs['Sheen Weight']"],
];

const CODE_LINE_REMOVALS = ["use_auto_smooth", "auto_smooth_angle"];

function sanitizeCode(code: string): string {
  for (const [old, rep] of CODE_REPLACEMENTS) {
    code = code.replaceAll(old, rep);
  }
  return code
    .split("\n")
    .map((line) => {
      const stripped = line.trim();
      if (
        CODE_LINE_REMOVALS.some((pat) => stripped.includes(pat)) &&
        !stripped.startsWith("#")
      ) {
        return line.replace(stripped, `pass  # removed: ${stripped}`);
      }
      return line;
    })
    .join("\n");
}

function buildWrapperScript(userCode: string, exportPath: string): string {
  const escapedPath = exportPath.replace(/\\/g, "\\\\");
  return `\
import sys, os

os.environ["EXPORT_PATH"] = "${escapedPath}"

# ===== BLENDER 4.x COMPATIBILITY SHIM =====
import bpy as _bpy_shim

_OrigMeshType = type(_bpy_shim.data.meshes.new("__probe"))
_bpy_shim.data.meshes.remove(_bpy_shim.data.meshes["__probe"])

if not hasattr(_OrigMeshType, "use_auto_smooth"):
    _OrigMeshType.use_auto_smooth = property(lambda self: False, lambda self, v: None)
    _OrigMeshType.auto_smooth_angle = property(lambda self: 0.0, lambda self, v: None)

if not hasattr(_bpy_shim.ops.export_scene, "obj"):
    def _legacy_obj_export(**kw):
        filepath = kw.pop("filepath", kw.pop("path", ""))
        use_selection = kw.pop("use_selection", False)
        return _bpy_shim.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=use_selection,
        )
    _bpy_shim.ops.export_scene.obj = _legacy_obj_export

import mathutils as _mathutils_mod
if not hasattr(_bpy_shim, "mathutils"):
    _bpy_shim.mathutils = _mathutils_mod

del _OrigMeshType, _bpy_shim, _mathutils_mod
# ===== END COMPATIBILITY SHIM =====

# ===== BEGIN USER CODE =====
${userCode}
# ===== END USER CODE =====

# ===== AUTO-EXPORT FALLBACK =====
import bpy as _bpy
import os as _os

_export_path = "${escapedPath}"

def _apply_modifiers_for_export():
    for obj in list(_bpy.data.objects):
        if obj.type != 'MESH':
            continue
        try:
            _bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            for mod in list(obj.modifiers):
                try:
                    _bpy.ops.object.modifier_apply(modifier=mod.name)
                except Exception:
                    pass
            obj.select_set(False)
        except Exception:
            pass

def _robust_export(path):
    mesh_objects = [o for o in _bpy.data.objects if o.type == 'MESH']
    curve_objects = [o for o in _bpy.data.objects if o.type == 'CURVE']
    exportable = mesh_objects + curve_objects
    if not exportable:
        return False
    _bpy.ops.object.select_all(action='DESELECT')
    for o in exportable:
        o.select_set(True)
    _bpy.context.view_layer.objects.active = exportable[0]
    try:
        _bpy.ops.wm.obj_export(
            filepath=path,
            export_selected_objects=True,
            export_uv=False,
            export_normals=True,
            export_materials=False,
            apply_modifiers=True,
        )
        return True
    except Exception:
        return False

_needs_export = True
if _os.path.exists(_export_path):
    _fsize = _os.path.getsize(_export_path)
    if _fsize > 100:
        _needs_export = False
    else:
        _os.remove(_export_path)

if _needs_export:
    _apply_modifiers_for_export()
    _robust_export(_export_path)
`;
}

async function executeBlenderCode(
  code: string
): Promise<{
  success: boolean;
  objData?: string;
  error?: string;
  stdout: string;
  stderr: string;
  savedPath?: string | null;
}> {
  if (!BLENDER_PATH) {
    return {
      success: false,
      error:
        "Blender not found. Install Blender and add it to PATH, or set BLENDER_PATH env var.",
      stdout: "",
      stderr: "",
    };
  }

  const workDir = join(tmpdir(), `obj-viewer-${randomUUID()}`);
  await mkdir(workDir, { recursive: true });

  const exportPath = join(workDir, "output.obj");
  const scriptPath = join(workDir, "script.py");

  const sanitized = sanitizeCode(code);
  const wrapper = buildWrapperScript(sanitized, exportPath);
  await writeFile(scriptPath, wrapper);

  try {
    const proc = Bun.spawn([BLENDER_PATH, "--background", "--python", scriptPath], {
      stdout: "pipe",
      stderr: "pipe",
      cwd: workDir,
      env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
    });

    const timeout = setTimeout(() => proc.kill(), 120_000);
    const exitCode = await proc.exited;
    clearTimeout(timeout);

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();

    if (!existsSync(exportPath)) {
      return {
        success: false,
        error: `No output mesh produced (exit code ${exitCode}).`,
        stdout: stdout.slice(-2000),
        stderr: stderr.slice(-2000),
      };
    }

    const objData = await readFile(exportPath, "utf-8");
    const savedPath = await persistToCache(
      "generated",
      `generated-${randomUUID()}`,
      "generated.obj",
      new Uint8Array(Buffer.from(objData, "utf-8"))
    );

    return {
      success: true,
      objData,
      stdout: stdout.slice(-2000),
      stderr: stderr.slice(-2000),
      savedPath,
    };
  } catch (err: any) {
    return {
      success: false,
      error: err.message || String(err),
      stdout: "",
      stderr: "",
    };
  } finally {
    await rm(workDir, { recursive: true, force: true }).catch(() => {});
  }
}

const dir = import.meta.dir;

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(Bun.file(join(dir, "index.html")));
    }

    if (url.pathname === "/viewer.js") {
      return new Response(Bun.file(join(dir, "viewer.js")), {
        headers: { "Content-Type": "application/javascript" },
      });
    }

    if (url.pathname === "/api/blender-status") {
      return Response.json({
        available: !!BLENDER_PATH,
        path: BLENDER_PATH,
        libraryRoot: LIBRARY_ROOT,
        cacheRoot: CACHE_ROOT,
      });
    }

    if (url.pathname === "/api/remote-status") {
      return Response.json({
        enabled: !!REMOTE_ENDPOINT,
        endpoint: REMOTE_ENDPOINT || null,
        tokenConfigured: !!REMOTE_TOKEN,
        cacheRoot: CACHE_ROOT,
        cacheMaxBytes: CACHE_MAX_BYTES,
      });
    }

    if (url.pathname === "/api/files") {
      if (!LIBRARY_ROOT) {
        return Response.json({ root: null, files: [] });
      }
      return Response.json({ root: LIBRARY_ROOT, files: await listLibraryFiles(LIBRARY_ROOT) });
    }

    if (url.pathname === "/api/file") {
      const relPath = url.searchParams.get("path") || "";
      const filePath = resolveLibraryPath(relPath);
      if (!filePath || !existsSync(filePath)) {
        return Response.json({ success: false, error: "File not found" }, { status: 404 });
      }

      const format = extname(filePath).toLowerCase().replace(/^\./, "");
      const bytes = await readFile(filePath);
      await utimes(filePath, new Date(), new Date()).catch(() => {});

      if (format === "obj") {
        return Response.json({
          success: true,
          name: basename(filePath),
          format,
          textData: bytes.toString("utf-8"),
        });
      }

      return Response.json({
        success: true,
        name: basename(filePath),
        format,
        dataB64: Buffer.from(bytes).toString("base64"),
      });
    }

    if (url.pathname === "/api/download") {
      const relPath = url.searchParams.get("path") || "";
      const filePath = resolveLibraryPath(relPath);
      if (!filePath || !existsSync(filePath)) {
        return new Response("File not found", { status: 404 });
      }
      return new Response(Bun.file(filePath), {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition": `attachment; filename="${basename(filePath)}"`,
        },
      });
    }

    if (url.pathname === "/api/fetch/generated") {
      const uid = (url.searchParams.get("uid") || "").trim();
      if (!uid) {
        return Response.json({ success: false, error: "uid is required" }, { status: 400 });
      }
      return Response.json(await ensureRemoteArtifact("generated", uid));
    }

    if (url.pathname === "/api/fetch/gt") {
      const uid = (url.searchParams.get("uid") || "").trim();
      if (!uid) {
        return Response.json({ success: false, error: "uid is required" }, { status: 400 });
      }
      return Response.json(await ensureRemoteArtifact("gt", uid));
    }

    if (url.pathname === "/api/fetch/pair") {
      const uid = (url.searchParams.get("uid") || "").trim();
      if (!uid) {
        return Response.json({ success: false, error: "uid is required" }, { status: 400 });
      }

      try {
        const status = await fetchRemotePairStatus(uid);
        const generated = status.generated_available
          ? await ensureRemoteArtifact("generated", uid)
          : { uid, kind: "generated", available: false, localPath: null, filename: null };
        const gt = status.gt_available
          ? await ensureRemoteArtifact("gt", uid)
          : { uid, kind: "gt", available: false, localPath: null, filename: null };

        return Response.json({
          success: true,
          uid,
          status,
          generated,
          gt,
        });
      } catch (error: any) {
        return Response.json(
          { success: false, error: error?.message || String(error) },
          { status: 500 }
        );
      }
    }

    if (url.pathname === "/api/execute" && req.method === "POST") {
      try {
        const body = await req.json();
        const code = body.code;
        if (!code || typeof code !== "string") {
          return Response.json({ success: false, error: "No code provided" }, { status: 400 });
        }
        const result = await executeBlenderCode(code);
        return Response.json(result);
      } catch (err: any) {
        return Response.json(
          { success: false, error: err.message || "Internal error" },
          { status: 500 }
        );
      }
    }

    const filePath = join(dir, url.pathname.slice(1));
    const file = Bun.file(filePath);
    if (await file.exists()) {
      return new Response(file);
    }

    return new Response("Not Found", { status: 404 });
  },
});

const browseMsg = BROWSE_ROOT ? `  Local browse root: ${BROWSE_ROOT}\n` : "";
console.log(
  `\n  🔍 OBJ Viewer running at http://localhost:${PORT}\n` +
    `${browseMsg}` +
    `  Local cache root: ${CACHE_ROOT} (${formatBytes(CACHE_MAX_BYTES)} max)\n`
);
