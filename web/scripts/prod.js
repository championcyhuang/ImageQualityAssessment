const { spawn, execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const projectRoot = path.resolve(__dirname, "..", "..");
const webDir = path.resolve(__dirname, "..");
const pythonExe = path.join(projectRoot, ".venv", "Scripts", "python.exe");

const NEXT_PORT = 3000;
const PREFERRED_API_PORT = 8002;

// ── Port helpers (same as dev.js) ────────────────────────────────

function isPortInUse(port) {
  try {
    execSync(`netstat -ano | findstr :${port}`, { stdio: "pipe", shell: true });
    return true;
  } catch {
    return false;
  }
}

function killPort(port) {
  try {
    const out = execSync(`netstat -ano | findstr :${port}`, {
      encoding: "utf8",
      shell: true,
    });
    const pids = new Set();
    for (const line of out.trim().split("\n")) {
      const parts = line.trim().split(/\s+/);
      const pid = parts[parts.length - 1];
      if (pid && /^\d+$/.test(pid) && pid !== "0") pids.add(pid);
    }
    for (const pid of pids) {
      try {
        execSync(`taskkill /PID ${pid} /F /T`, { stdio: "ignore", shell: true });
      } catch {}
    }
  } catch {}
}

// ── Clean up ports ───────────────────────────────────────────────

console.log("[prod] Cleaning up ports...");
killPort(NEXT_PORT);
killPort(PREFERRED_API_PORT);

const startWait = Date.now();
while (Date.now() - startWait < 1500) {}

// ── Pick API port ────────────────────────────────────────────────

let apiPort = PREFERRED_API_PORT;
if (isPortInUse(apiPort)) {
  apiPort = apiPort + 1;
  while (isPortInUse(apiPort) && apiPort < PREFERRED_API_PORT + 50) apiPort++;
  console.log(`[prod] Port ${PREFERRED_API_PORT} occupied, using ${apiPort}`);
} else {
  console.log(`[prod] Port ${apiPort} is free`);
}

// Write env for build / start
const envPath = path.join(webDir, ".env.local");
fs.writeFileSync(envPath, `NEXT_PUBLIC_API_BASE=http://localhost:${apiPort}\n`);

// ── Build if needed ──────────────────────────────────────────────

const nextDir = path.join(webDir, ".next");
if (!fs.existsSync(nextDir)) {
  console.log("[prod] No build found, running pnpm build...");
  try {
    execSync("pnpm exec next build", { cwd: webDir, stdio: "inherit", shell: true });
    console.log("[prod] Build complete.");
  } catch {
    console.error("[prod] Build failed. Fix errors and try again.");
    process.exit(1);
  }
} else {
  console.log("[prod] Using existing build. Run 'pnpm build' manually if code changed.");
}

// ── Spawn processes ──────────────────────────────────────────────

const next = spawn("pnpm", ["exec", "next", "start", "--port", String(NEXT_PORT)], {
  cwd: webDir,
  stdio: "inherit",
  shell: true,
});

const uvicorn = spawn(
  pythonExe,
  ["-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", String(apiPort)],
  {
    cwd: projectRoot,
    stdio: "inherit",
    shell: true,
  }
);

// ── Graceful shutdown ────────────────────────────────────────────

function shutdown() {
  console.log("\n[prod] Shutting down...");
  next.kill();
  uvicorn.kill();
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

if (process.platform === "win32") {
  const readline = require("readline").createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  readline.on("SIGINT", () => {
    readline.close();
    shutdown();
  });
}
