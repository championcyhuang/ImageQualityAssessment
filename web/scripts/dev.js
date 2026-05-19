const { spawn, execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const projectRoot = path.resolve(__dirname, "..", "..");
const webDir = path.resolve(__dirname, "..");
const pythonExe = path.join(projectRoot, ".venv", "Scripts", "python.exe");

const NEXT_PORT = 3000;
const PREFERRED_API_PORT = 8002;

// ── Port helpers ─────────────────────────────────────────────────

function isPortInUse(port) {
  try {
    execSync(`netstat -ano | findstr :${port}`, { stdio: "pipe", shell: true });
    return true;
  } catch {
    return false;
  }
}

function findFreePort(start) {
  for (let port = start; port < start + 50; port++) {
    if (!isPortInUse(port)) return port;
  }
  throw new Error("No free port found in range " + start + "-" + (start + 50));
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
      if (pid && /^\d+$/.test(pid)) pids.add(pid);
    }
    for (const pid of pids) {
      try {
        execSync(`taskkill /PID ${pid} /F /T`, { stdio: "ignore", shell: true });
      } catch {}
    }
  } catch {}
}

// ── Clean up ports ───────────────────────────────────────────────

console.log("[dev] Cleaning up ports...");
killPort(NEXT_PORT);
killPort(PREFERRED_API_PORT);

// Give Windows a moment to release handles
const startWait = Date.now();
while (Date.now() - startWait < 1500) {
  // busy-wait ~1.5s
}

// ── Pick API port ────────────────────────────────────────────────

let apiPort = PREFERRED_API_PORT;
if (isPortInUse(apiPort)) {
  apiPort = findFreePort(apiPort + 1);
  console.log(`[dev] Port ${PREFERRED_API_PORT} still occupied, using ${apiPort}`);
} else {
  console.log(`[dev] Port ${apiPort} is free`);
}

// Write env file so Next.js can pick it up
const envPath = path.join(webDir, ".env.local");
fs.writeFileSync(envPath, `NEXT_PUBLIC_API_BASE=http://localhost:${apiPort}\n`);

// ── Spawn processes ──────────────────────────────────────────────

const next = spawn("pnpm", ["exec", "next", "dev", "--port", String(NEXT_PORT)], {
  cwd: webDir,
  stdio: "inherit",
  shell: true,
});

const uvicorn = spawn(
  pythonExe,
  ["-m", "uvicorn", "api.main:app", "--reload", "--port", String(apiPort)],
  {
    cwd: projectRoot,
    stdio: "inherit",
    shell: true,
  }
);

// ── Graceful shutdown ────────────────────────────────────────────

function shutdown() {
  console.log("\n[dev] Shutting down...");
  next.kill();
  uvicorn.kill();
  try {
    fs.unlinkSync(envPath);
  } catch {}
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

// Windows doesn't forward SIGINT to child processes properly
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
