const { execSync } = require("child_process");

const PORTS = [3000, 8002];

function getPidsOnPort(port) {
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
    return Array.from(pids);
  } catch {
    return [];
  }
}

let killedAny = false;
let hadFailures = false;

for (const port of PORTS) {
  const pids = getPidsOnPort(port);
  if (pids.length === 0) {
    console.log(`[kill] Port ${port}: already free`);
    continue;
  }
  for (const pid of pids) {
    try {
      execSync(`taskkill /PID ${pid} /F /T`, { stdio: "ignore", shell: true });
      console.log(`[kill] Port ${port}: killed PID ${pid}`);
      killedAny = true;
    } catch {
      console.log(`[kill] Port ${port}: failed to kill PID ${pid}`);
      hadFailures = true;
    }
  }
}

// Fallback for zombie processes (Windows netstat ghost entries)
if (hadFailures) {
  console.log("[kill] Trying process name fallback for zombie processes...");
  try {
    execSync("taskkill /IM python.exe /F", { stdio: "ignore", shell: true });
    console.log("[kill] Killed all python.exe processes");
    killedAny = true;
  } catch {}
  try {
    execSync("taskkill /IM node.exe /F", { stdio: "ignore", shell: true });
    console.log("[kill] Killed all node.exe processes");
    killedAny = true;
  } catch {}
}

if (!killedAny) {
  console.log("[kill] Nothing to kill — all ports are free.");
} else {
  console.log("[kill] Done. Wait 2 seconds before starting pnpm dev.");
}
