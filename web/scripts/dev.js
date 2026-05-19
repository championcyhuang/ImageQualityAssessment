const { spawn } = require("child_process");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..", "..");
const webDir = path.resolve(__dirname, "..");
const pythonExe = path.join(projectRoot, ".venv", "Scripts", "python.exe");

const next = spawn("pnpm", ["exec", "next", "dev"], {
  cwd: webDir,
  stdio: "inherit",
  shell: true,
});

const uvicorn = spawn(
  pythonExe,
  ["-m", "uvicorn", "api.main:app", "--reload", "--port", "8002"],
  {
    cwd: projectRoot,
    stdio: "inherit",
    shell: true,
  }
);

process.on("SIGINT", () => {
  next.kill();
  uvicorn.kill();
  process.exit(0);
});

process.on("SIGTERM", () => {
  next.kill();
  uvicorn.kill();
  process.exit(0);
});
