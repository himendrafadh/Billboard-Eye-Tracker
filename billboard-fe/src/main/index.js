import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { spawn } from 'child_process'
import readline from 'readline'

let pyProcess = null

// ── path Python backend ──────────────────────────────────────────────────
const PYTHON_DEV = 'C:\\ProgramData\\miniconda3\\envs\\billboard\\python.exe'
const MAIN_PY    = join(app.getAppPath(), '..', 'main.py')

const BACKEND_EXE = app.isPackaged
  ? join(process.resourcesPath, 'billboard_backend', 'billboard_backend.exe')
  : null

// ── buat window ──────────────────────────────────────────────────────────
function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 960,
    minHeight: 600,
    show: false,
    autoHideMenuBar: true,
    title: 'Billboard Eye Tracker',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => mainWindow.show())

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  mainWindow.on('closed', () => stopPython())

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  return mainWindow
}

// ── spawn / stop Python ──────────────────────────────────────────────────
function startPython(source, win) {
  if (pyProcess) stopPython()

  const cmd  = app.isPackaged ? BACKEND_EXE : PYTHON_DEV
  const args = app.isPackaged ? [] : ['-u', MAIN_PY]
  if (source !== undefined && source !== null) args.push(String(source))

  const cwd = app.isPackaged
    ? join(process.resourcesPath, 'billboard_backend')
    : join(app.getAppPath(), '..')

  console.log('[startPython]', cmd, args.join(' '), '| cwd:', cwd)

  pyProcess = spawn(cmd, args, { cwd, env: { ...process.env } })

  const rl = readline.createInterface({ input: pyProcess.stdout })
  rl.on('line', (line) => {
    try {
      const msg = JSON.parse(line)
      win.webContents.send('py-message', msg)
    } catch (_) {}
  })

  pyProcess.stderr.on('data', (data) => {
    console.error('[Python stderr]', data.toString().trim())
  })

  pyProcess.on('close', (code) => {
    pyProcess = null
    win.webContents.send('py-message', { type: 'done', message: `exited (${code})` })
  })
}

function stopPython() {
  if (pyProcess) {
    pyProcess.kill()
    pyProcess = null
  }
}

// ── IPC handlers ─────────────────────────────────────────────────────────
function setupIPC(win) {
  ipcMain.handle('start-webcam', () => startPython(0, win))
  ipcMain.handle('stop-pipeline', () => stopPython())
  ipcMain.handle('start-video', async () => {
    const result = await dialog.showOpenDialog(win, {
      title: 'Pilih file video',
      filters: [{ name: 'Video', extensions: ['mp4', 'avi', 'mov', 'mkv'] }],
      properties: ['openFile']
    })
    if (!result.canceled && result.filePaths.length > 0) {
      startPython(result.filePaths[0], win)
      return result.filePaths[0]
    }
    return null
  })
}

// ── lifecycle ────────────────────────────────────────────────────────────
app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.billboard.eyetracker')
  app.on('browser-window-created', (_, window) => optimizer.watchWindowShortcuts(window))

  const win = createWindow()
  setupIPC(win)

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  stopPython()
  if (process.platform !== 'darwin') app.quit()
})