import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

contextBridge.exposeInMainWorld('electron', electronAPI)

contextBridge.exposeInMainWorld('api', {
  startWebcam:  ()  => ipcRenderer.invoke('start-webcam'),
  startVideo:   ()  => ipcRenderer.invoke('start-video'),
  stopPipeline: ()  => ipcRenderer.invoke('stop-pipeline'),
  onMessage:    (cb) => ipcRenderer.on('py-message', (_e, msg) => cb(msg)),
  offMessage:   ()  => ipcRenderer.removeAllListeners('py-message')
})