const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fetch = require("node-fetch");

function createWindow() {
  const win = new BrowserWindow({
    width: 600,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true
    }
  });
  win.loadFile("index.html");
}

app.whenReady().then(createWindow);
