import json
import re
import cv2
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = ROOT / 'projects'
PROJECTS_REGISTRY = ROOT / 'projects.json'
SHARED_VIDEOS_DIR = ROOT / 'videos'
PROJECT_NAME_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
ASSIGNED_VIDEO_PATTERN = re.compile(r'^camera_(\d{6})\.(mp4|mov|avi|mkv)$', re.IGNORECASE)
VIDEO_PATTERN = re.compile(r'^output_camera_(\d+)\.mp4$', re.IGNORECASE)
SUPPORTED_VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}
REGISTRY_LOCK = threading.RLock()
ASSIGNMENT_LOCK = threading.RLock()
RUN_LOCK = threading.RLock()
RUN_STATES = {}
RUN_LOG_LIMIT = 250


def _natural_sort_key(value):
  return [
    int(part) if part.isdigit() else part.lower()
    for part in re.split(r'(\d+)', value)
  ]


def _is_safe_leaf(value):
  return bool(
    isinstance(value, str)
    and value not in {'', '.', '..'}
    and '/' not in value
    and '\\' not in value
    and '\0' not in value
    and Path(value).name == value
  )


def _is_within(path, parent):
  try:
    path.resolve().relative_to(parent.resolve())
    return True
  except (OSError, RuntimeError, ValueError):
    return False


def _supported_video(path):
  return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Person Tracking Outputs</title>
  <style>
    :root {
      color-scheme: dark;
      --background: #101419;
      --panel: #1a2028;
      --border: #303944;
      --text: #f2f5f7;
      --muted: #9da8b4;
      --accent: #62d6b1;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--background);
      color: var(--text);
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      padding: 1.5rem 2rem 1rem;
    }
    h1 { margin: 0; font-size: 1.35rem; }
    .status { color: var(--muted); font-size: 0.9rem; }
    .project-picker { display: flex; align-items: center; gap: 0.55rem; }
    .project-picker select {
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 0.4rem 0.6rem;
      background: var(--panel);
      color: var(--text);
    }
    .run-button {
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 0.45rem 0.7rem;
      background: #285844;
      color: var(--text);
      cursor: pointer;
    }
    .delete-button {
      border: 1px solid #75414a;
      border-radius: 5px;
      padding: 0.45rem 0.7rem;
      background: #4a252b;
      color: var(--text);
      cursor: pointer;
    }
    .run-button:disabled, .delete-button:disabled { opacity: 0.55; cursor: not-allowed; }
    main { padding: 0 2rem 2rem; }
    .project-tools { margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); }
    .tool-row { display: flex; align-items: end; flex-wrap: wrap; gap: 0.75rem; padding: 0.8rem 0; }
    .tool-row label { display: grid; gap: 0.3rem; color: var(--muted); font-size: 0.85rem; }
    .tool-row input, .tool-row select, .slot-row select {
      min-width: 12rem;
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 0.45rem 0.6rem;
      background: var(--panel);
      color: var(--text);
    }
    .tool-row button, .slot-row button {
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 0.45rem 0.7rem;
      background: #252e38;
      color: var(--text);
      cursor: pointer;
    }
    .slots { display: grid; gap: 0.6rem; padding: 0.2rem 0 1rem; }
    .slot-row { display: grid; grid-template-columns: 6rem minmax(12rem, 1fr) auto minmax(12rem, 1fr) auto; align-items: center; gap: 0.6rem; }
    .slot-row input[type="file"] { min-width: 0; max-width: 100%; color: var(--muted); }
    .slot-assignment { grid-column: 2 / -1; color: var(--muted); font-size: 0.8rem; }
    .run-panel { padding: 0 0 1rem; }
    .run-state { margin: 0.2rem 0 0.5rem; font-size: 0.9rem; }
    .run-log { max-height: 10rem; overflow: auto; margin: 0; padding: 0.7rem; border: 1px solid var(--border); border-radius: 5px; background: #0b0e11; color: #c8d4d0; white-space: pre-wrap; overflow-wrap: anywhere; font: 0.78rem/1.45 Consolas, monospace; }
    .live-preview { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #07090b; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1rem;
    }
    .tile {
      overflow: hidden;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
    }
    .tile canvas { display: block; width: 100%; aspect-ratio: 16 / 9; background: #07090b; object-fit: contain; cursor: pointer; }
    .tile-label { display: block; padding: 0.8rem 1rem; font-weight: 650; }
    .controls { display: flex; align-items: center; gap: 0.65rem; padding: 0.2rem 0.8rem 0.8rem; }
    .controls button {
      min-width: 4.3rem;
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 0.35rem 0.55rem;
      background: #252e38;
      color: var(--text);
      cursor: pointer;
    }
    .controls input { flex: 1; min-width: 40px; accent-color: var(--accent); }
    .time { min-width: 3.5rem; color: var(--muted); font-size: 0.8rem; text-align: right; }
    .empty { color: var(--muted); padding: 2rem 0; }
    .viewer {
      position: fixed;
      inset: 0;
      z-index: 2;
      display: grid;
      place-items: center;
      padding: 4rem 2rem 2rem;
      background: rgba(5, 7, 9, 0.96);
    }
    .viewer[hidden] { display: none; }
    .viewer-content { width: min(1200px, 100%); }
    .viewer canvas { display: block; width: 100%; max-height: calc(100vh - 11rem); background: #000; object-fit: contain; }
    .viewer .controls { padding: 1rem 0; }
    .viewer-label { position: absolute; top: 1.25rem; left: 2rem; font-weight: 650; }
    .close {
      position: absolute;
      top: 0.9rem;
      right: 1.25rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.45rem 0.75rem;
      background: var(--panel);
      color: var(--text);
      cursor: pointer;
    }
    @media (max-width: 600px) {
      header, main { padding-left: 1rem; padding-right: 1rem; }
      header { display: block; }
      .project-picker { margin-top: 0.6rem; }
      .status { display: block; margin-top: 0.4rem; }
      .slot-row { grid-template-columns: 1fr; }
      .slot-assignment { grid-column: auto; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Processed Camera Outputs</h1>
    <label class="project-picker">Project
      <select id="project-select">
        <option value="">Select a project</option>
      </select>
    </label>
    <button class="run-button" id="run-detection" type="button" disabled>Run detection</button>
    <button class="delete-button" id="delete-project" type="button" disabled>Delete project</button>
    <span class="status" id="status">Loading videos...</span>
  </header>
  <main>
    <section class="project-tools">
      <form class="tool-row" id="new-project-form">
        <label>Project name<input id="new-project-name" required maxlength="64"></label>
        <label>Camera count<input id="new-project-camera-count" type="number" min="1" max="64" value="2" required></label>
        <button type="submit">Create project</button>
      </form>
      <div id="project-assignment" hidden>
        <div class="tool-row">
          <label>Shared video folder<select id="shared-folder-select"><option value="">Choose a folder</option></select></label>
          <button type="button" id="assign-all">Assign all from this folder</button>
        </div>
        <section class="slots" id="camera-slots" aria-label="Camera video assignments"></section>
      </div>
      <section class="run-panel" id="run-panel" hidden aria-live="polite">
        <p class="run-state" id="run-state">Run status: idle</p>
        <pre class="run-log" id="run-log" aria-label="Detection run output"></pre>
      </section>
    </section>
    <section class="grid" id="grid"></section>
  </main>

  <section class="viewer" id="viewer" hidden aria-label="Expanded video viewer">
    <button class="close" id="close" type="button">Back to grid</button>
    <strong class="viewer-label" id="viewer-label"></strong>
    <div class="viewer-content">
      <canvas id="expanded-canvas"></canvas>
      <div class="controls" id="expanded-controls"></div>
    </div>
  </section>

  <script>
    const grid = document.getElementById('grid');
    const status = document.getElementById('status');
    const projectSelect = document.getElementById('project-select');
    const runButton = document.getElementById('run-detection');
    const deleteProjectButton = document.getElementById('delete-project');
    const runPanel = document.getElementById('run-panel');
    const runStateLabel = document.getElementById('run-state');
    const runLog = document.getElementById('run-log');
    const projectAssignment = document.getElementById('project-assignment');
    const cameraSlots = document.getElementById('camera-slots');
    const sharedFolderSelect = document.getElementById('shared-folder-select');
    const viewer = document.getElementById('viewer');
    const expandedCanvas = document.getElementById('expanded-canvas');
    const expandedControls = document.getElementById('expanded-controls');
    const viewerLabel = document.getElementById('viewer-label');
    const states = new Map();
    let sharedVideos = [];
    let runPollTimer = null;
    const liveFrameTimers = new Map();
    let liveProjectName = '';
    let selectedProjectCameraCount = 0;

    async function requestJSON(url, options = {}) {
      const response = await fetch(url, options);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      return data;
    }

    function closeViewer() {
      const state = states.get(Number(viewer.dataset.cameraId));
      if (state) state.canvases = state.canvases.filter((canvas) => canvas !== expandedCanvas);
      expandedControls.replaceChildren();
      viewer.hidden = true;
    }

    function stopLivePreview() {
      liveFrameTimers.forEach((timer) => clearTimeout(timer));
      liveFrameTimers.clear();
      grid.querySelectorAll('img[data-object-url]').forEach((image) => URL.revokeObjectURL(image.dataset.objectUrl));
      liveProjectName = '';
    }

    function loadLivePreview(projectName, cameraCount) {
      if (liveProjectName === projectName && grid.querySelectorAll('.live-preview').length === cameraCount) return;
      stopLivePreview();
      states.forEach((state) => { state.playing = false; });
      states.clear();
      grid.replaceChildren();
      liveProjectName = projectName;
      for (let cameraId = 1; cameraId <= cameraCount; cameraId += 1) {
        const tile = document.createElement('article');
        tile.className = 'tile';
        const image = document.createElement('img');
        image.className = 'live-preview';
        image.alt = `Live preview for Camera ${cameraId}`;
        const label = document.createElement('span');
        label.className = 'tile-label';
        label.textContent = `Camera ${cameraId} (live)`;
        tile.append(image, label);
        grid.append(tile);
        pollLiveFrame(projectName, cameraId, image);
      }
    }

    async function pollLiveFrame(projectName, cameraId, image) {
      if (liveProjectName !== projectName || projectSelect.value !== projectName) return;
      try {
        const response = await fetch(
          `/api/projects/${encodeURIComponent(projectName)}/live-frame?camera_id=${cameraId}&t=${Date.now()}`,
          { cache: 'no-store' }
        );
        if (response.ok && response.status !== 204) {
          const objectUrl = URL.createObjectURL(await response.blob());
          const oldObjectUrl = image.dataset.objectUrl;
          if (oldObjectUrl) URL.revokeObjectURL(oldObjectUrl);
          image.dataset.objectUrl = objectUrl;
          image.src = objectUrl;
        }
      } catch (error) {
        runStateLabel.textContent = `Live preview reconnecting: ${error.message}`;
      }
      if (liveProjectName === projectName && projectSelect.value === projectName) {
        liveFrameTimers.set(cameraId, setTimeout(() => pollLiveFrame(projectName, cameraId, image), 250));
      }
    }

    function formatTime(frame, fps) {
      const seconds = Math.floor(frame / fps);
      return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
    }

    function updateControls(state) {
      state.controls.forEach(({ play, seek, time }) => {
        play.textContent = state.playing ? 'Pause' : 'Play';
        seek.value = state.frame;
        time.textContent = formatTime(state.frame, state.fps);
      });
    }

    async function showFrame(state, frame) {
      state.frame = Math.max(0, Math.min(frame, state.frameCount - 1));
      const params = new URLSearchParams({ project: projectSelect.value, camera_id: state.cameraId, frame: state.frame });
      const response = await fetch(`/api/frame?${params}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`Frame request failed: HTTP ${response.status}`);
      const bitmap = await createImageBitmap(await response.blob());
      state.canvases.forEach((canvas) => {
        canvas.width = state.width;
        canvas.height = state.height;
        canvas.getContext('2d').drawImage(bitmap, 0, 0, state.width, state.height);
      });
      bitmap.close();
      updateControls(state);
    }

    function makeControls(state, container) {
      const play = document.createElement('button');
      play.type = 'button';
      const seek = document.createElement('input');
      seek.type = 'range';
      seek.min = 0;
      seek.max = Math.max(0, state.frameCount - 1);
      seek.value = state.frame;
      const time = document.createElement('span');
      time.className = 'time';
      play.addEventListener('click', async () => {
        state.playing = !state.playing;
        updateControls(state);
        while (state.playing && state.frame < state.frameCount - 1) {
          await new Promise((resolve) => setTimeout(resolve, 1000 / state.fps));
          if (!state.playing) break;
          try {
            await showFrame(state, state.frame + 1);
          } catch (error) {
            state.playing = false;
            status.textContent = error.message;
          }
        }
        if (state.frame >= state.frameCount - 1) state.playing = false;
        updateControls(state);
      });
      seek.addEventListener('input', async () => {
        state.playing = false;
        try {
          await showFrame(state, Number(seek.value));
        } catch (error) {
          status.textContent = error.message;
        }
      });
      state.controls.push({ play, seek, time });
      container.replaceChildren(play, seek, time);
      updateControls(state);
    }

    function openViewer(state) {
      viewerLabel.textContent = state.label;
      viewer.dataset.cameraId = state.cameraId;
      state.canvases.push(expandedCanvas);
      makeControls(state, expandedControls);
      viewer.hidden = false;
      showFrame(state, state.frame).catch((error) => { status.textContent = error.message; });
    }

    async function loadVideos() {
      stopLivePreview();
      states.forEach((state) => { state.playing = false; });
      states.clear();
      grid.replaceChildren();
      const projectName = projectSelect.value;
      if (!projectName) {
        status.textContent = 'Select a project';
        return;
      }

      try {
        const response = await fetch(`/api/videos?project=${encodeURIComponent(projectName)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const videos = await response.json();
        status.textContent = `${videos.length} camera${videos.length === 1 ? '' : 's'} available`;
        if (!videos.length) {
          grid.innerHTML = '<p class="empty">No processed output videos found yet.</p>';
          return;
        }
        videos.forEach((item) => {
          const tile = document.createElement('article');
          tile.className = 'tile';
          const canvas = document.createElement('canvas');
          canvas.setAttribute('aria-label', `${item.label} video preview; click to expand`);
          const label = document.createElement('span');
          label.className = 'tile-label';
          label.textContent = item.label;
          const controls = document.createElement('div');
          controls.className = 'controls';
          const state = {
            cameraId: item.camera_id,
            label: item.label,
            frameCount: item.frame_count,
            fps: item.fps,
            width: item.width,
            height: item.height,
            frame: 0,
            playing: false,
            canvases: [canvas],
            controls: []
          };
          states.set(state.cameraId, state);
          makeControls(state, controls);
          canvas.addEventListener('click', () => openViewer(state));
          tile.append(canvas, label, controls);
          showFrame(state, 0).catch((error) => { status.textContent = error.message; });
          grid.appendChild(tile);
        });
      } catch (error) {
        status.textContent = 'Could not load videos';
        grid.innerHTML = `<p class="empty">${error.message}</p>`;
      }
    }

    async function loadSharedFolders() {
      try {
        const folders = await requestJSON('/api/shared-folders');
        sharedFolderSelect.replaceChildren(new Option('Choose a folder', ''));
        folders.forEach((folder) => {
          sharedFolderSelect.add(new Option(`${folder.name} (${folder.count} videos)`, folder.name));
        });
        const listings = await Promise.all(folders.map(async (folder) => {
          const filenames = await requestJSON(`/api/shared-folders/${encodeURIComponent(folder.name)}/videos`);
          return filenames.map((filename) => ({ folder: folder.name, filename }));
        }));
        sharedVideos = listings.flat();
      } catch (error) {
        status.textContent = `Could not load shared videos: ${error.message}`;
      }
    }

    function renderCameraSlots(project) {
      cameraSlots.replaceChildren();
      for (let slot = 1; slot <= project.camera_count; slot += 1) {
        const row = document.createElement('div');
        row.className = 'slot-row';
        const title = document.createElement('strong');
        title.textContent = `Camera ${slot}`;
        const sharedSelect = document.createElement('select');
        sharedSelect.setAttribute('aria-label', `Shared video for camera ${slot}`);
        sharedSelect.add(new Option('Choose shared video', ''));
        sharedVideos.forEach(({ folder, filename }) => {
          const option = new Option(`${folder}/${filename}`, JSON.stringify({ folder, filename }));
          sharedSelect.add(option);
        });
        const assignButton = document.createElement('button');
        assignButton.type = 'button';
        assignButton.textContent = 'Assign';
        assignButton.addEventListener('click', async () => {
          try {
            if (!sharedSelect.value) throw new Error('Choose a shared video first');
            const response = await requestJSON(`/api/projects/${encodeURIComponent(project.name)}/slots/${slot}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: sharedSelect.value
            });
            status.textContent = response.message;
            await loadSelectedProject();
          } catch (error) {
            status.textContent = error.message;
          }
        });
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.mp4,.mov,.avi,.mkv';
        fileInput.setAttribute('aria-label', `Upload video for camera ${slot}`);
        const uploadButton = document.createElement('button');
        uploadButton.type = 'button';
        uploadButton.textContent = 'Upload';
        uploadButton.addEventListener('click', async () => {
          try {
            const file = fileInput.files[0];
            if (!file) throw new Error('Choose a video file first');
            const response = await requestJSON(
              `/api/projects/${encodeURIComponent(project.name)}/slots/${slot}?filename=${encodeURIComponent(file.name)}`,
              { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file }
            );
            status.textContent = response.message;
            await loadSelectedProject();
          } catch (error) {
            status.textContent = error.message;
          }
        });
        const assignmentLabel = document.createElement('span');
        assignmentLabel.className = 'slot-assignment';
        const assignment = project.assignments.find((item) => item.camera_id === slot);
        assignmentLabel.textContent = assignment ? `Assigned: ${assignment.filename}` : 'Unassigned';
        row.append(title, sharedSelect, assignButton, fileInput, uploadButton, assignmentLabel);
        cameraSlots.append(row);
      }
    }

    function renderRunStatus(data) {
      runPanel.hidden = !projectSelect.value;
      runButton.disabled = !projectSelect.value || data.state === 'running';
      deleteProjectButton.disabled = !projectSelect.value || data.state === 'running';
      if (data.state === 'running') {
        runStateLabel.textContent = 'Detection status: running';
      } else if (data.state === 'done') {
        runStateLabel.textContent = `Detection completed successfully (exit ${data.exit_code})`;
      } else if (data.state === 'failed') {
        runStateLabel.textContent = `Detection failed (exit ${data.exit_code})`;
      } else {
        runStateLabel.textContent = 'Detection status: idle';
      }
      runLog.textContent = data.logs && data.logs.length ? data.logs.join('\\n') : 'No process output yet.';
      runLog.scrollTop = runLog.scrollHeight;
    }

    function scheduleRunPoll() {
      clearTimeout(runPollTimer);
      runPollTimer = setTimeout(() => loadRunStatus(), 1500);
    }

    async function loadRunStatus() {
      const projectName = projectSelect.value;
      if (!projectName) return;
      try {
        const data = await requestJSON(`/api/projects/${encodeURIComponent(projectName)}/run-status`);
        if (projectSelect.value !== projectName) return;
        renderRunStatus(data);
        if (data.state === 'running') {
          loadLivePreview(projectName, selectedProjectCameraCount);
          scheduleRunPoll();
        } else {
          clearTimeout(runPollTimer);
          stopLivePreview();
          await loadVideos();
        }
      } catch (error) {
        if (projectSelect.value === projectName) {
          runStateLabel.textContent = `Could not read run status: ${error.message}`;
          runButton.disabled = false;
          deleteProjectButton.disabled = false;
        }
      }
    }

    async function loadSelectedProject() {
      if (!viewer.hidden) closeViewer();
      const name = projectSelect.value;
      if (!name) {
        clearTimeout(runPollTimer);
        stopLivePreview();
        selectedProjectCameraCount = 0;
        runButton.disabled = true;
        deleteProjectButton.disabled = true;
        runPanel.hidden = true;
        projectAssignment.hidden = true;
        cameraSlots.replaceChildren();
        await loadVideos();
        return;
      }
      localStorage.setItem('selectedProject', name);
      runButton.disabled = true;
      deleteProjectButton.disabled = true;
      runPanel.hidden = false;
      try {
        const project = await requestJSON(`/api/projects/${encodeURIComponent(name)}`);
        if (projectSelect.value !== name) return;
        selectedProjectCameraCount = project.camera_count;
        projectAssignment.hidden = false;
        renderCameraSlots(project);
        await loadRunStatus();
      } catch (error) {
        status.textContent = error.message;
      }
    }

    async function loadProjects(preferredProject = null) {
      try {
        const currentProject = preferredProject !== null
          ? preferredProject
          : projectSelect.value || localStorage.getItem('selectedProject') || '';
        const projects = await requestJSON('/api/projects');
        projectSelect.replaceChildren(
          new Option('Select a project', ''),
          ...projects.map((name) => new Option(name, name))
        );
        projectSelect.value = projects.includes(currentProject) ? currentProject : '';
        if (projectSelect.value) await loadSelectedProject();
        else await loadVideos();
      } catch (error) {
        status.textContent = `Could not load projects: ${error.message}`;
      }
    }

    document.getElementById('new-project-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      try {
        const project = await requestJSON('/api/projects', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: document.getElementById('new-project-name').value.trim(),
            camera_count: Number(document.getElementById('new-project-camera-count').value)
          })
        });
        await loadProjects(project.name);
        status.textContent = `Created ${project.name}`;
      } catch (error) {
        status.textContent = error.message;
      }
    });

    runButton.addEventListener('click', async () => {
      const projectName = projectSelect.value;
      if (!projectName) return;
      runButton.disabled = true;
      runPanel.hidden = false;
      runStateLabel.textContent = 'Starting detection…';
      try {
        await requestJSON(`/api/projects/${encodeURIComponent(projectName)}/run`, { method: 'POST' });
        await loadRunStatus();
      } catch (error) {
        runStateLabel.textContent = error.message;
        await loadRunStatus();
      }
    });

    deleteProjectButton.addEventListener('click', async () => {
      const projectName = projectSelect.value;
      if (!projectName) return;
      const confirmed = window.confirm(
        `Permanently delete project "${projectName}" and all its videos, outputs, and re-identification data?`
      );
      if (!confirmed) return;
      deleteProjectButton.disabled = true;
      try {
        await requestJSON(`/api/projects/${encodeURIComponent(projectName)}`, { method: 'DELETE' });
        localStorage.removeItem('selectedProject');
        projectSelect.value = '';
        await loadProjects('');
        status.textContent = 'Select a project';
      } catch (error) {
        status.textContent = error.message;
        await loadRunStatus();
      }
    });

    document.getElementById('assign-all').addEventListener('click', async () => {
      const projectName = projectSelect.value;
      const folder = sharedFolderSelect.value;
      if (!projectName || !folder) {
        status.textContent = 'Select a project and shared folder';
        return;
      }
      if (!window.confirm('Replace this project’s current video assignments?')) return;
      try {
        const response = await requestJSON(`/api/projects/${encodeURIComponent(projectName)}/assign-folder`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ folder })
        });
        status.textContent = response.message;
        await loadSelectedProject();
      } catch (error) {
        status.textContent = error.message;
      }
    });

    projectSelect.addEventListener('change', loadSelectedProject);
    document.getElementById('close').addEventListener('click', closeViewer);
    viewer.addEventListener('click', (event) => {
      if (event.target === viewer) closeViewer();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !viewer.hidden) closeViewer();
    });
    loadSharedFolders().then(() => loadProjects());
  </script>
</body>
</html>"""


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
      parsed_path = urlparse(self.path)
      path = parsed_path.path
      parts = [unquote(part) for part in path.strip('/').split('/')]
      if path == '/':
        self._send_html(PAGE)
        return
      if path == '/api/projects':
        self._send_json(self._discover_projects())
        return
      if len(parts) == 3 and parts[:2] == ['api', 'projects']:
        project_name = parts[2]
        project = self._get_project_info(project_name)
        if project is None:
          self._send_json({'error': 'Project not found'}, 404)
          return
        self._send_json(project)
        return
      if len(parts) == 4 and parts[:2] == ['api', 'projects'] and parts[3] == 'run-status':
        self._get_run_status(parts[2])
        return
      if len(parts) == 4 and parts[:2] == ['api', 'projects'] and parts[3] == 'live-frame':
        params = parse_qs(parsed_path.query)
        try:
          camera_id = int(params['camera_id'][0])
        except (KeyError, IndexError, ValueError):
          self._send_json({'error': 'camera_id is required'}, 400)
          return
        self._send_live_frame(parts[2], camera_id)
        return
      if path == '/api/shared-folders':
        self._send_json(self._find_shared_folders())
        return
      if len(parts) == 4 and parts[:2] == ['api', 'shared-folders'] and parts[3] == 'videos':
        video_paths = self._shared_video_paths(parts[2])
        if video_paths is None:
          self._send_json({'error': 'Shared folder not found'}, 404)
          return
        self._send_json([path.name for path in video_paths])
        return
      if path == '/api/videos':
        project_name = parse_qs(parsed_path.query).get('project', [''])[0]
        if not project_name:
          self._send_json([])
          return
        if not self._is_existing_project(project_name):
          self._send_json({'error': 'Project not found'}, 404)
          return
        self._send_json(self._find_videos(project_name))
        return
      if path == '/api/frame':
        params = parse_qs(parsed_path.query)
        try:
          project_name = params['project'][0]
          camera_id = int(params['camera_id'][0])
          frame_number = int(params['frame'][0])
        except (KeyError, IndexError, ValueError):
          self._send_json({'error': 'project, camera_id, and frame are required'}, 400)
          return
        if not self._is_existing_project(project_name):
          self._send_json({'error': 'Project not found'}, 404)
          return
        self._send_frame(project_name, camera_id, frame_number)
        return
      self.send_error(404, 'Not found')

    def do_POST(self):
      parsed_path = urlparse(self.path)
      parts = [unquote(part) for part in parsed_path.path.strip('/').split('/')]
      if parsed_path.path == '/api/projects':
        self._create_project()
        return
      if len(parts) == 4 and parts[:2] == ['api', 'projects'] and parts[3] == 'run':
        self._start_project_run(parts[2])
        return
      if len(parts) == 4 and parts[:2] == ['api', 'projects'] and parts[3] == 'live-frame':
        self._receive_live_frame(parts[2])
        return
      if len(parts) == 4 and parts[:2] == ['api', 'projects'] and parts[3] == 'assign-folder':
        self._assign_folder(parts[2])
        return
      if len(parts) == 5 and parts[:2] == ['api', 'projects'] and parts[3] == 'slots':
        self._assign_slot(parts[2], parts[4], parsed_path.query)
        return
      self.send_error(404, 'Not found')

    def do_DELETE(self):
      parts = [unquote(part) for part in urlparse(self.path).path.strip('/').split('/')]
      if len(parts) == 3 and parts[:2] == ['api', 'projects']:
        self._delete_project(parts[2])
        return
      self.send_error(404, 'Not found')

    def _read_json_body(self):
      try:
        content_length = int(self.headers.get('Content-Length', '0'))
        if content_length <= 0 or content_length > 65536:
          return None
        value = json.loads(self.rfile.read(content_length))
      except (ValueError, json.JSONDecodeError, OSError):
        return None
      return value if isinstance(value, dict) else None

    def _delete_project(self, project_name):
      if not isinstance(project_name, str) or not PROJECT_NAME_PATTERN.fullmatch(project_name):
        self._send_json({'error': 'Project name contains unsafe characters'}, 400)
        return

      project_dir = PROJECTS_DIR / project_name
      if not _is_within(project_dir, PROJECTS_DIR):
        self._send_json({'error': 'Invalid project path'}, 400)
        return

      with ASSIGNMENT_LOCK:
        with RUN_LOCK:
          if project_dir.is_symlink():
            self._send_json({'error': 'Project path must not be a symbolic link'}, 400)
            return
          if not project_dir.is_dir():
            self._send_json({'error': 'Project not found'}, 404)
            return
          if not _is_within(project_dir, PROJECTS_DIR):
            self._send_json({'error': 'Invalid project path'}, 400)
            return

          run_state = RUN_STATES.get(project_name)
          if run_state and run_state['state'] == 'running' and run_state['process'].poll() is None:
            self._send_json({'error': 'Cannot delete a project while detection is running'}, 409)
            return

          try:
            shutil.rmtree(project_dir)
          except OSError as error:
            self._send_json({'error': f'Could not delete project: {error}'}, 500)
            return

          with REGISTRY_LOCK:
            registry = self._read_registry()
            registry.pop(project_name, None)
            self._write_registry(registry)
          RUN_STATES.pop(project_name, None)

      self._send_json({'message': f'Deleted project {project_name}'})

    def _start_project_run(self, project_name):
      project = self._get_project_info(project_name)
      if project is None:
        self._send_json({'error': 'Project not found'}, 404)
        return
      if not self._project_video_paths(project_name):
        self._send_json({'error': 'Assign at least one video before running detection'}, 409)
        return

      live_frame_url = f'http://127.0.0.1:5000/api/projects/{project_name}/live-frame'
      command = [
        sys.executable, '-u', str(ROOT / 'main.py'),
        '--project', project_name,
        '--live-frame-url', live_frame_url
      ]
      creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
      with RUN_LOCK:
        if not self._is_existing_project(project_name):
          self._send_json({'error': 'Project not found'}, 404)
          return
        if not self._project_video_paths(project_name):
          self._send_json({'error': 'Assign at least one video before running detection'}, 409)
          return
        existing = RUN_STATES.get(project_name)
        if existing and existing['state'] == 'running' and existing['process'].poll() is None:
          self._send_json({'error': 'Detection is already running for this project'}, 409)
          return
        try:
          process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            creationflags=creation_flags
          )
        except OSError as error:
          self._send_json({'error': f'Could not start detection: {error}'}, 500)
          return

        RUN_STATES[project_name] = {
          'state': 'running',
          'logs': [f'Started: main.py --project {project_name}'],
          'process': process,
          'exit_code': None,
          'started_at': time.time(),
          'finished_at': None,
          'live_frames': {}
        }
        threading.Thread(
          target=self._capture_run_output,
          args=(project_name, process),
          name=f'project-run-{project_name}',
          daemon=True
        ).start()

      self._send_json({'project': project_name, 'state': 'running'}, 202)

    @staticmethod
    def _append_run_log(project_name, process, message):
      with RUN_LOCK:
        state = RUN_STATES.get(project_name)
        if state is not None and state['process'] is process:
          state['logs'].append(message)
          state['logs'] = state['logs'][-RUN_LOG_LIMIT:]

    @classmethod
    def _capture_run_output(cls, project_name, process):
      try:
        if process.stdout is not None:
          for output_line in process.stdout:
            for line in output_line.replace('\r', '\n').splitlines():
              cls._append_run_log(project_name, process, line)
      except (OSError, UnicodeError) as error:
        cls._append_run_log(project_name, process, f'Could not read process output: {error}')
      finally:
        if process.stdout is not None:
          process.stdout.close()
        exit_code = process.wait()
        with RUN_LOCK:
          state = RUN_STATES.get(project_name)
          if state is not None and state['process'] is process and state['state'] == 'running':
            state['exit_code'] = exit_code
            state['state'] = 'done' if exit_code == 0 else 'failed'
            state['finished_at'] = time.time()
            state['logs'].append(f'Process exited with code {exit_code}')
            state['logs'] = state['logs'][-RUN_LOG_LIMIT:]

    def _get_run_status(self, project_name):
      if not self._is_existing_project(project_name):
        self._send_json({'error': 'Project not found'}, 404)
        return
      with RUN_LOCK:
        state = RUN_STATES.get(project_name)
        if state is None:
          result = {'project': project_name, 'state': 'idle', 'logs': [], 'exit_code': None}
        else:
          result = {
            'project': project_name,
            'state': state['state'],
            'logs': list(state['logs']),
            'exit_code': state['exit_code'],
            'started_at': state['started_at'],
            'finished_at': state['finished_at']
          }
      self._send_json(result)

    def _receive_live_frame(self, project_name):
      if not self._is_existing_project(project_name):
        self._send_json({'error': 'Project not found'}, 404)
        return
      try:
        camera_id = int(parse_qs(urlparse(self.path).query)['camera_id'][0])
        content_length = int(self.headers.get('Content-Length', '0'))
      except (KeyError, IndexError, ValueError):
        self._send_json({'error': 'camera_id is required'}, 400)
        return
      if camera_id < 1 or content_length < 4 or content_length > 12 * 1024 * 1024:
        self._send_json({'error': 'Invalid live frame'}, 400)
        return
      if self.headers.get_content_type() != 'image/jpeg':
        self._send_json({'error': 'Live frames must be JPEG'}, 415)
        return
      with RUN_LOCK:
        state = RUN_STATES.get(project_name)
        if state is None or state['state'] != 'running' or state['process'].poll() is not None:
          self._send_json({'error': 'Project has no active detection run'}, 409)
          return

      jpeg_bytes = self.rfile.read(content_length)
      if len(jpeg_bytes) != content_length or not jpeg_bytes.startswith(b'\xff\xd8') or not jpeg_bytes.endswith(b'\xff\xd9'):
        self._send_json({'error': 'Invalid JPEG frame'}, 400)
        return
      with RUN_LOCK:
        state = RUN_STATES.get(project_name)
        if state is None or state['state'] != 'running':
          self._send_json({'error': 'Project has no active detection run'}, 409)
          return
        state['live_frames'][camera_id] = jpeg_bytes
      self.send_response(204)
      self.send_header('Cache-Control', 'no-store')
      self.end_headers()

    def _send_live_frame(self, project_name, camera_id):
      if not self._is_existing_project(project_name):
        self._send_json({'error': 'Project not found'}, 404)
        return
      if camera_id < 1:
        self._send_json({'error': 'camera_id must be positive'}, 400)
        return
      with RUN_LOCK:
        state = RUN_STATES.get(project_name)
        jpeg_bytes = state['live_frames'].get(camera_id) if state else None
      if jpeg_bytes is None:
        self.send_response(204)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        return
      self._send_bytes(jpeg_bytes, 'image/jpeg')

    def log_message(self, format, *args):
      if self.command == 'POST' and urlparse(self.path).path.endswith('/live-frame'):
        return
      super().log_message(format, *args)

    def _create_project(self):
      data = self._read_json_body()
      if data is None:
        self._send_json({'error': 'Expected a JSON object'}, 400)
        return
      name = data.get('name')
      camera_count = data.get('camera_count')
      if not isinstance(name, str) or len(name) > 64 or not PROJECT_NAME_PATTERN.fullmatch(name):
        self._send_json({'error': 'Project name contains unsafe characters'}, 400)
        return
      if isinstance(camera_count, bool) or not isinstance(camera_count, int) or not 1 <= camera_count <= 64:
        self._send_json({'error': 'camera_count must be an integer from 1 to 64'}, 400)
        return

      project_dir = PROJECTS_DIR / name
      if not _is_within(project_dir, PROJECTS_DIR):
        self._send_json({'error': 'Invalid project path'}, 400)
        return
      with ASSIGNMENT_LOCK:
        if project_dir.exists():
          self._send_json({'error': 'Project already exists'}, 409)
          return
        try:
          (project_dir / 'videos').mkdir(parents=True)
          (project_dir / 'outputs').mkdir()
          (project_dir / 'reid_database.json').write_text('{}\n', encoding='utf-8')
        except OSError as error:
          self._send_json({'error': f'Could not create project: {error}'}, 500)
          return
        project = {'name': name, 'camera_count': camera_count, 'assignments': []}
        self._save_project_info(name, project)
      self._send_json(project, 201)

    def _assign_folder(self, project_name):
      project = self._get_project_info(project_name)
      if project is None:
        self._send_json({'error': 'Project not found'}, 404)
        return
      data = self._read_json_body()
      if data is None or not isinstance(data.get('folder'), str):
        self._send_json({'error': 'Expected a JSON object with a folder name'}, 400)
        return
      folder_name = data['folder']
      source_paths = self._shared_video_paths(folder_name)
      if source_paths is None:
        self._send_json({'error': 'Shared folder not found'}, 404)
        return
      if len(source_paths) != project['camera_count']:
        self._send_json({
          'error': f"Folder contains {len(source_paths)} videos but project has {project['camera_count']} camera slots"
        }, 409)
        return

      video_dir = PROJECTS_DIR / project_name / 'videos'
      if not self._validate_destination_directory(video_dir, PROJECTS_DIR / project_name):
        self._send_json({'error': 'Invalid project video directory'}, 400)
        return
      targets = [video_dir / f'camera_{slot:06d}{source.suffix.lower()}'
             for slot, source in enumerate(source_paths, start=1)]
      if not all(_is_within(target, video_dir) for target in targets):
        self._send_json({'error': 'Invalid destination path'}, 400)
        return
      if not self._validate_existing_videos(video_dir):
        self._send_json({'error': 'Project contains an unsafe video path'}, 400)
        return

      assignments = []
      with ASSIGNMENT_LOCK:
        try:
          with tempfile.TemporaryDirectory(prefix='.assignment-', dir=PROJECTS_DIR / project_name) as staging_name:
            staging_dir = Path(staging_name)
            for source, target in zip(source_paths, targets):
              shutil.copy2(source, staging_dir / target.name)
            for old_path in self._project_video_paths(project_name):
              old_path.unlink()
            for target in targets:
              (staging_dir / target.name).replace(target)
            assignments = [
              {
                'camera_id': slot,
                'filename': target.name,
                'source_folder': folder_name,
                'source_filename': source.name
              }
              for slot, (source, target) in enumerate(zip(source_paths, targets), start=1)
            ]
          project['assignments'] = assignments
          self._save_project_info(project_name, project)
        except OSError as error:
          self._send_json({'error': f'Could not assign videos: {error}'}, 500)
          return
      self._send_json({'message': f'Assigned {len(assignments)} videos to {project_name}', 'assignments': assignments})

    def _assign_slot(self, project_name, slot_text, query):
      project = self._get_project_info(project_name)
      if project is None:
        self._send_json({'error': 'Project not found'}, 404)
        return
      try:
        slot = int(slot_text)
      except ValueError:
        self._send_json({'error': 'Camera slot must be an integer'}, 400)
        return
      if not 1 <= slot <= project['camera_count']:
        self._send_json({'error': 'Camera slot is outside this project'}, 400)
        return

      if self.headers.get_content_type() == 'application/json':
        data = self._read_json_body()
        if data is None or not isinstance(data.get('folder'), str) or not isinstance(data.get('filename'), str):
          self._send_json({'error': 'Expected folder and filename'}, 400)
          return
        source = self._shared_video_path(data['folder'], data['filename'])
        if source is None:
          self._send_json({'error': 'Shared video not found or unsafe'}, 404)
          return
        self._copy_to_slot(project, slot, source, {
          'source_folder': data['folder'],
          'source_filename': source.name
        })
        return

      filename = parse_qs(query).get('filename', [''])[0]
      if not self._is_safe_video_name(filename):
        self._send_json({'error': 'Upload filename is unsafe or unsupported'}, 400)
        return
      try:
        content_length = int(self.headers.get('Content-Length', '0'))
      except ValueError:
        content_length = 0
      if content_length <= 0 or content_length > 2 * 1024 * 1024 * 1024:
        self._send_json({'error': 'Upload size is invalid or exceeds 2 GiB'}, 400)
        return
      self._upload_to_slot(project, slot, filename, content_length)

    def _copy_to_slot(self, project, slot, source, source_info):
      project_name = project['name']
      project_dir = PROJECTS_DIR / project_name
      video_dir = project_dir / 'videos'
      if not self._validate_destination_directory(video_dir, project_dir):
        self._send_json({'error': 'Invalid project video directory'}, 400)
        return
      if not _is_within(source, SHARED_VIDEOS_DIR):
        self._send_json({'error': 'Invalid shared video path'}, 400)
        return
      try:
        with ASSIGNMENT_LOCK:
          if not self._normalize_project_videos(project):
            self._send_json({'error': 'Could not normalize existing project videos'}, 400)
            return
          destination = video_dir / f'camera_{slot:06d}{source.suffix.lower()}'
          if not _is_within(destination, video_dir) or not self._validate_existing_videos(video_dir):
            self._send_json({'error': 'Invalid destination path'}, 400)
            return
          with tempfile.TemporaryDirectory(prefix='.slot-', dir=project_dir) as staging_name:
            staged = Path(staging_name) / destination.name
            shutil.copy2(source, staged)
            self._remove_slot_files(video_dir, slot)
            staged.replace(destination)
        self._save_slot_assignment(project, slot, destination.name, source_info)
      except OSError as error:
        self._send_json({'error': f'Could not assign video: {error}'}, 500)
        return
      self._send_json({'message': f'Assigned {destination.name} to camera {slot}'})

    def _upload_to_slot(self, project, slot, filename, content_length):
      project_name = project['name']
      project_dir = PROJECTS_DIR / project_name
      video_dir = project_dir / 'videos'
      if not self._validate_destination_directory(video_dir, project_dir):
        self._send_json({'error': 'Invalid project video directory'}, 400)
        return
      destination = video_dir / f'camera_{slot:06d}{Path(filename).suffix.lower()}'
      if not _is_within(destination, video_dir) or not self._validate_existing_videos(video_dir):
        self._send_json({'error': 'Invalid destination path'}, 400)
        return

      try:
        with ASSIGNMENT_LOCK:
          if not self._normalize_project_videos(project):
            self._send_json({'error': 'Could not normalize existing project videos'}, 400)
            return
          with tempfile.TemporaryDirectory(prefix='.upload-', dir=project_dir) as staging_name:
            staged = Path(staging_name) / destination.name
            remaining = content_length
            with staged.open('wb') as output_file:
              while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                  raise OSError('Upload ended before Content-Length bytes were received')
                output_file.write(chunk)
                remaining -= len(chunk)
            self._remove_slot_files(video_dir, slot)
            staged.replace(destination)
        self._save_slot_assignment(project, slot, destination.name, {
          'source_folder': 'upload',
          'source_filename': filename
        })
      except OSError as error:
        self._send_json({'error': f'Could not save uploaded video: {error}'}, 500)
        return
      self._send_json({'message': f'Uploaded {filename} to camera {slot}'})

    @staticmethod
    def _is_safe_video_name(filename):
      return _is_safe_leaf(filename) and Path(filename).suffix.lower() in SUPPORTED_VIDEO_SUFFIXES

    @staticmethod
    def _validate_destination_directory(directory, project_dir):
      return directory.is_dir() and _is_within(directory, project_dir)

    @classmethod
    def _validate_existing_videos(cls, directory):
      for path in directory.iterdir():
        if path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES and not _is_within(path, directory):
          return False
      return True

    @classmethod
    def _project_video_paths(cls, project_name):
      directory = PROJECTS_DIR / project_name / 'videos'
      if not directory.is_dir() or not _is_within(directory, PROJECTS_DIR / project_name):
        return []
      paths = [path for path in directory.iterdir() if _supported_video(path) and _is_within(path, directory)]
      return sorted(paths, key=lambda path: _natural_sort_key(path.name))

    @classmethod
    def _normalize_project_videos(cls, project):
      project_name = project['name']
      project_dir = PROJECTS_DIR / project_name
      video_dir = project_dir / 'videos'
      paths = cls._project_video_paths(project_name)
      if len(paths) > project['camera_count'] or not cls._validate_existing_videos(video_dir):
        return False
      matches = [ASSIGNED_VIDEO_PATTERN.fullmatch(path.name) for path in paths]
      slots = [int(match.group(1)) for match in matches if match]
      if len(slots) == len(paths) and len(set(slots)) == len(slots):
        return True
      targets = [video_dir / f'camera_{slot:06d}{path.suffix.lower()}'
             for slot, path in enumerate(paths, start=1)]
      if not all(_is_within(target, video_dir) for target in targets):
        return False
      try:
        with tempfile.TemporaryDirectory(prefix='.normalize-', dir=project_dir) as staging_name:
          staging_dir = Path(staging_name)
          for source, target in zip(paths, targets):
            shutil.copy2(source, staging_dir / target.name)
          for source in paths:
            source.unlink()
          for target in targets:
            (staging_dir / target.name).replace(target)
      except OSError:
        return False
      project['assignments'] = [
        {'camera_id': slot, 'filename': target.name, 'source_folder': 'project', 'source_filename': source.name}
        for slot, (source, target) in enumerate(zip(paths, targets), start=1)
      ]
      cls._save_project_info(project_name, project)
      return True

    @classmethod
    def _remove_slot_files(cls, video_dir, slot):
      prefix = f'camera_{slot:06d}.'
      for path in video_dir.iterdir():
        if path.name.lower().startswith(prefix.lower()) and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES:
          if not _is_within(path, video_dir):
            raise OSError('Unsafe existing slot path')
          path.unlink()

    @classmethod
    def _save_slot_assignment(cls, project, slot, filename, source_info):
      assignments = [item for item in project['assignments'] if item.get('camera_id') != slot]
      assignments.append({'camera_id': slot, 'filename': filename, **source_info})
      project['assignments'] = sorted(assignments, key=lambda item: item['camera_id'])
      cls._save_project_info(project['name'], project)

    @staticmethod
    def _read_registry():
      try:
        loaded = json.loads(PROJECTS_REGISTRY.read_text(encoding='utf-8'))
      except (OSError, json.JSONDecodeError):
        loaded = []
      registry = {}
      if isinstance(loaded, list):
        for item in loaded:
          if isinstance(item, str) and PROJECT_NAME_PATTERN.fullmatch(item):
            registry[item] = {}
          elif isinstance(item, dict):
            name = item.get('name')
            if isinstance(name, str) and PROJECT_NAME_PATTERN.fullmatch(name):
              registry[name] = {
                'camera_count': item.get('camera_count'),
                'assignments': item.get('assignments')
              }
      elif isinstance(loaded, dict):
        for name, item in loaded.items():
          if isinstance(name, str) and PROJECT_NAME_PATTERN.fullmatch(name) and isinstance(item, dict):
            registry[name] = item
      return registry

    @staticmethod
    def _write_registry(registry):
      records = [
        {'name': name, **registry[name]}
        for name in sorted(registry)
      ]
      temp_path = PROJECTS_REGISTRY.with_suffix('.tmp')
      temp_path.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
      temp_path.replace(PROJECTS_REGISTRY)

    @classmethod
    def _default_project_info(cls, project_name):
      video_paths = cls._project_video_paths(project_name)
      output_dir = PROJECTS_DIR / project_name / 'outputs'
      output_count = sum(
        1 for path in output_dir.glob('output_camera_*.mp4')
        if VIDEO_PATTERN.fullmatch(path.name) and path.is_file()
      ) if output_dir.is_dir() else 0
      assignments = [
        {'camera_id': slot, 'filename': path.name, 'source_folder': 'project', 'source_filename': path.name}
        for slot, path in enumerate(video_paths, start=1)
      ]
      return {
        'name': project_name,
        'camera_count': max(len(video_paths), output_count, 1),
        'assignments': assignments
      }

    @classmethod
    def _discover_projects(cls):
      projects = sorted(
        path.name for path in PROJECTS_DIR.iterdir()
        if path.is_dir() and PROJECT_NAME_PATTERN.fullmatch(path.name) and _is_within(path, PROJECTS_DIR)
      ) if PROJECTS_DIR.is_dir() else []
      with REGISTRY_LOCK:
        old_registry = cls._read_registry()
        new_registry = {}
        for name in projects:
          stored = old_registry.get(name, {})
          defaults = cls._default_project_info(name)
          camera_count = stored.get('camera_count')
          if isinstance(camera_count, bool) or not isinstance(camera_count, int) or camera_count < 1:
            camera_count = defaults['camera_count']
          assignments = stored.get('assignments')
          if not isinstance(assignments, list):
            assignments = defaults['assignments']
          new_registry[name] = {
            'camera_count': camera_count,
            'assignments': assignments
          }
        if new_registry != old_registry:
          cls._write_registry(new_registry)
      return projects

    @classmethod
    def _save_project_info(cls, project_name, project):
      with REGISTRY_LOCK:
        registry = cls._read_registry()
        registry[project_name] = {
          'camera_count': project['camera_count'],
          'assignments': project['assignments']
        }
        cls._write_registry(registry)

    @classmethod
    def _get_project_info(cls, project_name):
      if not cls._is_existing_project(project_name):
        return None
      cls._discover_projects()
      with REGISTRY_LOCK:
        entry = cls._read_registry().get(project_name)
      if entry is None:
        return None
      return {'name': project_name, **entry}

    @staticmethod
    def _is_existing_project(project_name):
      if not isinstance(project_name, str) or not PROJECT_NAME_PATTERN.fullmatch(project_name):
        return False
      project_dir = PROJECTS_DIR / project_name
      return project_dir.is_dir() and _is_within(project_dir, PROJECTS_DIR)

    @classmethod
    def _shared_folder_path(cls, folder_name):
      if not _is_safe_leaf(folder_name):
        return None
      root = SHARED_VIDEOS_DIR
      folder = root / folder_name
      if not folder.is_dir() or folder.parent.resolve() != root.resolve() or not _is_within(folder, root):
        return None
      return folder

    @classmethod
    def _shared_video_paths(cls, folder_name):
      folder = cls._shared_folder_path(folder_name)
      if folder is None:
        return None
      paths = [
        path for path in folder.iterdir()
        if _is_safe_leaf(path.name) and _supported_video(path) and _is_within(path, folder)
      ]
      return sorted(paths, key=lambda path: _natural_sort_key(path.name))

    @classmethod
    def _shared_video_path(cls, folder_name, filename):
      if not cls._is_safe_video_name(filename):
        return None
      folder = cls._shared_folder_path(folder_name)
      if folder is None:
        return None
      path = folder / filename
      return path if _supported_video(path) and _is_within(path, folder) else None

    @classmethod
    def _find_shared_folders(cls):
      if not SHARED_VIDEOS_DIR.is_dir():
        return []
      folders = [
        path for path in SHARED_VIDEOS_DIR.iterdir()
        if path.is_dir() and _is_safe_leaf(path.name) and _is_within(path, SHARED_VIDEOS_DIR)
      ]
      folders.sort(key=lambda path: _natural_sort_key(path.name))
      return [
        {'name': folder.name, 'count': len(cls._shared_video_paths(folder.name) or [])}
        for folder in folders
      ]

    @classmethod
    def _find_videos(cls, project_name):
      videos = []
      output_dir = PROJECTS_DIR / project_name / 'outputs'
      if not output_dir.is_dir() or not _is_within(output_dir, PROJECTS_DIR / project_name):
        return videos
      for path in output_dir.glob('output_camera_*.mp4'):
        match = VIDEO_PATTERN.fullmatch(path.name)
        if match and path.is_file() and _is_within(path, output_dir):
          camera_id = int(match.group(1))
          capture = cv2.VideoCapture(str(path))
          width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
          height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
          frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
          fps = capture.get(cv2.CAP_PROP_FPS) or 30
          capture.release()
          videos.append({
            'camera_id': camera_id,
            'label': f'Camera {camera_id}',
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'fps': fps
          })
      return sorted(videos, key=lambda item: item['camera_id'])

    def _send_frame(self, project_name, camera_id, frame_number):
      video_path = PROJECTS_DIR / project_name / 'outputs' / f'output_camera_{camera_id}.mp4'
      if camera_id < 1 or frame_number < 0 or not video_path.is_file() or not _is_within(video_path, PROJECTS_DIR / project_name / 'outputs'):
        self._send_json({'error': 'Video or frame not found'}, 404)
        return

      capture = cv2.VideoCapture(str(video_path))
      try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
      finally:
        capture.release()
      if not success:
        self._send_json({'error': 'Could not decode requested frame'}, 404)
        return

      success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
      if not success:
        self._send_json({'error': 'Could not encode requested frame'}, 500)
        return
      self._send_bytes(encoded.tobytes(), 'image/jpeg')

    def _send_html(self, content):
        body = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, content, status=200):
        body = json.dumps(content).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)


def main():
    if not PROJECTS_REGISTRY.exists():
        PROJECTS_REGISTRY.write_text('[]\n', encoding='utf-8')
    server = ThreadingHTTPServer(('127.0.0.1', 5000), ViewerHandler)
    print('Viewer running at http://localhost:5000')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nViewer stopped.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()