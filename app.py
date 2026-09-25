import json
import re
import cv2
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
VIDEO_PATTERN = re.compile(r'^output_camera_(\d+)\.mp4$', re.IGNORECASE)


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
    main { padding: 0 2rem 2rem; }
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
      .status { display: block; margin-top: 0.4rem; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Processed Camera Outputs</h1>
    <span class="status" id="status">Loading videos...</span>
  </header>
  <main><section class="grid" id="grid"></section></main>

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
    const viewer = document.getElementById('viewer');
    const expandedCanvas = document.getElementById('expanded-canvas');
    const expandedControls = document.getElementById('expanded-controls');
    const viewerLabel = document.getElementById('viewer-label');
    const states = new Map();

    function closeViewer() {
      const state = states.get(Number(viewer.dataset.cameraId));
      if (state) state.canvases = state.canvases.filter((canvas) => canvas !== expandedCanvas);
      expandedControls.replaceChildren();
      viewer.hidden = true;
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
      const response = await fetch(`/api/frame?camera_id=${state.cameraId}&frame=${state.frame}`, { cache: 'no-store' });
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
      try {
        const response = await fetch('/api/videos');
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

    document.getElementById('close').addEventListener('click', closeViewer);
    viewer.addEventListener('click', (event) => {
      if (event.target === viewer) closeViewer();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !viewer.hidden) closeViewer();
    });
    loadVideos();
  </script>
</body>
</html>"""


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        if path == '/':
            self._send_html(PAGE)
            return
        if path == '/api/videos':
            self._send_json(self._find_videos())
            return
        if path == '/api/frame':
            params = parse_qs(parsed_path.query)
            try:
                camera_id = int(params['camera_id'][0])
                frame_number = int(params['frame'][0])
            except (KeyError, IndexError, ValueError):
                self.send_error(400, 'camera_id and frame must be integers')
                return
            self._send_frame(camera_id, frame_number)
            return
        self.send_error(404, 'Not found')

    def _send_frame(self, camera_id, frame_number):
        video_path = ROOT / f'output_camera_{camera_id}.mp4'
        if frame_number < 0 or not video_path.is_file():
            self.send_error(404, 'Video or frame not found')
            return

        capture = cv2.VideoCapture(str(video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
        finally:
            capture.release()
        if not success:
            self.send_error(404, 'Could not decode requested frame')
            return

        success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not success:
            self.send_error(500, 'Could not encode requested frame')
            return
        self._send_bytes(encoded.tobytes(), 'image/jpeg')

    @staticmethod
    def _find_videos():
        videos = []
        for path in ROOT.glob('output_camera_*.mp4'):
            match = VIDEO_PATTERN.fullmatch(path.name)
            if match and path.is_file():
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

    def _send_html(self, content):
        body = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, content):
        body = json.dumps(content).encode('utf-8')
        self.send_response(200)
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