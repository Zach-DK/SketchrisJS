// Expected model details from notebook:
// - Input: [1, 1, 128, 128] grayscale
// - Normalization during training: ToTensor then Normalize(mean=[0.5], std=[0.5])
//   which maps pixel [0..1] to (x - 0.5) / 0.5 => [-1, 1]
// - Classes order taken from LabelEncoder(classes_):
//   ['I_piece','J_piece','L_piece','O_piece','S_piece','T_piece','Z_piece']

(async () => {
    const CANVAS_SIZE = 128;
    const CLASS_NAMES = ['I Piece','J Piece','L Piece','O Piece','S Piece','T Piece','Z Piece'];

    // DOM references
    const canvas = document.getElementById('pad');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    // Fixed brush size = 2
    const FIXED_BRUSH_SIZE = 2;
    const invert = document.getElementById('invert');
    const probsTableBody = document.querySelector('#probsTable tbody');

    // Init canvas: black background
    function resetCanvas() {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    }
    resetCanvas();

    // Simple drawing
    let drawing = false;
    let lastX = 0, lastY = 0;
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: Math.max(0, Math.min(CANVAS_SIZE, (clientX - rect.left) * (canvas.width / rect.width))),
            y: Math.max(0, Math.min(CANVAS_SIZE, (clientY - rect.top) * (canvas.height / rect.height)))
        };
    }
    function start(e) { drawing = true; const p = getPos(e); lastX = p.x; lastY = p.y; draw(e); }
    function end() { drawing = false; }
    function draw(e) {
        if (!drawing) return;
        const p = getPos(e);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = FIXED_BRUSH_SIZE;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
        lastX = p.x; lastY = p.y;
        e.preventDefault();

        // Mark for live prediction; throttled by rAF loop
        needsPredict = true;
    }

    canvas.addEventListener('mousedown', start);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', end);
    canvas.addEventListener('touchstart', start, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', end);

    clearBtn.addEventListener('click', () => { resetCanvas(); needsPredict = true; });
    invert.addEventListener('change', () => { needsPredict = true; });

    // Create table rows
    function initTable() {
        probsTableBody.innerHTML = '';
        for (const name of CLASS_NAMES) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${name}</td>
                <td><div class="prob"><span style="width:0%"></span></div></td>
                <td class="score">0.000</td>
            `;
            probsTableBody.appendChild(tr);
        }
    }
    initTable();

    // Load ONNX model
    // Prefer WebGL for faster inference; fallback to WASM with SIMD and threads
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = Math.min(4, (navigator.hardwareConcurrency || 2));
    let session;
    try {
        session = await ort.InferenceSession.create('tetris_classifier.onnx', { executionProviders: ['webgl'] });
    } catch (err) {
        session = await ort.InferenceSession.create('tetris_classifier.onnx', { executionProviders: ['wasm'] });
    }

    // Preprocess canvas -> Float32Array [1,1,128,128], normalized to [-1,1]
    function preprocess() {
        const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        const { data } = imageData; // RGBA
        const out = new Float32Array(1 * 1 * CANVAS_SIZE * CANVAS_SIZE);
        let o = 0;
        for (let i = 0; i < data.length; i += 4) {
            // grayscale from RGB; canvas is black background with white drawing
            const r = data[i], g = data[i + 1], b = data[i + 2];
            let gray = 0.299 * r + 0.587 * g + 0.114 * b; // [0..255]
            if (invert.checked) gray = 255 - gray;
            const norm01 = gray / 255;         // [0..1]
            const norm = (norm01 - 0.5) / 0.5; // [-1..1]
            out[o++] = norm;
        }
        return out;
    }

    function softmax(logits) {
        const maxLogit = Math.max(...logits);
        const exps = logits.map(v => Math.exp(v - maxLogit));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / sum);
    }

    function updateTable(probs) {
        // Find argmax
        let maxIdx = 0; for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;
        // Update rows
        const rows = probsTableBody.querySelectorAll('tr');
        rows.forEach((tr, idx) => {
            tr.classList.toggle('row-max', idx === maxIdx);
            const bar = tr.querySelector('.prob > span');
            const scoreCell = tr.querySelector('.score');
            const pct = (probs[idx] * 100);
            bar.style.width = `${pct.toFixed(1)}%`;
            scoreCell.textContent = probs[idx].toFixed(3);
        });
    }

    let predictQueued = false; // unused after rAF, but keep to minimize diff
    let predicting = false;
    let needsPredict = true;
    let lastInferTs = 0;
    const MIN_INFER_MS = 80; // ~12.5 FPS inference to keep UI smooth
    async function predict() {
        predicting = true;
        const inputData = preprocess();
        const tensor = new ort.Tensor('float32', inputData, [1, 1, CANVAS_SIZE, CANVAS_SIZE]);
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        const logits = Array.from(results.output.data); // [1, 7]
        const probs = softmax(logits);
        updateTable(probs);
        predicting = false;
    }
    function schedulePredict() { needsPredict = true; }
    function tick(ts) {
        if (!predicting && needsPredict && (ts - lastInferTs >= MIN_INFER_MS)) {
            lastInferTs = ts;
            needsPredict = false;
            predict();
        }
        requestAnimationFrame(tick);
    }

    predictBtn.addEventListener('click', () => { needsPredict = true; });
    // Start rAF loop and run initial inference
    requestAnimationFrame(tick);
})();


