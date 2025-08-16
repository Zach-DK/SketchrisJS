// Sketchris: Tetris + Sketch-based piece prediction
// Uses onnxruntime-web to predict piece class from a 128x128 canvas on pen/mouse release

(async () => {
    // ===== Colors and UI =====
    const COLORS = {
        bg: '#181359',
        panel: '#01377d',
        text: '#ffffff',
        muted: '#bbbbbb',
        accent: '#6e61a4',
    };

    // ===== Predict UI =====
    const PAD_SIZE = 128;
    const CLASS_NAMES = ['I Piece','J Piece','L Piece','O Piece','S Piece','T Piece','Z Piece'];
    const PIECE_CODES = ['I','J','L','O','S','T','Z']; // aligned with CLASS_NAMES order
    const pad = document.getElementById('pad');
    const padCtx = pad.getContext('2d', { willReadFrequently: true });
    const padOverlay = document.getElementById('padOverlay');
    // removed invert/clear controls per request
    const probsTableBody = document.querySelector('#probsTable tbody');
    // mini predictions under controls
    const sfxLock = document.getElementById('sfxLock');
    const sfxLine = document.getElementById('sfxLine');
    const sfxSummon = document.getElementById('sfxSummon');
    // Low-latency audio manager using Web Audio API
    const AudioMgr = (() => {
        let ctx = null;
        const buffers = {};
        let initialized = false;
        async function ensureCtx(){
            if (ctx) return ctx;
            ctx = new (window.AudioContext || window.webkitAudioContext)();
            return ctx;
        }
        async function loadBuffer(key, url){
            const c = await ensureCtx();
            const res = await fetch(url);
            const arr = await res.arrayBuffer();
            buffers[key] = await c.decodeAudioData(arr);
        }
        async function init(){
            if (initialized) return;
            initialized = true;
            await ensureCtx();
            await Promise.all([
                loadBuffer('summon', 'static/assets/audio/Jug Unplug Button 1.wav'),
                loadBuffer('lock', 'static/assets/audio/Rock Impact 2.wav'),
                loadBuffer('line', 'static/assets/audio/Rubble Burst 3.wav'),
            ]);
        }
        function play(key, volume = 1){
            if (!ctx || !buffers[key]) return;
            if (ctx.state === 'suspended') ctx.resume();
            const src = ctx.createBufferSource();
            src.buffer = buffers[key];
            const gain = ctx.createGain();
            gain.gain.value = volume;
            src.connect(gain).connect(ctx.destination);
            src.start();
        }
        return { init, play };
    })();
    // Prime audio on first user interaction to satisfy autoplay policies and pre-decode buffers
    let audioPrimed = false;
    function primeAudio(){
        if(audioPrimed) return;
        [sfxSummon, sfxLock, sfxLine].forEach(a=>{ try { a.muted = true; const p=a.play(); if(p && p.catch) p.catch(()=>{}); a.pause(); a.currentTime=0; a.muted=false; } catch(e){} });
        AudioMgr.init().catch(()=>{});
        audioPrimed = true;
        document.removeEventListener('pointerdown', primeAudio);
        document.removeEventListener('keydown', primeAudio);
    }
    document.addEventListener('pointerdown', primeAudio);
    document.addEventListener('keydown', primeAudio);
    let gameReady = false;
    // Initialize pad
    function resetPad(){ padCtx.fillStyle = '#000'; padCtx.fillRect(0,0,PAD_SIZE,PAD_SIZE); }
    resetPad();
    // Init table
    function initTable(){
        probsTableBody.innerHTML='';
        for(const name of CLASS_NAMES){
            const tr=document.createElement('tr');
            tr.innerHTML=`<td>${name}</td><td><div class="prob"><span style="width:0%"></span></div></td><td class="score">0.000</td>`;
            probsTableBody.appendChild(tr);
        }
    }
    initTable();
    // no external toggle; predictions always visible in mini panel

    // ===== ONNX session =====
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = Math.min(4, (navigator.hardwareConcurrency || 2));
    let session;
    try { session = await ort.InferenceSession.create('tetris_classifier_fp16.onnx', { executionProviders: ['webgl'] }); }
    catch { session = await ort.InferenceSession.create('tetris_classifier_fp16.onnx', { executionProviders: ['wasm'] }); }

    function preprocessPad(){
        const img = padCtx.getImageData(0,0,PAD_SIZE,PAD_SIZE);
        const d = img.data;
        const out = new Float32Array(1*1*PAD_SIZE*PAD_SIZE);
        let o=0;
        for(let i=0;i<d.length;i+=4){
            const r=d[i], g=d[i+1], b=d[i+2];
            let gray = 0.299*r + 0.587*g + 0.114*b;
            // invert removed
            const norm = (gray/255 - 0.5)/0.5;
            out[o++] = norm;
        }
        return out;
    }
    function softmax(logits){
        const m = Math.max(...logits);
        const exps = logits.map(v=>Math.exp(v-m));
        const s = exps.reduce((a,b)=>a+b,0);
        return exps.map(v=>v/s);
    }
    function updateTable(probs){
        let maxIdx=0; for(let i=1;i<probs.length;i++){ if(probs[i]>probs[maxIdx]) maxIdx=i; }
        const rows = probsTableBody.querySelectorAll('tr');
        rows.forEach((tr,idx)=>{
            tr.classList.toggle('row-max', idx===maxIdx);
            const bar = tr.querySelector('.prob>span');
            const cell = tr.querySelector('.score');
            const pct = probs[idx]*100;
            bar.style.width = `${pct.toFixed(1)}%`;
            cell.textContent = probs[idx].toFixed(3);
        });
    }

    async function predictPad(){
        const data = preprocessPad();
        const tensor = new ort.Tensor('float32', data, [1,1,PAD_SIZE,PAD_SIZE]);
        const res = await session.run({ input: tensor });
        const logits = Array.from(res.output.data);
        const probs = softmax(logits);
        updateTable(probs);
        let best = 0; for(let i=1;i<probs.length;i++) if(probs[i]>probs[best]) best=i;
        // Return the piece code directly to avoid string mapping issues
        return PIECE_CODES[best];
    }

    // Drawing on pad
    const BRUSH = 2;
    let drawing=false; let lastX=0, lastY=0;
    function getPos(e){ const r=pad.getBoundingClientRect(); const cx=e.touches?e.touches[0].clientX:e.clientX; const cy=e.touches?e.touches[0].clientY:e.clientY; return {x: (cx-r.left)*(pad.width/r.width), y: (cy-r.top)*(pad.height/r.height)};}
    function isPieceActive(){ return !!player.matrix; }
    function start(e){ if(isPieceActive()) return; drawing=true; const p=getPos(e); lastX=p.x; lastY=p.y; draw(e); }
    function end(){ if(!drawing) return; drawing=false; onPadReleased(); }
    function draw(e){ if(!drawing) return; const p=getPos(e); padCtx.strokeStyle='#fff'; padCtx.lineWidth=BRUSH; padCtx.lineCap='round'; padCtx.lineJoin='round'; padCtx.beginPath(); padCtx.moveTo(lastX,lastY); padCtx.lineTo(p.x,p.y); padCtx.stroke(); lastX=p.x; lastY=p.y; e.preventDefault(); }
    pad.addEventListener('mousedown', start); pad.addEventListener('mousemove', draw); window.addEventListener('mouseup', end);
    pad.addEventListener('touchstart', start, {passive:false}); pad.addEventListener('touchmove', draw, {passive:false}); pad.addEventListener('touchend', end);
    // controls removed

    function showPadOverlay(show){ padOverlay.classList.toggle('hidden', !show); }
    async function onPadReleased(){
        if(isPieceActive() || !session || !gameReady) return;
        // Play summon sound immediately on release (within user gesture)
        try { AudioMgr.play('summon'); } catch(e) { try { sfxSummon.currentTime = 0; sfxSummon.play(); } catch(_){} }
        showPadOverlay(true);
        const pieceCode = await predictPad();
        resetPad();
        if(pieceCode){
            // Queue and spawn via unified path
            queue.unshift(pieceCode);
            spawnNext();
            renderBoard();
        } else {
            showPadOverlay(false);
        }
        // overlay remains shown while piece is active
    }

    // ===== Tetris core =====
    const boardCanvas = document.getElementById('board');
    const boardCtx = boardCanvas.getContext('2d');
    // removed next/hold UI

    const COLS = 10, ROWS = 20, TILE = 24; // 240x480 canvas
    const DROP_START_SPEED_MS = 800;
    const SPEED_PER_LEVEL = 60;

    // Piece shapes (SRS orientations simplified)
    const SHAPES = {
        I: [
            [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
            [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
            [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],
            [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]],
        ],
        O: [
            [[1,1],[1,1]],
            [[1,1],[1,1]],
            [[1,1],[1,1]],
            [[1,1],[1,1]],
        ],
        T: [
            [[0,1,0],[1,1,1],[0,0,0]],
            [[0,1,0],[0,1,1],[0,1,0]],
            [[0,0,0],[1,1,1],[0,1,0]],
            [[0,1,0],[1,1,0],[0,1,0]],
        ],
        S: [
            [[0,1,1],[1,1,0],[0,0,0]],
            [[0,1,0],[0,1,1],[0,0,1]],
            [[0,0,0],[0,1,1],[1,1,0]],
            [[1,0,0],[1,1,0],[0,1,0]],
        ],
        Z: [
            [[1,1,0],[0,1,1],[0,0,0]],
            [[0,0,1],[0,1,1],[0,1,0]],
            [[0,0,0],[1,1,0],[0,1,1]],
            [[0,1,0],[1,1,0],[1,0,0]],
        ],
        J: [
            [[1,0,0],[1,1,1],[0,0,0]],
            [[0,1,1],[0,1,0],[0,1,0]],
            [[0,0,0],[1,1,1],[0,0,1]],
            [[0,1,0],[0,1,0],[1,1,0]],
        ],
        L: [
            [[0,0,1],[1,1,1],[0,0,0]],
            [[0,1,0],[0,1,0],[0,1,1]],
            [[0,0,0],[1,1,1],[1,0,0]],
            [[1,1,0],[0,1,0],[0,1,0]],
        ],
    };

    // Colors per piece
    const PIECE_COLOR = {
        I: '#6e61a4', O: '#ffffff', T: '#bbbbbb', S: '#6e61a4', Z: '#01377d', J: '#6e61a4', L: '#bbbbbb'
    };

    function createMatrix(){ return Array.from({length: ROWS}, ()=>Array(COLS).fill(null)); }
    const arena = createMatrix();

    function collide(arena, piece){
        const {matrix, pos} = piece;
        for(let y=0;y<matrix.length;y++){
            for(let x=0;x<matrix[y].length;x++){
                if(matrix[y][x]){
                    const ay = y + pos.y;
                    const ax = x + pos.x;
                    if(ay<0 || ay>=ROWS || ax<0 || ax>=COLS || arena[ay][ax]) return true;
                }
            }
        }
        return false;
    }
    function merge(arena, piece){
        piece.matrix.forEach((row,y)=>{
            row.forEach((v,x)=>{ if(v){ arena[y+piece.pos.y][x+piece.pos.x] = piece.type; } });
        });
    }
    function rotateMatrix(m){
        const N = m.length; const ret = Array.from({length:N},()=>Array(N).fill(0));
        for(let y=0;y<N;y++) for(let x=0;x<N;x++) ret[x][N-1-y]=m[y][x];
        return ret;
    }

    function clearLines(){
        let lines=0;
        outer: for(let y=ROWS-1;y>=0;y--){
            for(let x=0;x<COLS;x++){ if(!arena[y][x]) continue outer; }
            const row = arena.splice(y,1)[0].fill(null);
            arena.unshift(row);
            lines++;
            y++;
        }
        if(lines>0){
            // play line clear sfx
            try { AudioMgr.play('line'); } catch(e) { try { sfxLine.currentTime = 0; sfxLine.play(); } catch(_){} }
            onLinesCleared(lines);
        }
    }

    // Player state
    const player = {
        pos:{x:3,y:0},
        matrix: null,
        rot: 0,
        type: null,
    };
    let score=0, level=1, cleared=0;
    const scoreEl=document.getElementById('score');
    const levelEl=document.getElementById('level');
    const linesEl=document.getElementById('lines');
    function onLinesCleared(n){
        const points=[0,40,100,300,1200][n]||0;
        score+=points*level; cleared+=n; if(cleared>=level*10) level++;
        scoreEl.textContent=score; levelEl.textContent=level; linesEl.textContent=cleared;
    }

    // Queue of pieces: we integrate predictions by appending to queue end
    const queue = [];
    function enqueuePredictedPiece(code){ queue.push(code); if(!player.matrix) spawnNext(); renderSide(); }
    function getNextCode(){
        // Never spawn randomly; require predicted pieces only
        return queue.length>0 ? queue.shift() : null;
    }
    function createPiece(code){
        const shape = SHAPES[code][0];
        return { type: code, matrix: shape.map(r=>r.slice()), rot:0, pos:{x:3,y:0} };
    }
    function summonPieceNow(code){
        const p = createPiece(code);
        player.type = code;
        player.matrix = p.matrix;
        player.rot = 0;
        player.pos = { x: 3, y: 0 };
        // Show overlay while piece is active
        showPadOverlay(true);
        // If immediate collision at spawn: treat as game over/reset
        if(collide(arena, player)){
            arena.forEach(r=>r.fill(null));
            score=0; level=1; cleared=0; scoreEl.textContent=0; levelEl.textContent=1; linesEl.textContent=0;
        }
        // Reset gravity counter so it doesn't drop instantly
        dropCounter = 0;
        renderSide();
    }
    function spawnNext(){
        const code = getNextCode();
        if(!code){
            // No piece available: show overlay and wait for user to draw
            player.matrix = null;
            showPadOverlay(false); // allow drawing
            return;
        }
        const p = createPiece(code);
        player.type=code; player.matrix=p.matrix; player.rot=0; player.pos={x:3,y:0};
        if(collide(arena, player)){ // game over
            arena.forEach(r=>r.fill(null));
            score=0; level=1; cleared=0; scoreEl.textContent=0; levelEl.textContent=1; linesEl.textContent=0;
        }
        // New piece is active; block drawing
        showPadOverlay(true);
    }

    // Controls
    let dropCounter=0; let lastTime=0; let paused=false;
    function dropInterval(){ return Math.max(100, DROP_START_SPEED_MS - (level-1)*SPEED_PER_LEVEL); }
    function playerDrop(){
        if(!player.matrix) return;
        player.pos.y++;
        if(collide(arena,player)){
            player.pos.y--;
            merge(arena,player);
            // play lock sfx
            try { AudioMgr.play('lock'); } catch(e) { try { sfxLock.currentTime = 0; sfxLock.play(); } catch(_){} }
            clearLines();
            // Piece locked; no active piece now
            player.matrix = null;
            showPadOverlay(false);
            spawnNext();
        }
        dropCounter=0;
        renderBoard();
    }
    function playerMove(dir){ player.pos.x+=dir; if(collide(arena,player)) player.pos.x-=dir; renderBoard(); }
    function playerRotate(){
        if(!player.matrix) return;
        // rotate with basic wall kicks
        const rotated = rotateMatrix(player.matrix);
        const oldX = player.pos.x;
        const kicks=[0,1,-1,2,-2];
        for(const k of kicks){
            player.pos.x = oldX + k;
            const old = player.matrix; player.matrix = rotated;
            if(!collide(arena,player)){ return; }
            player.matrix = old;
        }
        player.pos.x = oldX;
        renderBoard();
    }
    function holdPiece(){ /* basic single-hold could be added later if desired */ }

    document.addEventListener('keydown', (e)=>{
        if(e.key==='ArrowLeft') playerMove(-1);
        else if(e.key==='ArrowRight') playerMove(1);
        else if(e.key==='ArrowDown') playerDrop();
        else if(e.key===' '){ if(!player.matrix) return; while(true){ player.pos.y++; if(collide(arena,player)){ player.pos.y--; merge(arena,player); clearLines(); spawnNext(); break; } } renderBoard(); }
        else if(e.key==='ArrowUp' || e.key==='x' || e.key==='X') playerRotate();
        else if(e.key==='z' || e.key==='Z') { playerRotate(); playerRotate(); playerRotate(); }
        else if(e.key==='c' || e.key==='C') holdPiece();
        else if(e.key==='p' || e.key==='P') paused=!paused;
    });

    // Touch/click controls around the canvas
    const btnLeft = document.getElementById('btnLeft');
    const btnRight = document.getElementById('btnRight');
    const btnRotate = document.getElementById('btnRotate');
    const btnSoft = document.getElementById('btnSoft');
    const btnHard = document.getElementById('btnHard');
    if (btnLeft) btnLeft.addEventListener('click', ()=> playerMove(-1));
    if (btnRight) btnRight.addEventListener('click', ()=> playerMove(1));
    if (btnRotate) btnRotate.addEventListener('click', ()=> playerRotate());
    if (btnSoft) btnSoft.addEventListener('click', ()=> playerDrop());
    if (btnHard) btnHard.addEventListener('click', ()=>{
        if(!player.matrix) return;
        while(true){ player.pos.y++; if(collide(arena,player)){ player.pos.y--; merge(arena,player); clearLines(); spawnNext(); break; } }
        renderBoard();
    });

    // Rendering
    function drawCell(ctx,x,y,color){
        ctx.fillStyle=color; ctx.fillRect(x*TILE,y*TILE,TILE,TILE);
        ctx.strokeStyle='rgba(255,255,255,.1)'; ctx.strokeRect(x*TILE+0.5,y*TILE+0.5,TILE-1,TILE-1);
    }
    function renderBoard(){
        boardCtx.fillStyle = '#0b0b26'; boardCtx.fillRect(0,0,boardCanvas.width, boardCanvas.height);
        // Draw subtle grid lines to help alignment
        boardCtx.strokeStyle = 'rgba(255,255,255,0.06)';
        boardCtx.lineWidth = 1;
        for(let x=0; x<=COLS; x++){ const px = x*TILE + 0.5; boardCtx.beginPath(); boardCtx.moveTo(px, 0); boardCtx.lineTo(px, ROWS*TILE); boardCtx.stroke(); }
        for(let y=0; y<=ROWS; y++){ const py = y*TILE + 0.5; boardCtx.beginPath(); boardCtx.moveTo(0, py); boardCtx.lineTo(COLS*TILE, py); boardCtx.stroke(); }
        for(let y=0;y<ROWS;y++) for(let x=0;x<COLS;x++){
            const t = arena[y][x]; if(t) drawCell(boardCtx,x,y,PIECE_COLOR[t]);
        }
        if(player.matrix){
            player.matrix.forEach((row,yy)=>row.forEach((v,xx)=>{ if(v) drawCell(boardCtx, xx+player.pos.x, yy+player.pos.y, PIECE_COLOR[player.type]); }));
        }
    }
    // removed side rendering

    function update(time=0){
        const delta = time - lastTime; lastTime = time;
        if(!paused && player.matrix){
            dropCounter += delta;
            if(dropCounter > dropInterval()) playerDrop();
        }
        renderBoard();
        requestAnimationFrame(update);
    }

    // Boot
    spawnNext(); update();
    gameReady = true;
})();


