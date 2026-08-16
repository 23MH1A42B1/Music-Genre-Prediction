/* ==========================================================================
   SONICGENRE AI - FRONTEND APPLICATION ENGINE
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    
    const dropZone = document.getElementById("dropZone");
    const browseBtn = document.getElementById("browseBtn");
    const audioFileInput = document.getElementById("audioFileInput");
    const presetTracksGrid = document.getElementById("presetTracksGrid");
    const modelSelect = document.getElementById("modelSelect");
    const predictBtn = document.getElementById("predictBtn");
    
    const playerPanel = document.getElementById("playerPanel");
    const mainAudioPlayer = document.getElementById("mainAudioPlayer");
    const currentTrackTitle = document.getElementById("currentTrackTitle");
    const currentTrackGenre = document.getElementById("currentTrackGenre");
    const waveformCanvas = document.getElementById("waveformCanvas");
    
    const emptyState = document.getElementById("emptyState");
    const predictionView = document.getElementById("predictionView");
    const heroGenreTitle = document.getElementById("heroGenreTitle");
    const heroConfidenceText = document.getElementById("heroConfidenceText");
    const genreBadgeIcon = document.getElementById("genreBadgeIcon");
    const probList = document.getElementById("probList");
    
    const recPanel = document.getElementById("recPanel");
    const recGrid = document.getElementById("recGrid");

    // Friendly Preset Metadata
    const presetMetadata = {
        "Blues": { icon: "fa-guitar", title: "Classic Blues Riff", desc: "12-Bar Shuffle progression" },
        "Classical": { icon: "fa-music", title: "Symphonic Piano Sonata", desc: "Arpeggiated C-Major triad" },
        "Country": { icon: "fa-guitar", title: "Nashville Acoustic Twang", desc: "Fingerpicked acoustic guitar" },
        "Disco": { icon: "fa-record-vinyl", title: "120 BPM Synth Disco", desc: "4-on-the-floor kick & bass" },
        "Hiphop": { icon: "fa-headphones", title: "808 Sub-Bass Beat", desc: "90 BPM sub-bass & snare" },
        "Jazz": { icon: "fa-saxophone", title: "Midnight Swing Quartet", desc: "Major 7th swing chord swell" },
        "Metal": { icon: "fa-drum", title: "High-Octane Distortion Riff", desc: "Fast double kick & drive" },
        "Pop": { icon: "fa-microphone-lines", title: "Modern Dance Pop Anthem", desc: "Upbeat synth chord sequence" },
        "Reggae": { icon: "fa-sun", title: "Dub Rhythm Skank", desc: "75 BPM off-beat guitar skank" },
        "Rock": { icon: "fa-bolt-lightning", title: "Driving Hard Rock Riff", desc: "Distorted E5/G5 power chord" }
    };

    // State Variables
    let currentSelectedFile = null;
    let currentSelectedSampleUrl = null;
    let audioContext = null;
    let audioAnalyser = null;
    let audioSource = null;
    let isVisualizerRunning = false;

    // Chart Instances
    let radarChart = null;
    let mfccChart = null;
    let correlationChart = null;
    let pcaChart = null;
    let benchmarkChart = null;

    let benchmarkData = null;
    let edaData = null;

    // ==========================================
    // 1. Navigation Tab Handling
    // ==========================================
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");
            
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(tabId).classList.add("active");

            if (tabId === "profiler-tab") renderProfilerTab();
            if (tabId === "eda-tab") loadEdaTab();
            if (tabId === "benchmarks-tab") loadBenchmarkTab();
        });
    });

    // ==========================================
    // 2. Load Curated Preset Tracks from API
    // ==========================================
    async function loadPresetTracks() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            if (data.status === "success") {
                presetTracksGrid.innerHTML = "";
                data.samples.forEach(sample => {
                    const meta = presetMetadata[sample.genre] || { icon: "fa-music", title: `${sample.genre} Track`, desc: "Audio Sample" };
                    const card = document.createElement("div");
                    card.className = "preset-card";
                    card.innerHTML = `
                        <div style="display:flex; align-items:center; gap:6px; color: var(--primary-cyan);">
                            <i class="fa-solid ${meta.icon}"></i>
                            <span class="preset-title">${sample.genre}</span>
                        </div>
                        <span class="preset-subtitle">${meta.title}</span>
                    `;
                    card.addEventListener("click", () => selectPresetTrack(sample, meta, card));
                    presetTracksGrid.appendChild(card);
                });
            }
        } catch (e) {
            console.error("Failed to load preset tracks:", e);
        }
    }
    loadPresetTracks();

    function selectPresetTrack(sample, meta, cardElement) {
        document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("active"));
        cardElement.classList.add("active");

        currentSelectedFile = null;
        currentSelectedSampleUrl = sample.url;
        
        currentTrackTitle.textContent = meta.title;
        currentTrackGenre.textContent = `${sample.genre} • GTZAN Audio Benchmark`;

        mainAudioPlayer.src = sample.url;
        playerPanel.classList.remove("hidden");
        predictBtn.disabled = false;
    }

    // ==========================================
    // 3. Audio File Uploader & Drag-and-Drop
    // ==========================================
    browseBtn.addEventListener("click", () => audioFileInput.click());

    audioFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
    });

    function handleFileUpload(file) {
        document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("active"));
        currentSelectedFile = file;
        currentSelectedSampleUrl = null;

        currentTrackTitle.textContent = file.name;
        currentTrackGenre.textContent = `Custom Audio Upload (${(file.size / (1024*1024)).toFixed(2)} MB)`;

        const objectUrl = URL.createObjectURL(file);
        mainAudioPlayer.src = objectUrl;
        playerPanel.classList.remove("hidden");
        predictBtn.disabled = false;
    }

    // ==========================================
    // 4. Web Audio API Canvas Spectrum Visualizer
    // ==========================================
    mainAudioPlayer.addEventListener("play", () => {
        setupAudioContext();
        if (!isVisualizerRunning) {
            isVisualizerRunning = true;
            drawVisualizer();
        }
    });

    function setupAudioContext() {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            audioAnalyser = audioContext.createAnalyser();
            audioAnalyser.fftSize = 64;
            audioSource = audioContext.createMediaElementSource(mainAudioPlayer);
            audioSource.connect(audioAnalyser);
            audioAnalyser.connect(audioContext.destination);
        }
        if (audioContext.state === "suspended") {
            audioContext.resume();
        }
    }

    function drawVisualizer() {
        if (!audioAnalyser) return;
        const canvasCtx = waveformCanvas.getContext("2d");
        const bufferLength = audioAnalyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function renderFrame() {
            if (mainAudioPlayer.paused) {
                isVisualizerRunning = false;
                return;
            }
            requestAnimationFrame(renderFrame);
            audioAnalyser.getByteFrequencyData(dataArray);

            canvasCtx.fillStyle = "rgba(5, 7, 13, 0.4)";
            canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

            const barWidth = (waveformCanvas.width / bufferLength) * 1.5;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * waveformCanvas.height;
                const gradient = canvasCtx.createLinearGradient(0, waveformCanvas.height, 0, 0);
                gradient.addColorStop(0, "#00f2fe");
                gradient.addColorStop(1, "#f43f5e");

                canvasCtx.fillStyle = gradient;
                canvasCtx.fillRect(x, waveformCanvas.height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }
        renderFrame();
    }

    function resizeCanvas() {
        waveformCanvas.width = waveformCanvas.parentElement.clientWidth;
        waveformCanvas.height = waveformCanvas.parentElement.clientHeight;
    }
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    // Client-side Web Audio API PCM Decoder
    async function decodeAudioToPcm(fileOrBlob) {
        const arrayBuffer = await fileOrBlob.arrayBuffer();
        const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);
        const channelData = audioBuffer.getChannelData(0);
        
        const targetSr = 22050;
        const step = Math.max(1, Math.floor(audioBuffer.sampleRate / targetSr));
        const maxSamples = targetSr * 5;
        const pcm = [];

        for (let i = 0; i < channelData.length && pcm.length < maxSamples; i += step) {
            pcm.push(channelData[i]);
        }
        return { pcm, sample_rate: targetSr };
    }

    // ==========================================
    // 5. Predict Button Click & API Call
    // ==========================================
    predictBtn.addEventListener("click", async () => {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Audio Signal...';

        try {
            let res;
            if (currentSelectedFile) {
                const formData = new FormData();
                formData.append("model", modelSelect.value);
                formData.append("file", currentSelectedFile);

                res = await fetch("/api/predict", { method: "POST", body: formData });
                
                if (!res.ok) {
                    const pcmPayload = await decodeAudioToPcm(currentSelectedFile);
                    pcmPayload.model = modelSelect.value;
                    pcmPayload.filename = currentSelectedFile.name;

                    res = await fetch("/api/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(pcmPayload)
                    });
                }
            } else if (currentSelectedSampleUrl) {
                const audioBlob = await fetch(currentSelectedSampleUrl).then(r => r.blob());
                const formData = new FormData();
                formData.append("model", modelSelect.value);
                formData.append("file", audioBlob, currentTrackTitle.textContent);

                res = await fetch("/api/predict", { method: "POST", body: formData });
            }

            const data = await res.json();
            if (data.status === "success") {
                renderPredictionResults(data);
                fetchRecommendations(data.features);
            } else {
                alert(`Error analyzing audio: ${data.message || "Unknown error"}`);
            }
        } catch (e) {
            console.error("Prediction error:", e);
            alert(`Error analyzing audio: ${e.message}`);
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<i class="fa-solid fa-bolt-lightning"></i> Analyze & Predict Genre';
        }
    });

    function renderPredictionResults(data) {
        emptyState.classList.add("hidden");
        predictionView.classList.remove("hidden");

        const genreName = data.predicted_genre.toUpperCase();
        heroGenreTitle.textContent = genreName;
        heroConfidenceText.textContent = `${data.confidence}% Match Confidence (${data.model_used} Model)`;

        const meta = presetMetadata[data.predicted_genre] || { icon: "fa-music" };
        genreBadgeIcon.innerHTML = `<i class="fa-solid ${meta.icon}"></i>`;

        // Render Probability Bars
        probList.innerHTML = "";
        data.predictions.forEach(item => {
            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <div class="prob-meta">
                    <span>${item.genre}</span>
                    <span>${item.confidence}%</span>
                </div>
                <div class="prob-track-bg">
                    <div class="prob-track-fill" style="width: ${item.confidence}%"></div>
                </div>
            `;
            probList.appendChild(row);
        });

        window.activeAudioFeatures = data.features;
        window.activeAcousticProfile = data.acoustic_profile;
    }

    // Fetch AI Song Recommendations
    async function fetchRecommendations(features) {
        try {
            const res = await fetch("/api/recommend", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features })
            });
            const data = await res.json();
            if (data.status === "success") {
                recPanel.classList.remove("hidden");
                recGrid.innerHTML = "";
                data.recommendations.forEach(rec => {
                    const card = document.createElement("div");
                    card.className = "rec-item";
                    card.innerHTML = `
                        <div class="rec-top">
                            <span class="rec-genre-tag">${rec.genre}</span>
                            <span class="rec-match-pill">${rec.similarity}% Match</span>
                        </div>
                        <div class="rec-info">
                            <span>Track Sample (${rec.tempo} BPM)</span>
                        </div>
                        <audio controls src="${rec.audio_url}" class="custom-audio-element" style="height: 32px;"></audio>
                    `;
                    recGrid.appendChild(card);
                });
            }
        } catch (e) {
            console.error("Recommendations error:", e);
        }
    }

    // ==========================================
    // 6. Profiler Tab Charts
    // ==========================================
    function renderProfilerTab() {
        const profile = window.activeAcousticProfile || {
            "Energy": 0.75, "Tempo / 180": 0.68, "Brightness": 0.60,
            "Danceability": 0.70, "Harmonics": 0.65, "Dynamic Range": 0.55
        };
        const features = window.activeAudioFeatures || {};

        document.getElementById("profTempo").textContent = features.tempo ? `${Math.round(features.tempo)} BPM` : "120 BPM";
        document.getElementById("profCentroid").textContent = features.spectral_centroid_mean ? `${Math.round(features.spectral_centroid_mean)} Hz` : "2400 Hz";
        document.getElementById("profZcr").textContent = features.zero_crossing_rate_mean ? features.zero_crossing_rate_mean.toFixed(3) : "0.095";
        document.getElementById("profEnergy").textContent = profile.Energy.toFixed(2);
        document.getElementById("profDance").textContent = profile.Danceability.toFixed(2);
        document.getElementById("profRms").textContent = features.rms_mean ? features.rms_mean.toFixed(3) : "0.145";

        // Radar Chart
        const radarCtx = document.getElementById("radarChart").getContext("2d");
        if (radarChart) radarChart.destroy();
        radarChart = new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: Object.keys(profile),
                datasets: [{
                    label: 'Track Audio Profile',
                    data: Object.values(profile),
                    backgroundColor: 'rgba(0, 242, 254, 0.25)',
                    borderColor: '#00f2fe',
                    pointBackgroundColor: '#f43f5e',
                    pointBorderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
                        ticks: { display: false }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });

        // MFCC Chart
        const mfccLabels = Array.from({length: 20}, (_, i) => `MFCC ${i+1}`);
        const mfccValues = mfccLabels.map((_, i) => features[`mfcc${i+1}_mean`] || (Math.sin(i) * 15 - 40));

        const mfccCtx = document.getElementById("mfccChart").getContext("2d");
        if (mfccChart) mfccChart.destroy();
        mfccChart = new Chart(mfccCtx, {
            type: 'bar',
            data: {
                labels: mfccLabels,
                datasets: [{
                    label: 'MFCC Coefficient',
                    data: mfccValues,
                    backgroundColor: 'rgba(244, 63, 94, 0.65)',
                    borderColor: '#f43f5e',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // ==========================================
    // 7. EDA Tab Charts
    // ==========================================
    async function loadEdaTab() {
        if (edaData) return renderEdaCharts();
        try {
            const res = await fetch("/api/eda");
            edaData = await res.json();
            if (edaData.status === "success") renderEdaCharts();
        } catch (e) {
            console.error("EDA load error:", e);
        }
    }

    function renderEdaCharts() {
        if (!edaData) return;

        const corrCols = Object.keys(edaData.correlation_matrix);
        const tempoCorr = corrCols.map(c => edaData.correlation_matrix["tempo"][c]);

        const corrCtx = document.getElementById("correlationChart").getContext("2d");
        if (correlationChart) correlationChart.destroy();
        correlationChart = new Chart(corrCtx, {
            type: 'bar',
            data: {
                labels: corrCols.map(c => c.replace("_mean", "").replace("_", " ")),
                datasets: [{
                    label: 'Correlation with Tempo',
                    data: tempoCorr,
                    backgroundColor: tempoCorr.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(244, 63, 94, 0.7)'),
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#94a3b8', font: { size: 10 } } },
                    y: { ticks: { color: '#94a3b8' }, min: -1, max: 1 }
                }
            }
        });

        const genreColors = {
            "blues": "#3b82f6", "classical": "#6366f1", "country": "#eab308",
            "disco": "#ec4899", "hiphop": "#8b5cf6", "jazz": "#14b8a6",
            "metal": "#ef4444", "pop": "#f43f5e", "reggae": "#10b981", "rock": "#f97316"
        };

        const datasets = edaData.genres.map(g => {
            const points = edaData.pca_scatter.filter(p => p.label === g).map(p => ({ x: p.pca_x, y: p.pca_y }));
            return {
                label: g.toUpperCase(),
                data: points,
                backgroundColor: genreColors[g] || "#00f2fe",
                pointRadius: 4
            };
        });

        const pcaCtx = document.getElementById("pcaChart").getContext("2d");
        if (pcaChart) pcaChart.destroy();
        pcaChart = new Chart(pcaCtx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'PCA Component 1', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
                    y: { title: { display: true, text: 'PCA Component 2', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
                },
                plugins: { legend: { labels: { color: '#94a3b8', font: { size: 10 } } } }
            }
        });
    }

    // ==========================================
    // 8. Model Benchmarks Tab
    // ==========================================
    async function loadBenchmarkTab() {
        if (!benchmarkData) {
            try {
                const res = await fetch("/api/models");
                benchmarkData = await res.json();
            } catch (e) {
                console.error("Benchmark load error:", e);
                return;
            }
        }
        renderBenchmarkCharts();
    }

    function renderBenchmarkCharts() {
        if (!benchmarkData || !benchmarkData.benchmarks) return;

        const modelNames = Object.keys(benchmarkData.benchmarks);
        const accuracies = modelNames.map(m => (benchmarkData.benchmarks[m].accuracy * 100).toFixed(1));
        const f1Scores = modelNames.map(m => (benchmarkData.benchmarks[m].f1_score * 100).toFixed(1));

        const benchCtx = document.getElementById("benchmarkChart").getContext("2d");
        if (benchmarkChart) benchmarkChart.destroy();
        benchmarkChart = new Chart(benchCtx, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [
                    { label: 'Accuracy (%)', data: accuracies, backgroundColor: 'rgba(0, 242, 254, 0.7)', borderRadius: 6 },
                    { label: 'Weighted F1 (%)', data: f1Scores, backgroundColor: 'rgba(139, 92, 246, 0.7)', borderRadius: 6 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#94a3b8' } },
                    y: { ticks: { color: '#94a3b8' }, min: 80, max: 100 }
                }
            }
        });

        document.getElementById("matrixModelSelect").addEventListener("change", (e) => {
            renderConfusionTable(e.target.value);
        });

        renderConfusionTable("RandomForest");
    }

    function renderConfusionTable(modelName) {
        if (!benchmarkData || !benchmarkData.benchmarks[modelName]) return;
        const cm = benchmarkData.benchmarks[modelName].confusion_matrix;
        const genres = benchmarkData.genres || ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"];

        const table = document.getElementById("confusionTable");
        let html = `<thead><tr><th>Actual \\ Pred</th>${genres.map(g => `<th>${g.substring(0,3).toUpperCase()}</th>`).join("")}</tr></thead><tbody>`;

        cm.forEach((row, i) => {
            html += `<tr><th>${genres[i].substring(0,3).toUpperCase()}</th>`;
            row.forEach((val, j) => {
                const isDiag = i === j;
                html += `<td class="${isDiag ? 'cell-diag' : ''}">${val}</td>`;
            });
            html += `</tr>`;
        });

        html += `</tbody>`;
        table.innerHTML = html;
    }

});
