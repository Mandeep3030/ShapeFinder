// STEP 0 ========= BAR GRAPH & UI HELPERS =========
// Draws bar graphs and sets up initial UI
function showResults() {
    console.log('  [BAR GRAPH] Drawing initial dummy bar graphs for both models.');
    drawGraph('graph1', [30, 50, 20]);
    drawGraph('graph2', [20, 60, 20]);
}

// Draws a bar graph for given canvas and percentages
function drawGraph(canvasId, percentages) {
    console.log(`  [BAR GRAPH] Drawing bar graph for ${canvasId} with values:`, percentages);
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const labels = ['Square', 'Triangle', 'Circle'];
    const colors = ['#a29bfe', '#fd79a8', '#74b9ff'];

    percentages.forEach((percent, index) => {
        const centerX = 75 + index * 80;
        // Draw bar
        const barHeight = percent * 2;
        const barX = 50 + index * 80;
        const barY = canvas.height - barHeight - 20;
        const barWidth = 50;
        const radius = 12;
        ctx.fillStyle = colors[index];
        ctx.beginPath();
        ctx.moveTo(barX, barY + barHeight); // bottom left
        ctx.lineTo(barX, barY + radius); // up left
        ctx.quadraticCurveTo(barX, barY, barX + radius, barY); // top left corner
        ctx.lineTo(barX + barWidth - radius, barY); // top edge
        ctx.quadraticCurveTo(barX + barWidth, barY, barX + barWidth, barY + radius); // top right corner
        ctx.lineTo(barX + barWidth, barY + barHeight); // down right
        ctx.closePath();
        ctx.fill();

        // Draw label
        ctx.fillStyle = '#000';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(labels[index], centerX, canvas.height - 5);

        // Draw percentage
        ctx.fillText(`${Math.round(percent)}%`, centerX, canvas.height - barHeight - 30);
    });
}

// STEP 1 ========= CANVAS DRAWING & CLEAR =========
// Sets up canvas drawing events
function setupCanvas() {
    console.log(`  [CANVAS] Initializing drawing events for canvas.`);
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    canvas.addEventListener('mousedown', (e) => {
        canvas.drawing = true;
        ctx.beginPath();
    });
    canvas.addEventListener('mouseup', () => {
        canvas.drawing = false;
        ctx.beginPath();
    });
    canvas.addEventListener('mousemove', (e) => {
        if (!canvas.drawing) return;
        const rect = canvas.getBoundingClientRect();
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'black';
        ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        predictShape1();
    });

}
// Clears the canvas
function clearCanvas() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawGraph('graph1', [0, 0, 0]);
    drawGraph('graph2', [0, 0, 0]);
};

// STEP 2 ========= CANVAS IMAGE EXTRACTION =========
// Extracts and preprocesses canvas image for prediction
function getCanvasImageData(canvasId) {
    console.log(`  [CANVAS] Extracting and preprocessing image from canvas: ${canvasId}`);
    const canvas = document.getElementById(canvasId);
    const smallCanvas = document.createElement('canvas');
    smallCanvas.width = 32;
    smallCanvas.height = 32;
    const smallCtx = smallCanvas.getContext('2d', { willReadFrequently: true });
    smallCtx.fillStyle = 'white';
    smallCtx.fillRect(0, 0, 32, 32);
    smallCtx.drawImage(canvas, 0, 0, 32, 32);
    const smallImageData = smallCtx.getImageData(0, 0, 32, 32);
    const grayImage = [];
    for (let y = 0; y < 32; y++) {
        const rowPixels = [];
        for (let x = 0; x < 32; x++) {
            const index = (y * 32 + x) * 4;
            const pixelValue = smallImageData.data[index] > 127 ? 1 : 0;
            rowPixels.push([pixelValue]);
        }
        grayImage.push(rowPixels);
    }
    return tf.tensor4d([grayImage]);
}

// STEP 3 ========= TRAINING PROGRESS BAR (OPTIONAL) =========
// Simulates training progress bar (not used in main flow)
function simulateTraining(progress) {
    console.log(`  [TRAINING SIM] Simulating training progress for ${progress}.`);
    // Support both model1 and model2
    const progressBar = document.getElementById(`progress-bar${progress === 'model1' ? '1' : '2'}`);
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
    }
}

// STEP 4 ========= DOMContentLoaded: UI WIRING =========
document.addEventListener('DOMContentLoaded', () => {
    showResults();
    setupCanvas();
    // Predict button
    const predictBtn = document.getElementById('test-button');
    if (predictBtn) predictBtn.onclick = predictShape1;
    // Train button (trains both models sequentially)
    const trainBtn = document.getElementById('trainModel1Btn');
    if (trainBtn) trainBtn.onclick = trainBothModelsSequentially;
    // Clear button
    const clearBtn = document.getElementById('clear-canvas');
    if (clearBtn) clearBtn.onclick = clearCanvas;
});


