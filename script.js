let model;
let trainingImages = [];
let trainingLabels = [];

const shapeLabels = ['Circle', 'Square', 'Triangle'];
const imageFiles = [
    { src: 'training_circle.jpeg', label: 0 },
    { src: 'training_square.jpeg', label: 1 },
    { src: 'training_triangle.jpeg', label: 2 }
];

async function loadAndSliceImage(imageSrc, label) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            const images = [];
            for (let row = 0; row < 10; row++) {
                for (let col = 0; col < 20; col++) {
                    const imageData = ctx.getImageData(col * 28, row * 28, 28, 28);
                    const grayImage = [];
                    for (let i = 0; i < imageData.data.length; i += 4) {
                        grayImage.push(imageData.data[i] / 255); // Normalize
                    }
                    images.push(grayImage);
                    trainingLabels.push(label);
                }
            }
            resolve(images);
        };
        img.src = imageSrc;
    });
}

async function startTraining() {
    document.getElementById('trainingStatus').innerText = 'Loading and slicing images...';

    trainingImages = [];
    trainingLabels = [];

    let allImages = [];

    for (let i = 0; i < imageFiles.length; i++) {
        const images = await loadAndSliceImage(imageFiles[i].src, imageFiles[i].label);
        allImages = allImages.concat(images);
    }

    const xs = tf.tensor4d(allImages, [allImages.length, 28, 28, 1]);
    const ys = tf.oneHot(tf.tensor1d(trainingLabels, 'int32'), 3);

    document.getElementById('trainingStatus').innerText = 'Building model...';

    model = tf.sequential();
    model.add(tf.layers.conv2d({inputShape: [28, 28, 1], filters: 16, kernelSize: 3, activation: 'relu'}));
    model.add(tf.layers.maxPooling2d({poolSize: 2}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({units: 3, activation: 'softmax'}));

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    document.getElementById('trainingStatus').innerText = 'Training model...';

    await model.fit(xs, ys, {
        epochs: 10,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                document.getElementById('trainingStatus').innerText = 
                    `Epoch ${epoch + 1}: Accuracy = ${logs.acc.toFixed(4)}`;
            }
        }
    });

    document.getElementById('trainingStatus').innerText = 'Training completed!';
}

function clearCanvas() {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function getCanvasImage() {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');

    const smallCanvas = document.createElement('canvas');
    smallCanvas.width = 28;
    smallCanvas.height = 28;
    const smallCtx = smallCanvas.getContext('2d');
    smallCtx.drawImage(canvas, 0, 0, 28, 28);
    const smallImageData = smallCtx.getImageData(0, 0, 28, 28);

    const grayImage = [];
    for (let i = 0; i < smallImageData.data.length; i += 4) {
        grayImage.push(smallImageData.data[i] / 255);
    }

    return tf.tensor4d(grayImage, [1, 28, 28, 1]);
}

function predictShape() {
    if (!model) {
        alert('Please train the model first.');
        return;
    }

    const inputTensor = getCanvasImage();

    const prediction = model.predict(inputTensor);
    prediction.array().then(predArray => {
        const classIndex = predArray[0].indexOf(Math.max(...predArray[0]));
        const predictedShape = shapeLabels[classIndex];
        const confidence = (predArray[0][classIndex] * 100).toFixed(2);

        document.getElementById('predictionResult').innerText = 
            `Predicted Shape: ${predictedShape} (Confidence: ${confidence}%)`;
    });
}

window.onload = clearCanvas;
