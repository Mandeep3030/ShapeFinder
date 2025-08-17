// STEP 0 ========= HEADER & GLOBALS =========
// Shape detector - Two AI models (sequential training)
// Author: Mandeep Singh | Date: 25 June 2025

// STEP 1 ========= GLOBAL VARIABLES =========
console.log('****SEC 1. GLOBAL VARIABLES: Defining model, labels, and image file sources.****');
let model1;
let model2;
const shapeLabels = ['Square', 'Triangle', 'Circle'];
const imageFiles = [
    { src: 'training_square.png', label: 0 },
    { src: 'training_triangle.png', label: 1 },
    { src: 'training_circle.png', label: 2 }
];

// STEP 2 ========= IMAGE LOADING & SLICING =========
// Loads and slices training images into 32x32 grayscale samples
console.log('****SEC 2. IMAGE LOADING & SLICING: Loading and slicing training images into 32x32 samples.****');
async function loadAndSliceImage(imageSrc) {
    console.log(`  [IMG LOAD] Loading and slicing image: ${imageSrc}`);
    document.getElementById('training-status1').innerText = 'Loading and slicing images...';
    document.getElementById('training-status2').innerText = 'Loading and slicing images...';
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(img, 0, 0);
            const images = [];
            const sliceWidth = 32;
            const sliceHeight = 32;
            const cols = 20;
            const rows = 10;
            for (let row = 0; row < rows; row++) {
                for (let col = 0; col < cols; col++) {
                    const imageData = ctx.getImageData(col * sliceWidth, row * sliceHeight, sliceWidth, sliceHeight);
                    const grayImage = [];
                    for (let y = 0; y < sliceHeight; y++) {
                        const rowPixels = [];
                        for (let x = 0; x < sliceWidth; x++) {
                            const index = (y * sliceWidth + x) * 4;
                            const pixelValue = imageData.data[index] > 127 ? 1 : 0;
                            rowPixels.push([pixelValue]);
                        }
                        grayImage.push(rowPixels);
                    }
                    images.push(grayImage);
                    //for (let aug = 0; aug < 2; aug++) {
                    //    const augImage = augmentImage(grayImage);
                    //    images.push(augImage);
                    //}
                }
            }

            document.getElementById('training-status1').innerText = 'Images loaded and sliced.';
            document.getElementById('training-status2').innerText = 'Images loaded and sliced.';
            resolve(images);

        };
        img.src = imageSrc;
    });
}

// STEP 3 ========= DATA AUGMENTATION =========
// Augments a 32x32 image sample with random rotation and shift
console.log('****SEC 3. DATA AUGMENTATION: Augmenting images with rotation and shift.****');
function augmentImage(grayImage) {
    console.log('  [AUGMENT] Augmenting a 32x32 image sample.');
    let tensor = tf.tensor2d(grayImage.map(row => row.map(pixel => pixel[0])), [32, 32]);
    tensor = tensor.expandDims(0).expandDims(-1);
    const angle = (Math.random() - 0.5) * (Math.PI / 6);
    tensor = tf.image.rotateWithOffset(tensor, angle, 0, 0.5);
    const dx = Math.floor((Math.random() - 0.5) * 6);
    const dy = Math.floor((Math.random() - 0.5) * 6);
    let padTop = Math.max(dy, 0);
    let padBottom = Math.max(-dy, 0);
    let padLeft = Math.max(dx, 0);
    let padRight = Math.max(-dx, 0);
    tensor = tf.pad(tensor, [[0, 0], [padTop, padBottom], [padLeft, padRight], [0, 0]], 0);
    tensor = tensor.slice([0, padTop, padLeft, 0], [1, 32, 32, 1]);
    tensor = tensor.squeeze();
    tensor = tensor.greater(0.5).cast('int32');
    const augArray = tensor.arraySync().map(row => row.map(val => [val]));
    tensor.dispose();
    return augArray;
}

// STEP 4.1 ========= MODEL 1 TRAINING (5-Layer) =========
// Builds, compiles, and trains Model 1
console.log('****SEC 4. MODEL 1 TRAINING: Building, compiling, and training the model.****');
async function startTrainingModel1() {


    let allImages = [];
    let allLabels = [];
    for (let i = 0; i < imageFiles.length; i++) {
        console.log(`  [TRAIN] Loading and augmenting images for label ${imageFiles[i].label}`);
        const images = await loadAndSliceImage(imageFiles[i].src);
        allImages = allImages.concat(images);
        for (let j = 0; j < images.length; j++) allLabels.push(imageFiles[i].label);
    }
    const xs = tf.tensor4d(allImages, [allImages.length, 32, 32, 1]);
    const ys = tf.oneHot(tf.tensor1d(allLabels, 'int32'), 3);


    console.log('  [TRAIN] Training data prepared, starting model training.');
    model1 = tf.sequential();
    model1.add(tf.layers.conv2d({ inputShape: [32, 32, 1], filters: 32, kernelSize: 3, activation: 'relu' }));
    model1.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model1.add(tf.layers.flatten());
    model1.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model1.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model1.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    await model1.fit(xs, ys, {
        epochs: 7,
        callbacks: {
            onEpochEnd: (epoch, logs) => { // update status and bar
                console.log(`    [EPOCH] Epoch ${epoch + 1}: Accuracy = ${logs.acc}, Loss = ${logs.loss}`);
                document.getElementById('training-status1').innerText = `Epoch ${epoch + 1}: Accuracy = ${logs.acc.toFixed(4)}`;
                document.getElementById('accuracy-model1').innerText = `Accuracy: ${(logs.acc * 100).toFixed(2)}%`;
                document.getElementById('loss-model1').innerText = `Loss: ${logs.loss.toFixed(4)}`;
                document.getElementById('epochs-model1').innerText = `Epochs: ${epoch + 1}`;

                const progressBar = document.getElementById('progress-bar1');
                if (progressBar) {
                    const progress = Math.round(((epoch + 1) / 7) * 100);
                    progressBar.style.width = `${progress}%`;
                    progressBar.setAttribute('aria-valuenow', progress);
                }
                const percent = Math.round(logs.acc * 100);
                drawGraph('graph1', [percent, percent, percent]);
            }
        }
    });
    document.getElementById('training-status1').innerText = 'Training completed!';
    document.getElementById('images-model1').innerText = `Images: ${allImages.length}`;

    xs.dispose();
    ys.dispose();
}

// STEP 4.2 ========= MODEL 2 TRAINING (8-Layer) =========
// Builds, compiles, and trains Model 2
console.log('****SEC 4.5. MODEL 2 TRAINING: Building, compiling, and training the model.****');
async function startTrainingModel2() {


    let allImages = [];
    let allLabels = [];
    for (let i = 0; i < imageFiles.length; i++) {
        console.log(`  [TRAIN] Loading and augmenting images for label ${imageFiles[i].label}`);
        const images = await loadAndSliceImage(imageFiles[i].src);

        // Augment images: for each image, add original and two augmentations
        let augmentedImages = [];
        for (let j = 0; j < images.length; j++) {
            const grayImage = images[j];
            augmentedImages.push(grayImage); // original
            for (let aug = 0; aug < 2; aug++) {
                const augImage = augmentImage(grayImage);
                augmentedImages.push(augImage);
            }
        }
        allImages = allImages.concat(augmentedImages);
        for (let j = 0; j < augmentedImages.length; j++) allLabels.push(imageFiles[i].label);
    }
    const xs = tf.tensor4d(allImages, [allImages.length, 32, 32, 1]);
    const ys = tf.oneHot(tf.tensor1d(allLabels, 'int32'), 3);


    console.log('  [TRAIN] Training data prepared, starting model training.');
    model2 = tf.sequential();
    model2.add(tf.layers.conv2d({ inputShape: [32, 32, 1], filters: 32, kernelSize: 3, activation: 'relu' })); // 1
    model2.add(tf.layers.maxPooling2d({ poolSize: 2 })); // 2
    model2.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' })); // 3
    model2.add(tf.layers.maxPooling2d({ poolSize: 2 })); // 4
    model2.add(tf.layers.flatten()); // 5
    model2.add(tf.layers.dense({ units: 128, activation: 'relu' })); // 6
    model2.add(tf.layers.dropout({ rate: 0.3 })); // 7
    model2.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // 8
    model2.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    await model2.fit(xs, ys, {
        epochs: 10,
        callbacks: {
            onEpochEnd: (epoch, logs) => { // update status and bar
                console.log(`    [EPOCH] Epoch ${epoch + 1}: Accuracy = ${logs.acc}, Loss = ${logs.loss}`);
                document.getElementById('training-status2').innerText = `Epoch ${epoch + 1}: Accuracy = ${logs.acc.toFixed(4)}`;
                document.getElementById('accuracy-model2').innerText = `Accuracy: ${(logs.acc * 100).toFixed(2)}%`;
                document.getElementById('loss-model2').innerText = `Loss: ${logs.loss.toFixed(4)}`;
                document.getElementById('epochs-model2').innerText = `Epochs: ${epoch + 1}`;

                const progressBar = document.getElementById('progress-bar2');
                if (progressBar) {
                    const progress = Math.round(((epoch + 1) / 10) * 100);
                    progressBar.style.width = `${progress}%`;
                    progressBar.setAttribute('aria-valuenow', progress);
                }
                const percent = Math.round(logs.acc * 100);
                drawGraph('graph2', [percent, percent, percent]);
            }
        }
    });
    document.getElementById('training-status2').innerText = 'Training completed!';
    document.getElementById('images-model2').innerText = `Images: ${allImages.length}`;

    xs.dispose();
    ys.dispose();
}

// STEP 4.3 ========= TRAIN BOTH MODELS SEQUENTIALLY =========
// Trains Model 1, then Model 2
async function trainBothModelsSequentially() {
    await startTrainingModel1();
    await startTrainingModel2();
}

// STEP 5 ========= PREDICTION =========
// Predicts the shape from user canvas for both models
console.log('****SEC 5. PREDICTION: Predicting the shape from user canvas.****');
function predictShape1() {
    console.log('  [PREDICT] Predicting shape for Model 1 from canvas.');
    if (!model1) {
        alert('Please train Model 1 first.');
        return;
    }
    const inputTensor = getCanvasImageData('canvas');
    const prediction1 = model1.predict(inputTensor);
    prediction1.array().then(predArray => {
        const probs = predArray[0];
        const classIndex = probs.indexOf(Math.max(...probs));
        const predictedShape = shapeLabels[classIndex];
        const confidence = (probs[classIndex] * 100).toFixed(2);
        // Convert probabilities to percentages for the graph
        const percentages = probs.map(p => p * 100);
        console.log('  [PREDICT] Raw probabilities:', probs);
        console.log('  [PREDICT] Percentages for graph:', percentages);
        drawGraph('graph1', percentages);
        const allProbsText = probs.map((p, i) => `${shapeLabels[i]}: ${(p * 100).toFixed(2)}%`).join(' | ');
    });


    console.log('  [PREDICT] Predicting shape for Model 2 from canvas.');
    if (!model2) {
        alert('Please train Model 2 first.');
        return;
    }
    const prediction2 = model2.predict(inputTensor);
    prediction2.array().then(predArray => {
        const probs = predArray[0];
        const classIndex = probs.indexOf(Math.max(...probs));
        const predictedShape = shapeLabels[classIndex];
        const confidence = (probs[classIndex] * 100).toFixed(2);
        // Convert probabilities to percentages for the graph
        const percentages = probs.map(p => p * 100);
        console.log('  [PREDICT] Raw probabilities:', probs);
        console.log('  [PREDICT] Percentages for graph:', percentages);
        drawGraph('graph2', percentages);
        const allProbsText = probs.map((p, i) => `${shapeLabels[i]}: ${(p * 100).toFixed(2)}%`).join(' | ');
    });
}


