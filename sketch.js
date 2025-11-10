let classifier;
let trainBtn;
let uploadInput;
let status;
let result;
let uploadedImage;
let isModelReady = false;
let currentImage = null;
let label = "Iniciando...";

const cubiertos = ['cuchara', 'cuchillo', 'tenedor'];

function setup() {
    createCanvas(640, 480);
    trainBtn = select('#trainBtn');
    uploadInput = select('#uploadInput');
    status = select('#status');
    result = select('#result');

    ml5.setBackend("webgl");

    const options = {
        task: 'classification',
        inputs: [224, 224, 4],
        outputs: cubiertos.length,
        epochs: 30,
        batchSize: 16,
        numLabels: 3
    };
    
    classifier = ml5.neuralNetwork(options);

    try {
        classifier.load('model/model.json', modelLoaded);
    } catch (error) {
        console.log('No se ha encontrado un modelo preentrenado. El modelo está listo para entrenar.');
        modelLoaded();
    }

    trainBtn.mousePressed(() => {
        console.log('Iniciando Entrenamiento');
        trainModel();
    });
    
    uploadInput.changed((e) => {
        console.log('Archivo cargado');
        handleUpload(e);
    });
}

function modelLoaded() {
    console.log('Modelo Listo');
    status.html('Modelo listo. Puede cargar una imagen o entrenar un nuevo modelo.');
    label = 'Modelo listo. Cargue una imagen';
    isModelReady = true;
}

async function trainModel() {
    console.log('Comenzando a cargar imágenes para entrenamiento...');
    status.html('Cargando imágenes para entrenamiento...');
    label = 'Cargando imágenes...';
    
    try {
        await addImagesToClassifier();
        status.html('Todas las imágenes cargadas. Iniciando entrenamiento...');
        label = 'Iniciando entrenamiento...';
        
        const trainingOptions = {
            epochs: 30,
            batchSize: 16
        };

        classifier.train(trainingOptions, (epoch, loss) => {
            const progress = ((epoch + 1) / trainingOptions.epochs * 100).toFixed(1);
            status.html(`Entrenando: Época: ${epoch}, Perdida: ${loss}`);
            label = `Entrenando ${progress}%`;
        }, () => {
            console.log('Entrenamiento completo');
            status.html('Entrenamiento completo. Listo para clasificar');
            label = '¡Entrenamiento completo!';
            isModelReady = true;
            classifier.save('model');
        });
    } catch (error) {
        console.error("Error en entrenamiento:", error);
        status.html('Error durante el entrenamiento. Revise la consola para más detalles.');
    }
}

function addImagesToClassifier() {
    const promises = [];
    let loadedImages = 0;
    
    cubiertos.forEach(cubierto => {
        for (let i = 1; i <= 50; i++) {
            const imgPath = `data/${cubierto}/${cubierto}(${i}).jpg`;
            const promise = new Promise((resolve, reject) => {
                loadImage(imgPath, img => {
                    loadedImages++;
                    status.html(`Loading images: ${loadedImages}/${cubiertos.length * 50}`);
                    let buffer = createGraphics(224, 224);
                    buffer.image(img, 0, 0, 224, 224);
                    classifier.addData(
                        { image: buffer.get() }, 
                        { label: cubierto }
                    );
                    
                    buffer.remove();
                    resolve();
                }, () => {
                    console.error(`Falló al cargar imagen: ${imgPath}`);
                    reject(new Error(`Falló al cargar imagen: ${imgPath}`));
                });
            });
            promises.push(promise);
        }
    });
    
    return Promise.all(promises).then(() => {
        console.log('Normalizando datos...');
        classifier.normalizeData();
    });
}

function handleUpload(event) {
    if (event.target.files && event.target.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            loadImage(e.target.result, img => {
                let buffer = createGraphics(224, 224);
                buffer.image(img, 0, 0, 224, 224);
                uploadedImage = buffer.get();
                currentImage = img; // Guardamos la imagen original para mostrarla
                buffer.remove();
                
                if (isModelReady) {
                    classifyImage();
                } else {
                    status.html('Entrene el modelo primero.');
                    label = 'Entrene el modelo primero';
                }
            });
        };
        reader.readAsDataURL(event.target.files[0]);
    }
}

function classifyImage() {
    if (!classifier || !isModelReady) {
        status.html('Modelo no listo, por favor entrene el modelo primero.');
        label = 'Modelo no listo';
        return;
    }

    status.html('Clasificando imagen...');
    label = 'Clasificando...';
    
    classifier.classify({ image: uploadedImage }, (error, results) => {
        if (error) {
            console.error('Error de clasificación:', error);
            status.html('Error clasificando imagen.');
            label = 'Error en la clasificación';
            return;
        }
        
        if (results && results[0]) {
            const confidence = results[0].confidence * 100;
            if (confidence > 80) {
                result.html(`Clasificación: ${results[0].label} (${confidence.toFixed(2)}% confident)`);
                label = `${results[0].label} (${confidence.toFixed(2)}% confianza)`;
            } else {
                result.html(`Predicción de baja confianza: ${results[0].label} (${confidence.toFixed(2)}%)`);
                label = `${results[0].label} (${confidence.toFixed(2)}% - Baja confianza)`;
            }
            status.html('Clasificación completa.');
        }
    });
}

function draw() {
    background(220);
    
    // Si hay una imagen cargada, mostrarla en el canvas
    if (uploadedImage) {
        currentImage = uploadedImage;
    }
    
    if (currentImage) {
        const scale = min(width / currentImage.width, height / currentImage.height) * 0.8;
        const w = currentImage.width * scale;
        const h = currentImage.height * scale;
        const x = (width - w) / 2;
        const y = (height - h) / 2;
        image(currentImage, x, y, w, h);
    }
    
    // Mostrar el texto del estado actual
    textAlign(CENTER, CENTER);
    textSize(32);
    fill(0);
    noStroke();
    text(label, width/2, 40);
}
