/*
 * 👋 Hello! This is an ml5.js example made and shared with ❤️.
 * Learn more about the ml5.js project: https://ml5js.org/
 * ml5.js license and Code of Conduct: https://github.com/ml5js/ml5-next-gen/blob/main/LICENSE.md
 *
 * This example demonstrates training a color classifier through ml5.neuralNetwork.
 */

// Variables para el manejo de imágenes y modelo
let classifier;
let currentImage;
let ready = false;
let images = [];
let labels = [];

// Asegurar que TensorFlow.js use el backend correcto
async function initializeML() {
    await ml5.tf.setBackend('webgl');
    await ml5.tf.ready();
    console.log('TensorFlow backend:', ml5.tf.getBackend());
}

// Función para cargar imágenes
function preload() {
  // Cargar imágenes de anverso
  for (let i = 1; i <= 10; i++) {
    let img = loadImage(`data/raw/1000_anverso/1000(${i}).jpg`);
    images.push(img);
    labels.push("anverso");
  }
  
  // Cargar imágenes de reverso
  for (let i = 1; i <= 10; i++) {
    let img = loadImage(`data/raw/1000_reverso/1000(${i}).jpg`);
    images.push(img);
    labels.push("reverso");
  }
}

async function setup() {
  // Crear el canvas dentro del contenedor específico
  let canvas = createCanvas(400, 400);
  canvas.parent('canvas-container');
  background(255);
  
  try {
    // Inicializar ML antes de cualquier operación
    await initializeML();
    
    // Step 2: Configurar las opciones de la red neuronal
    let options = {
      inputs: [224, 224, 4], // Tamaño de entrada para las imágenes
      task: 'imageClassification',
      debug: true
    };

    // Step 3: Inicializar la red neuronal
    classifier = ml5.neuralNetwork(options);
    
    // Step 4: Procesar y agregar las imágenes al conjunto de entrenamiento
    processImages();
  } catch (error) {
    console.error('Error durante la inicialización:', error);
    updateStatus('Error durante la inicialización. Por favor, recarga la página.');
  }
}
// Función para procesar las imágenes y agregarlas al conjunto de entrenamiento
function processImages() {
  updateStatus('Procesando imágenes de entrenamiento...');
  
  for (let i = 0; i < images.length; i++) {
    let img = images[i];
    // Redimensionar la imagen a 224x224
    img.resize(224, 224);
    let imageData = {
      data: img,
      label: labels[i]
    };
    classifier.addData({ image: imageData.data }, { label: imageData.label });
    updateStatus(`Procesando imágenes: ${i + 1}/${images.length}`);
  }
  
  updateStatus('Comenzando entrenamiento...');
  // Normalizar los datos y comenzar el entrenamiento
  classifier.normalizeData();
  classifier.train({ 
    epochs: 50,
    batchSize: 16 
  }, finishedTraining, onEpochEnd);
}

// Función para mostrar el progreso del entrenamiento
function onEpochEnd(epoch, logs) {
  updateStatus(`Entrenando... Época ${epoch + 1}/50 - Pérdida: ${logs.loss.toFixed(4)}`);
}

// Step 7: Cuando termine el entrenamiento
function finishedTraining() {
  ready = true;
  updateStatus('¡Modelo listo! Arrastra una imagen o haz clic para seleccionar una.');
}

// Función para manejar el arrastre de archivos
function dropHandler(file) {
  if (file.type === 'image') {
    updateStatus('Cargando imagen...');
    console.log("cargando imagen...")
    currentImage = loadImage(file.data, imageReady);
  }
}

// Cuando la imagen esté cargada, clasificarla
function imageReady() {
  updateStatus('Procesando imagen...');
  console.log("procesando imagen...")
  currentImage.resize(224, 224);
  classify();
}

// Step 8: Clasificar la imagen
function classify() {
  if (currentImage && ready) {
    classifier.classify({ image: currentImage }, handleResults);
  }
}

function draw() {
  background(255);
  
  // Mostrar la imagen actual si existe
  if (currentImage) {
    // Calcular el factor de escala para ajustar la imagen al canvas
    let scale = min(width / currentImage.width, height / currentImage.height) * 0.8;
    let scaledW = currentImage.width * scale;
    let scaledH = currentImage.height * scale;
    
    // Centrar la imagen en el canvas
    let x = (width - scaledW) / 2;
    let y = (height - scaledH) / 2;
    
    // Dibujar la imagen escalada
    image(currentImage, x, y, scaledW, scaledH);
  } else {
    // Mostrar mensaje cuando no hay imagen
    textAlign(CENTER, CENTER);
    textSize(16);
    fill(150);
    text('La imagen clasificada aparecerá aquí', width/2, height/2);
  }
}

// Step 9: Manejar los resultados de la clasificación
function handleResults(error, results) {
  if (error) {
    console.error(error);
    updateStatus('Error al clasificar la imagen');
    return;
  }
  
  const prediction = results[0];
  const confidence = nf(prediction.confidence * 100, 2, 1);
  updateStatus(`Resultado: ${prediction.label} (Confianza: ${confidence}%)`);
}