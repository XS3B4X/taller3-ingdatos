let classifier;
let imageUpload;
let currentImage = null;
let label = "Iniciando...";
let isTraining = false;
let imagesLoaded = 0;
const IMAGE_SIZE = 160;
const MODEL_PATH = './model/clasificador_utencilios';

const TRAIN_SIZE = 46; // Imágenes para entrenamiento por clase
const TEST_SIZE = 7; // Imágenes para prueba por clase
const VAL_SIZE = 7; // Imágenes para validación por clase
const TOTAL_PER_CLASS = TRAIN_SIZE + TEST_SIZE + VAL_SIZE;
const TOTAL_IMAGES = TOTAL_PER_CLASS * 3;

let trainImages = {
  cuchara: [],
  cuchillo: [],
  tenedor: []
};

let testImages = {
  cuchara: [],
  cuchillo: [],
  tenedor: []
};

let valImages = {
  cuchara: [],
  cuchillo: [],
  tenedor: []
};

function preload() {
  loadTrainingImages();
}

function setup() {
  createCanvas(640, 480);

  ml5.setBackend("webgl");

  imageUpload = createFileInput(handleImage);
  imageUpload.position(10, 10);

  const options = {
    inputs: [IMAGE_SIZE, IMAGE_SIZE, 4],
    task: 'imageClassification',
  };

  classifier = ml5.neuralNetwork(options);

  try {
    fetch(MODEL_PATH + '.json')
      .then(response => {
        if (!response.ok) {
          throw new Error('Modelo no encontrado');
        }
        return classifier.load(MODEL_PATH, modelLoaded);
      })
      .catch(error => {
        console.log('Iniciando entrenamiento nuevo:', error.message);
        loadAndTrainModel();
      });
  } catch (error) {
    console.log('Error al cargar modelo, iniciando entrenamiento nuevo');
    loadAndTrainModel();
  }
}

function modelLoaded(error, model) {
  if (error) {
    console.log('Error al cargar el modelo, iniciando entrenamiento nuevo:', error);
    loadAndTrainModel();
  } else {
    console.log('Modelo cargado exitosamente');
    label = 'Listo para clasificar. Sube una imagen...';
  }
}

function loadAndTrainModel() {
  const checkImages = setInterval(() => {
    if (imagesLoaded >= TOTAL_IMAGES && !isTraining) {
      console.log('Imágenes cargadas, comenzando entrenamiento...');
      isTraining = true;
      clearInterval(checkImages);
      addTrainingData();
      
      const trainingOptions = {
        epochs: 35,
        batchSize: 16,
        learningRate: 0.0005
      };
      
      classifier.train(trainingOptions, finishedTraining);
    }
  }, 500);
}
function finishedTraining() {
  console.log('¡Entrenamiento completado!');
  label = 'Guardando modelo...';
  
  classifier.save('clasificador_utencilios', () => {
    console.log('Modelo guardado exitosamente');
    label = 'Listo para clasificar. Sube una imagen...';
  });
}

function loadTrainingImages() {
  for (let i = 1; i <= 50; i++) {
    loadImage(
      `data/cuchara/cuchara(${i}).jpg`,
      img => {
        img.resize(IMAGE_SIZE, IMAGE_SIZE);
        cucharaImages.push(img);
        imagesLoaded++;
      },
      () => console.log(`No se pudo cargar cuchara${i}.jpg`)
    );
    
    loadImage(
      `data/cuchillo/cuchillo(${i}).jpg`,
      img => {
        img.resize(IMAGE_SIZE, IMAGE_SIZE);
        cuchilloImages.push(img);
        imagesLoaded++;
      },
      () => console.log(`No se pudo cargar cuchillo${i}.jpg`)
    );
    
    loadImage(
      `data/tenedor/tenedor(${i}).jpg`,
      img => {
        img.resize(IMAGE_SIZE, IMAGE_SIZE);
        tenedorImages.push(img);
        imagesLoaded++;
      },
      () => console.log(`No se pudo cargar tenedor${i}.jpg`)
    );
  }
}

function addTrainingData() {
  // Agregar datos de entrenamiento
  for (const className in trainImages) {
    for (let img of trainImages[className]) {
      classifier.addData({ image: img }, { label: className });
    }
  }
  
  console.log('Datos agregados al conjunto de entrenamiento:');
  console.log(`Cucharas: ${trainImages.cuchara.length}`);
  console.log(`Cuchillos: ${trainImages.cuchillo.length}`);
  console.log(`Tenedores: ${trainImages.tenedor.length}`);
  
  // Normalizar los datos
  classifier.normalizeData();
}

// Función para evaluar el modelo con el conjunto de validación
function evaluateModel() {
  let correctPredictions = 0;
  let totalPredictions = 0;
  
  // Evaluar cada clase
  for (const className in valImages) {
    for (let img of valImages[className]) {
      classifier.classify({ image: img }, (error, results) => {
        if (!error) {
          totalPredictions++;
          if (results[0].label === className) {
            correctPredictions++;
          }
          
          // Mostrar precisión cuando se complete la evaluación
          if (totalPredictions === Object.values(valImages).flat().length) {
            const accuracy = (correctPredictions / totalPredictions) * 100;
            console.log(`Precisión en validación: ${accuracy.toFixed(2)}%`);
          }
        }
      });
    }
  }
}

function handleImage(file) {
  if (file.type === 'image') {
    label = 'Procesando...';
    currentImage = loadImage(file.data, img => {
      img.resize(IMAGE_SIZE, IMAGE_SIZE);
      classifier.classify({ image: img }, handleResults);
    });
  }
}

function draw() {
  background(220);
  
  if (currentImage) {
    image(currentImage, width/2 - currentImage.width/2, 60);
  }
  
  textAlign(CENTER, CENTER);
  textSize(32);
  text(label, width/2, 40);
}

function handleResults(error, result) {
  if (error) {
    console.error(error);
    return;
  }
  
  console.log('Predicción:', result);
  
  const predictions = [
    { label: 'cuchara', confidence: result.cuchara || 0 },
    { label: 'cuchillo', confidence: result.cuchillo || 0 },
    { label: 'tenedor', confidence: result.tenedor || 0 }
  ];
  
  predictions.sort((a, b) => b.confidence - a.confidence);
  
  const topPrediction = predictions[0];
  let confidence = floor(topPrediction.confidence * 100);
  
  if (confidence < 50) {
    label = `${topPrediction.label} (${confidence}% confianza - Baja certeza)`;
  } else {
    label = `${topPrediction.label} (${confidence}% confianza)`;
  }
  
  console.log('Predicciones ordenadas:');
  predictions.forEach(pred => {
    console.log(`${pred.label}: ${floor(pred.confidence * 100)}%`);
  });
}
