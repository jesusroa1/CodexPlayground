let cvRuntimeReady = false;
let imageReady = false;
let alreadyProcessed = false;

const originalImg = document.getElementById('original-img');

function tryProcessMouse() {
  if (alreadyProcessed || !cvRuntimeReady || !imageReady) {
    return;
  }
  alreadyProcessed = true;
  // Use rAF so the canvas has been laid out before drawing.
  window.requestAnimationFrame(processMouseImage);
}

function processMouseImage() {
  const imageElement = document.getElementById('original-img');
  if (!imageElement || typeof cv === 'undefined') {
    return;
  }

  const matsToRelease = [];
  let kernel;
  let contours;
  let hierarchy;

  try {
    const original = cv.imread(imageElement);
    matsToRelease.push(original);

    let working;
    const maxWidth = 800;
    if (original.cols > maxWidth) {
      working = new cv.Mat();
      const scale = maxWidth / original.cols;
      const size = new cv.Size(Math.round(original.cols * scale), Math.round(original.rows * scale));
      cv.resize(original, working, size, 0, 0, cv.INTER_AREA);
      matsToRelease.push(working);
    } else {
      working = original.clone();
      matsToRelease.push(working);
    }

    const frameArea = working.rows * working.cols;

    const gray = new cv.Mat();
    cv.cvtColor(working, gray, cv.COLOR_RGBA2GRAY);
    matsToRelease.push(gray);

    const blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(9, 9), 0);
    matsToRelease.push(blurred);

    const binary = new cv.Mat();
    cv.threshold(blurred, binary, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    matsToRelease.push(binary);

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(7, 7));
    cv.morphologyEx(binary, binary, cv.MORPH_CLOSE, kernel);

    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const minArea = frameArea * 0.002;
    const maxArea = frameArea * 0.5;
    let candidateIndex = -1;
    let candidateArea = 0;
    let fallbackIndex = -1;
    let fallbackArea = 0;

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);

      if (area > fallbackArea) {
        fallbackArea = area;
        fallbackIndex = i;
      }

      if (area >= minArea && area <= maxArea && area > candidateArea) {
        candidateArea = area;
        candidateIndex = i;
      }

      contour.delete();
    }

    const contourIndex = candidateIndex !== -1 ? candidateIndex : fallbackIndex;
    if (contourIndex !== -1) {
      const red = new cv.Scalar(255, 0, 0, 255);
      const lineType = typeof cv.LINE_AA !== 'undefined' ? cv.LINE_AA : 8;
      cv.drawContours(working, contours, contourIndex, red, 4, lineType);
    } else {
      console.warn('No contour detected for the mouse outline.');
    }

    cv.imshow('output-canvas', working);
  } catch (error) {
    console.error('OpenCV processing failed:', error);
  } finally {
    if (hierarchy) hierarchy.delete();
    if (contours) contours.delete();
    if (kernel) kernel.delete();
    matsToRelease.forEach((mat) => {
      if (mat && typeof mat.delete === 'function') {
        mat.delete();
      }
    });
  }
}

if (originalImg) {
  if (originalImg.complete && originalImg.naturalWidth !== 0) {
    imageReady = true;
  } else {
    originalImg.addEventListener('load', () => {
      imageReady = true;
      tryProcessMouse();
    }, { once: true });
    originalImg.addEventListener('error', () => {
      console.error('Failed to load the input image.');
    }, { once: true });
  }
} else {
  console.warn('Original image element was not found.');
}

function attachOpenCvListener() {
  if (typeof cv === 'undefined') {
    return false;
  }

  if (cv.Mat) {
    // Runtime already initialised.
    cvRuntimeReady = true;
    tryProcessMouse();
    return true;
  }

  cv.onRuntimeInitialized = () => {
    cvRuntimeReady = true;
    tryProcessMouse();
  };
  return true;
}

if (!attachOpenCvListener()) {
  const start = performance.now();
  const timeoutMs = 7000;
  const poll = setInterval(() => {
    if (attachOpenCvListener()) {
      clearInterval(poll);
      return;
    }
    if (performance.now() - start > timeoutMs) {
      clearInterval(poll);
      console.error('OpenCV.js failed to load within the expected time.');
    }
  }, 100);
}

if (imageReady) {
  tryProcessMouse();
}
