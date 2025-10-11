let cvRuntimeReady = false;
let imageReady = false;
let alreadyProcessed = false;
let coasterImageReady = false;
let coasterProcessed = false;

const originalImg = document.getElementById('original-img');
const coasterImg = document.getElementById('coaster-img');

function tryProcessMouse() {
  if (alreadyProcessed || !cvRuntimeReady || !imageReady) {
    return;
  }
  alreadyProcessed = true;
  // Use rAF so the canvas has been laid out before drawing.
  window.requestAnimationFrame(processMouseImage);
}

function tryProcessCoaster() {
  if (coasterProcessed || !cvRuntimeReady || !coasterImageReady) {
    return;
  }
  coasterProcessed = true;
  window.requestAnimationFrame(processCoasterImage);
}

function processMouseImage() {
  const imageElement = document.getElementById('original-img');
  if (!imageElement || typeof cv === 'undefined') {
    return;
  }

  const matsToRelease = [];
  let kernel;
  let secondaryKernel;
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

    const rgb = new cv.Mat();
    cv.cvtColor(working, rgb, cv.COLOR_RGBA2RGB);
    matsToRelease.push(rgb);

    const grabCutMask = new cv.Mat();
    grabCutMask.create(working.rows, working.cols, cv.CV_8UC1);
    grabCutMask.setTo(new cv.Scalar(cv.GC_PR_BGD));
    matsToRelease.push(grabCutMask);

    const bgdModel = new cv.Mat();
    bgdModel.create(1, 65, cv.CV_64FC1);
    bgdModel.setTo(new cv.Scalar(0));
    matsToRelease.push(bgdModel);

    const fgdModel = new cv.Mat();
    fgdModel.create(1, 65, cv.CV_64FC1);
    fgdModel.setTo(new cv.Scalar(0));
    matsToRelease.push(fgdModel);

    const gray = new cv.Mat();
    cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
    matsToRelease.push(gray);

    const grayBlurred = new cv.Mat();
    cv.GaussianBlur(gray, grayBlurred, new cv.Size(5, 5), 0);
    matsToRelease.push(grayBlurred);

    const darkMask = new cv.Mat();
    cv.threshold(grayBlurred, darkMask, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    matsToRelease.push(darkMask);

    const centralMask = cv.Mat.zeros(darkMask.rows, darkMask.cols, cv.CV_8UC1);
    matsToRelease.push(centralMask);
    const centralMargin = Math.round(Math.min(working.rows, working.cols) * 0.12);
    const centralX = Math.min(Math.max(0, centralMargin), Math.max(0, darkMask.cols - 2));
    const centralY = Math.min(Math.max(0, centralMargin), Math.max(0, darkMask.rows - 2));
    let centralWidth = Math.max(1, darkMask.cols - centralX * 2);
    let centralHeight = Math.max(1, darkMask.rows - centralY * 2);
    centralWidth = Math.min(centralWidth, darkMask.cols - centralX);
    centralHeight = Math.min(centralHeight, darkMask.rows - centralY);
    cv.rectangle(
      centralMask,
      new cv.Point(centralX, centralY),
      new cv.Point(centralX + centralWidth - 1, centralY + centralHeight - 1),
      new cv.Scalar(255),
      cv.FILLED
    );
    cv.bitwise_and(darkMask, centralMask, darkMask);

    const seedContours = new cv.MatVector();
    const seedHierarchy = new cv.Mat();
    cv.findContours(darkMask, seedContours, seedHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const sureForegroundMask = cv.Mat.zeros(darkMask.rows, darkMask.cols, cv.CV_8UC1);
    const probableForegroundMask = cv.Mat.zeros(darkMask.rows, darkMask.cols, cv.CV_8UC1);
    matsToRelease.push(sureForegroundMask, probableForegroundMask);

    let seedBestIndex = -1;
    let seedBestArea = 0;
    for (let i = 0; i < seedContours.size(); i++) {
      const contour = seedContours.get(i);
      const area = cv.contourArea(contour);
      if (area > seedBestArea) {
        seedBestArea = area;
        seedBestIndex = i;
      }
    }

    let useMaskForGrabCut = false;
    if (seedBestIndex !== -1) {
      useMaskForGrabCut = true;
      cv.drawContours(sureForegroundMask, seedContours, seedBestIndex, new cv.Scalar(255), cv.FILLED);
      const refineKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
      matsToRelease.push(refineKernel);
      cv.morphologyEx(sureForegroundMask, sureForegroundMask, cv.MORPH_CLOSE, refineKernel);
      cv.erode(sureForegroundMask, sureForegroundMask, refineKernel);

      const probableKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(19, 19));
      matsToRelease.push(probableKernel);
      cv.dilate(sureForegroundMask, probableForegroundMask, probableKernel);
      cv.bitwise_or(probableForegroundMask, sureForegroundMask, probableForegroundMask);
    } else {
      cv.rectangle(
        probableForegroundMask,
        new cv.Point(centralX, centralY),
        new cv.Point(centralX + centralWidth - 1, centralY + centralHeight - 1),
        new cv.Scalar(255),
        cv.FILLED
      );
    }

    for (let i = 0; i < seedContours.size(); i++) {
      const contour = seedContours.get(i);
      contour.delete();
    }
    seedContours.delete();
    seedHierarchy.delete();

    const borderMask = cv.Mat.zeros(grabCutMask.rows, grabCutMask.cols, cv.CV_8UC1);
    matsToRelease.push(borderMask);
    const borderBase = Math.max(3, Math.round(Math.min(working.rows, working.cols) * 0.05));
    const borderX = Math.max(1, Math.min(borderBase, Math.floor(grabCutMask.cols / 2)));
    const borderY = Math.max(1, Math.min(borderBase, Math.floor(grabCutMask.rows / 2)));
    cv.rectangle(borderMask, new cv.Point(0, 0), new cv.Point(borderMask.cols - 1, borderY - 1), new cv.Scalar(255), cv.FILLED);
    cv.rectangle(
      borderMask,
      new cv.Point(0, borderMask.rows - borderY),
      new cv.Point(borderMask.cols - 1, borderMask.rows - 1),
      new cv.Scalar(255),
      cv.FILLED
    );
    cv.rectangle(borderMask, new cv.Point(0, 0), new cv.Point(borderX - 1, borderMask.rows - 1), new cv.Scalar(255), cv.FILLED);
    cv.rectangle(
      borderMask,
      new cv.Point(borderMask.cols - borderX, 0),
      new cv.Point(borderMask.cols - 1, borderMask.rows - 1),
      new cv.Scalar(255),
      cv.FILLED
    );
    grabCutMask.setTo(new cv.Scalar(cv.GC_BGD), borderMask);

    const probableForegroundCount = cv.countNonZero(probableForegroundMask);
    if (probableForegroundCount > 0) {
      grabCutMask.setTo(new cv.Scalar(cv.GC_PR_FGD), probableForegroundMask);
    }

    const sureForegroundCount = cv.countNonZero(sureForegroundMask);
    if (useMaskForGrabCut && sureForegroundCount > 0) {
      grabCutMask.setTo(new cv.Scalar(cv.GC_FGD), sureForegroundMask);
    }

    const padding = Math.round(Math.min(working.rows, working.cols) * 0.08);
    const rectX = Math.min(Math.max(0, padding), Math.max(0, working.cols - 2));
    const rectY = Math.min(Math.max(0, padding), Math.max(0, working.rows - 2));
    let rectWidth = Math.max(1, working.cols - rectX * 2);
    let rectHeight = Math.max(1, working.rows - rectY * 2);
    rectWidth = Math.min(rectWidth, working.cols - rectX);
    rectHeight = Math.min(rectHeight, working.rows - rectY);
    const rect = new cv.Rect(rectX, rectY, rectWidth, rectHeight);

    if (typeof cv.grabCut !== 'function') {
      throw new Error('cv.grabCut is unavailable in this build of OpenCV.js');
    }

    const grabCutMode = useMaskForGrabCut && sureForegroundCount > 0
      ? cv.GC_INIT_WITH_MASK
      : cv.GC_INIT_WITH_RECT;

    cv.grabCut(rgb, grabCutMask, rect, bgdModel, fgdModel, 5, grabCutMode);

    const gcForeground = new cv.Mat();
    const gcProbableForeground = new cv.Mat();
    const gcForegroundValue = new cv.Mat(grabCutMask.rows, grabCutMask.cols, grabCutMask.type(), new cv.Scalar(cv.GC_FGD));
    const gcProbableForegroundValue = new cv.Mat(grabCutMask.rows, grabCutMask.cols, grabCutMask.type(), new cv.Scalar(cv.GC_PR_FGD));
    matsToRelease.push(gcForegroundValue, gcProbableForegroundValue);

    cv.compare(grabCutMask, gcForegroundValue, gcForeground, cv.CMP_EQ);
    cv.compare(grabCutMask, gcProbableForegroundValue, gcProbableForeground, cv.CMP_EQ);
    matsToRelease.push(gcForeground, gcProbableForeground);

    const combinedMask = new cv.Mat();
    cv.bitwise_or(gcForeground, gcProbableForeground, combinedMask);
    cv.bitwise_and(combinedMask, darkMask, combinedMask);
    matsToRelease.push(combinedMask);

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(13, 13));
    cv.morphologyEx(combinedMask, combinedMask, cv.MORPH_CLOSE, kernel);

    secondaryKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(7, 7));
    cv.morphologyEx(combinedMask, combinedMask, cv.MORPH_OPEN, secondaryKernel);

    const blurredMask = new cv.Mat();
    cv.GaussianBlur(combinedMask, blurredMask, new cv.Size(5, 5), 0);
    matsToRelease.push(blurredMask);

    const binaryMask = new cv.Mat();
    cv.threshold(blurredMask, binaryMask, 128, 255, cv.THRESH_BINARY);
    matsToRelease.push(binaryMask);

    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(binaryMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const minArea = frameArea * 0.002;
    const maxArea = frameArea * 0.2;
    let bestIndex = -1;
    let bestScore = Infinity;
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);

      if (area < minArea || area > maxArea) {
        continue;
      }

      const contourMask = cv.Mat.zeros(binaryMask.rows, binaryMask.cols, cv.CV_8UC1);
      cv.drawContours(contourMask, contours, i, new cv.Scalar(255), cv.FILLED);
      const meanScalar = cv.mean(gray, contourMask);
      const meanIntensity = Array.isArray(meanScalar) ? meanScalar[0] : meanScalar;
      contourMask.delete();

      if (meanIntensity < bestScore) {
        bestScore = meanIntensity;
        bestIndex = i;
      }
    }

    if (bestIndex !== -1) {
      const red = new cv.Scalar(255, 0, 0, 255);
      const lineType = typeof cv.LINE_AA !== 'undefined' ? cv.LINE_AA : 8;
      cv.drawContours(working, contours, bestIndex, red, 3, lineType);
    } else {
      console.warn('No contour detected for the mouse outline.');
    }

    cv.imshow('output-canvas', working);
  } catch (error) {
    console.error('OpenCV processing failed:', error);
  } finally {
    if (hierarchy) hierarchy.delete();
    if (contours) {
      contours.delete();
    }
    if (kernel) kernel.delete();
    if (secondaryKernel) secondaryKernel.delete();
    matsToRelease.forEach((mat) => {
      if (mat && typeof mat.delete === 'function') {
        mat.delete();
      }
    });
  }
}

function processCoasterImage() {
  const imageElement = document.getElementById('coaster-img');
  if (!imageElement || typeof cv === 'undefined') {
    return;
  }

  const matsToRelease = [];
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

    const rgb = new cv.Mat();
    cv.cvtColor(working, rgb, cv.COLOR_RGBA2RGB);
    matsToRelease.push(rgb);

    const hsv = new cv.Mat();
    cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
    matsToRelease.push(hsv);

    const hsvChannels = new cv.MatVector();
    cv.split(hsv, hsvChannels);
    const _hue = hsvChannels.get(0);
    const saturation = hsvChannels.get(1);
    const value = hsvChannels.get(2);
    matsToRelease.push(_hue, saturation, value);
    hsvChannels.delete();

    const saturationMask = new cv.Mat();
    cv.threshold(saturation, saturationMask, 40, 255, cv.THRESH_BINARY);
    matsToRelease.push(saturationMask);

    const valueMask = new cv.Mat();
    cv.threshold(value, valueMask, 215, 255, cv.THRESH_BINARY_INV);
    matsToRelease.push(valueMask);

    const combinedMask = new cv.Mat();
    cv.bitwise_and(saturationMask, valueMask, combinedMask);
    matsToRelease.push(combinedMask);

    const centralMask = cv.Mat.zeros(combinedMask.rows, combinedMask.cols, cv.CV_8UC1);
    const centerX = Math.floor(combinedMask.cols / 2);
    const centerY = Math.floor(combinedMask.rows / 2);
    const radius = Math.floor(Math.min(combinedMask.cols, combinedMask.rows) * 0.44);
    cv.circle(centralMask, new cv.Point(centerX, centerY), radius, new cv.Scalar(255), cv.FILLED);
    cv.bitwise_and(combinedMask, centralMask, combinedMask);
    matsToRelease.push(centralMask);

    const closingKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(41, 41));
    cv.morphologyEx(combinedMask, combinedMask, cv.MORPH_CLOSE, closingKernel);
    matsToRelease.push(closingKernel);

    const openingKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(19, 19));
    cv.morphologyEx(combinedMask, combinedMask, cv.MORPH_OPEN, openingKernel);
    matsToRelease.push(openingKernel);

    cv.GaussianBlur(combinedMask, combinedMask, new cv.Size(5, 5), 0);
    cv.threshold(combinedMask, combinedMask, 128, 255, cv.THRESH_BINARY);

    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(combinedMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const frameArea = working.rows * working.cols;
    const minArea = frameArea * 0.04;
    const maxArea = frameArea * 0.35;
    let bestIndex = -1;
    let bestCircularity = -Infinity;

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);

      if (area < minArea || area > maxArea) {
        continue;
      }

      const perimeter = cv.arcLength(contour, true);
      const circularity = perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0;

      if (circularity > bestCircularity) {
        bestCircularity = circularity;
        bestIndex = i;
      }
    }

    if (bestIndex !== -1) {
      const red = new cv.Scalar(255, 0, 0, 255);
      const lineType = typeof cv.LINE_AA !== 'undefined' ? cv.LINE_AA : 8;
      cv.drawContours(working, contours, bestIndex, red, 4, lineType);
    } else {
      console.warn('No contour detected for the coaster outline.');
    }

    cv.imshow('coaster-output-canvas', working);
  } catch (error) {
    console.error('OpenCV coaster processing failed:', error);
  } finally {
    if (hierarchy) hierarchy.delete();
    if (contours) {
      contours.delete();
    }
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

if (coasterImg) {
  if (coasterImg.complete && coasterImg.naturalWidth !== 0) {
    coasterImageReady = true;
  } else {
    coasterImg.addEventListener('load', () => {
      coasterImageReady = true;
      tryProcessCoaster();
    }, { once: true });
    coasterImg.addEventListener('error', () => {
      console.error('Failed to load the coaster input image.');
    }, { once: true });
  }
} else {
  console.warn('Coaster image element was not found.');
}

function attachOpenCvListener() {
  if (typeof cv === 'undefined') {
    return false;
  }

  if (cv.Mat) {
    // Runtime already initialised.
    cvRuntimeReady = true;
    tryProcessMouse();
    tryProcessCoaster();
    return true;
  }

  cv.onRuntimeInitialized = () => {
    cvRuntimeReady = true;
    tryProcessMouse();
    tryProcessCoaster();
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

if (coasterImageReady) {
  tryProcessCoaster();
}
