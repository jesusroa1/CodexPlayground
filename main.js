let cvRuntimeReady = false;
let imageReady = false;
let alreadyProcessed = false;
let coasterImageReady = false;
let coasterProcessed = false;
let remoteImageReady = false;
let remoteProcessed = false;

const originalImg = document.getElementById('original-img');
const coasterImg = document.getElementById('coaster-img');
const remoteImg = document.getElementById('remote-img');

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

function tryProcessRemote() {
  if (remoteProcessed || !remoteImageReady) {
    return;
  }

  if (typeof cv !== 'undefined' && !cvRuntimeReady) {
    return;
  }

  remoteProcessed = true;
  window.requestAnimationFrame(processRemoteImage);
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

    const frameArea = working.rows * working.cols;

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

function createRemoteFallbackContext(imageElement) {
  const width = imageElement.naturalWidth;
  const height = imageElement.naturalHeight;
  if (!width || !height) {
    return null;
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) {
    return null;
  }

  context.drawImage(imageElement, 0, 0, width, height);
  return { canvas, context, width, height };
}

function toGrayscalePixels(imageData, width, height) {
  const grayscale = new Float32Array(width * height);
  for (let i = 0, j = 0; i < imageData.length; i += 4, j++) {
    const r = imageData[i];
    const g = imageData[i + 1];
    const b = imageData[i + 2];
    grayscale[j] = r * 0.2126 + g * 0.7152 + b * 0.0722;
  }
  return grayscale;
}

function gaussianBlurSeparable(source, width, height, kernel) {
  const radius = Math.floor(kernel.length / 2);
  const temp = new Float32Array(width * height);
  const output = new Float32Array(width * height);
  const kernelSum = kernel.reduce((sum, value) => sum + value, 0);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let accum = 0;
      for (let k = -radius; k <= radius; k++) {
        const sampleX = Math.min(width - 1, Math.max(0, x + k));
        accum += source[y * width + sampleX] * kernel[k + radius];
      }
      temp[y * width + x] = accum / kernelSum;
    }
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let accum = 0;
      for (let k = -radius; k <= radius; k++) {
        const sampleY = Math.min(height - 1, Math.max(0, y + k));
        accum += temp[sampleY * width + x] * kernel[k + radius];
      }
      output[y * width + x] = accum / kernelSum;
    }
  }

  return output;
}

function otsuThreshold(grayscale) {
  const histogram = new Array(256).fill(0);
  const total = grayscale.length;

  for (let i = 0; i < total; i++) {
    const value = Math.max(0, Math.min(255, Math.round(grayscale[i])));
    histogram[value]++;
  }

  let sum = 0;
  for (let i = 0; i < 256; i++) {
    sum += i * histogram[i];
  }

  let sumB = 0;
  let weightB = 0;
  let weightF = 0;
  let maxVariance = -1;
  let threshold = 0;

  for (let i = 0; i < 256; i++) {
    weightB += histogram[i];
    if (weightB === 0) {
      continue;
    }
    weightF = total - weightB;
    if (weightF === 0) {
      break;
    }

    sumB += i * histogram[i];
    const meanB = sumB / weightB;
    const meanF = (sum - sumB) / weightF;
    const between = weightB * weightF * Math.pow(meanB - meanF, 2);

    if (between > maxVariance) {
      maxVariance = between;
      threshold = i;
    }
  }

  return threshold;
}

function applyMorphology(mask, width, height, radius, type) {
  const output = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let result = type === 'erode' ? 1 : 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const sampleY = y + dy;
        if (sampleY < 0 || sampleY >= height) {
          if (type === 'erode') {
            result = 0;
          }
          continue;
        }
        let earlyExit = false;
        for (let dx = -radius; dx <= radius; dx++) {
          const sampleX = x + dx;
          if (sampleX < 0 || sampleX >= width) {
            if (type === 'erode') {
              result = 0;
              earlyExit = true;
              break;
            }
            continue;
          }
          const value = mask[sampleY * width + sampleX];
          if (type === 'dilate') {
            if (value) {
              result = 1;
              earlyExit = true;
              break;
            }
          } else if (type === 'erode' && !value) {
            result = 0;
            earlyExit = true;
            break;
          }
        }
        if (earlyExit) {
          break;
        }
      }
      output[y * width + x] = result;
    }
  }
  return output;
}

function labelBestRemoteComponent(mask, width, height) {
  const labels = new Int32Array(width * height);
  const queueX = new Int32Array(width * height);
  const queueY = new Int32Array(width * height);
  let currentLabel = 1;
  let bestLabel = 0;
  let bestArea = 0;
  let bestScore = -Infinity;
  let fallbackLabel = 0;
  let fallbackArea = 0;

  const frameArea = width * height;
  const centerX = width / 2;
  const centerY = height / 2;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      if (!mask[index] || labels[index] !== 0) {
        continue;
      }

      let area = 0;
      let head = 0;
      let tail = 0;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let sumX = 0;
      let sumY = 0;

      queueX[tail] = x;
      queueY[tail] = y;
      tail++;
      labels[index] = currentLabel;

      while (head < tail) {
        const cx = queueX[head];
        const cy = queueY[head];
        head++;
        area++;

        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;
        sumX += cx;
        sumY += cy;

        const neighbors = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1]
        ];

        for (let i = 0; i < neighbors.length; i++) {
          const nx = neighbors[i][0];
          const ny = neighbors[i][1];
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            continue;
          }
          const nIndex = ny * width + nx;
          if (mask[nIndex] && labels[nIndex] === 0) {
            labels[nIndex] = currentLabel;
            queueX[tail] = nx;
            queueY[tail] = ny;
            tail++;
          }
        }
      }

      if (area > fallbackArea) {
        fallbackArea = area;
        fallbackLabel = currentLabel;
      }

      const componentWidth = maxX - minX + 1;
      const componentHeight = maxY - minY + 1;
      if (componentWidth <= 0 || componentHeight <= 0) {
        currentLabel++;
        continue;
      }

      const aspectRatio = componentWidth / componentHeight;
      const areaFraction = area / frameArea;
      const extent = area / (componentWidth * componentHeight);
      const centroidX = sumX / area;
      const centroidY = sumY / area;
      const normDx = (centroidX - centerX) / width;
      const normDy = (centroidY - centerY) / height;
      const centerPenalty = Math.hypot(normDx, normDy);

      if (areaFraction < 0.008 || areaFraction > 0.22) {
        currentLabel++;
        continue;
      }
      if (aspectRatio < 0.18 || aspectRatio > 0.62) {
        currentLabel++;
        continue;
      }
      if (extent < 0.35) {
        currentLabel++;
        continue;
      }

      const aspectTarget = 0.32;
      const extentTarget = 0.62;
      const areaTarget = 0.055;
      const aspectPenalty = Math.abs(Math.log((aspectRatio + Number.EPSILON) / aspectTarget));
      const extentPenalty = Math.abs(extent - extentTarget);
      const areaPenalty = Math.abs(areaFraction - areaTarget);
      const score =
        area -
        frameArea *
          (0.006 * aspectPenalty +
            0.005 * extentPenalty +
            0.004 * areaPenalty +
            0.003 * centerPenalty);

      if (score > bestScore) {
        bestScore = score;
        bestLabel = currentLabel;
        bestArea = area;
      }

      currentLabel++;
    }
  }

  if (!bestLabel && fallbackLabel) {
    bestLabel = fallbackLabel;
    bestArea = fallbackArea;
  }

  return { labels, bestLabel, bestArea };
}

function collectBoundaryPoints(labels, label, width, height) {
  const points = [];
  if (!label) {
    return points;
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      if (labels[index] !== label) {
        continue;
      }

      let isBoundary = false;
      for (let dy = -1; dy <= 1 && !isBoundary; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) {
            continue;
          }
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            isBoundary = true;
            break;
          }
          if (labels[ny * width + nx] !== label) {
            isBoundary = true;
            break;
          }
        }
      }

      if (isBoundary) {
        points.push([x, y]);
      }
    }
  }

  return points;
}

function computeConvexHull(points) {
  if (points.length < 3) {
    return points.slice();
  }

  const sorted = points
    .slice()
    .sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]));

  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const lower = [];
  for (let i = 0; i < sorted.length; i++) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], sorted[i]) <= 0) {
      lower.pop();
    }
    lower.push(sorted[i]);
  }

  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], sorted[i]) <= 0) {
      upper.pop();
    }
    upper.push(sorted[i]);
  }

  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function drawHullOnCanvas(outputCanvas, imageElement, hullPoints) {
  const context = outputCanvas.getContext('2d');
  if (!context) {
    return;
  }

  const width = imageElement.naturalWidth;
  const height = imageElement.naturalHeight;
  outputCanvas.width = width;
  outputCanvas.height = height;
  context.drawImage(imageElement, 0, 0, width, height);

  if (!hullPoints || hullPoints.length === 0) {
    console.warn('Fallback remote segmentation did not locate a contour to draw.');
    return;
  }

  context.save();
  context.strokeStyle = '#ff0000';
  context.lineWidth = Math.max(2, Math.round(Math.min(width, height) * 0.01));
  context.beginPath();
  context.moveTo(hullPoints[0][0], hullPoints[0][1]);
  for (let i = 1; i < hullPoints.length; i++) {
    context.lineTo(hullPoints[i][0], hullPoints[i][1]);
  }
  context.closePath();
  context.stroke();
  context.restore();
}

function processRemoteImageFallback(imageElement) {
  const outputCanvas = document.getElementById('remote-output-canvas');
  if (!outputCanvas) {
    return;
  }

  const contextBundle = createRemoteFallbackContext(imageElement);
  if (!contextBundle) {
    console.warn('Unable to create fallback canvas context for remote processing.');
    return;
  }

  const { context, width, height } = contextBundle;
  const imageData = context.getImageData(0, 0, width, height);
  const grayscale = toGrayscalePixels(imageData.data, width, height);
  const blurred = gaussianBlurSeparable(grayscale, width, height, [1, 4, 6, 4, 1]);
  const threshold = otsuThreshold(blurred);

  let mask = new Uint8Array(width * height);
  for (let i = 0; i < mask.length; i++) {
    mask[i] = blurred[i] <= threshold ? 1 : 0;
  }

  mask = applyMorphology(mask, width, height, 1, 'erode');
  mask = applyMorphology(mask, width, height, 1, 'dilate');
  mask = applyMorphology(mask, width, height, 3, 'dilate');
  mask = applyMorphology(mask, width, height, 3, 'erode');
  mask = applyMorphology(mask, width, height, 1, 'erode');

  const { labels, bestLabel, bestArea } = labelBestRemoteComponent(mask, width, height);
  if (!bestArea) {
    console.warn('Fallback remote segmentation could not find a dominant component.');
    drawHullOnCanvas(document.getElementById('remote-output-canvas'), imageElement, []);
    return;
  }

  const boundaryPoints = collectBoundaryPoints(labels, bestLabel, width, height);
  const hull = computeConvexHull(boundaryPoints);
  drawHullOnCanvas(document.getElementById('remote-output-canvas'), imageElement, hull);
}

function processRemoteImage() {
  const imageElement = document.getElementById('remote-img');
  if (!imageElement) {
    return;
  }

  if (typeof cv === 'undefined') {
    processRemoteImageFallback(imageElement);
    return;
  }

  const matsToRelease = [];

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

    const lab = new cv.Mat();
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    matsToRelease.push(lab);

    const labChannels = new cv.MatVector();
    cv.split(lab, labChannels);
    const lightness = labChannels.get(0);
    matsToRelease.push(lightness);
    const aChannel = labChannels.get(1);
    const bChannel = labChannels.get(2);
    aChannel.delete();
    bChannel.delete();
    labChannels.delete();

    const blurred = new cv.Mat();
    cv.GaussianBlur(lightness, blurred, new cv.Size(7, 7), 0);
    matsToRelease.push(blurred);

    const remoteMask = new cv.Mat();
    cv.threshold(blurred, remoteMask, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    matsToRelease.push(remoteMask);

    const openKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
    matsToRelease.push(openKernel);
    cv.morphologyEx(remoteMask, remoteMask, cv.MORPH_OPEN, openKernel);

    const closeKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(19, 19));
    matsToRelease.push(closeKernel);
    cv.morphologyEx(remoteMask, remoteMask, cv.MORPH_CLOSE, closeKernel);

    const refineKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(9, 9));
    matsToRelease.push(refineKernel);
    cv.erode(remoteMask, remoteMask, refineKernel);

    const labels = new cv.Mat();
    const stats = new cv.Mat();
    const centroids = new cv.Mat();
    const connectivity = 8;
    const componentCount = cv.connectedComponentsWithStats(
      remoteMask,
      labels,
      stats,
      centroids,
      connectivity,
      cv.CV_32S
    );
    matsToRelease.push(labels, stats, centroids);

    const statsData = stats.data32S;
    const statsCols = stats.cols;
    const centroidsData = centroids.data64F;
    const centerX = working.cols / 2;
    const centerY = working.rows / 2;
    let bestLabel = -1;
    let bestScore = -Infinity;

    for (let label = 1; label < componentCount; label++) {
      const baseIndex = label * statsCols;
      const area = statsData[baseIndex + cv.CC_STAT_AREA];
      const width = statsData[baseIndex + cv.CC_STAT_WIDTH];
      const height = statsData[baseIndex + cv.CC_STAT_HEIGHT];
      if (width <= 0 || height <= 0) {
        continue;
      }

      const areaFraction = area / frameArea;
      const aspectRatio = width / height;
      const extent = area / (width * height);
      const centroidX = centroidsData[label * 2];
      const centroidY = centroidsData[label * 2 + 1];
      const normDx = (centroidX - centerX) / working.cols;
      const normDy = (centroidY - centerY) / working.rows;
      const centerPenalty = Math.hypot(normDx, normDy);

      if (areaFraction < 0.008 || areaFraction > 0.22) {
        continue;
      }
      if (aspectRatio < 0.18 || aspectRatio > 0.62) {
        continue;
      }
      if (extent < 0.35) {
        continue;
      }

      const aspectTarget = 0.32;
      const extentTarget = 0.62;
      const areaTarget = 0.055;
      const aspectPenalty = Math.abs(Math.log((aspectRatio + Number.EPSILON) / aspectTarget));
      const extentPenalty = Math.abs(extent - extentTarget);
      const areaPenalty = Math.abs(areaFraction - areaTarget);
      const score =
        area -
        frameArea *
          (0.006 * aspectPenalty +
            0.005 * extentPenalty +
            0.004 * areaPenalty +
            0.003 * centerPenalty);

      if (score > bestScore) {
        bestScore = score;
        bestLabel = label;
      }
    }

    let hullDrawn = false;
    const thickness = Math.max(3, Math.round(Math.min(working.rows, working.cols) * 0.01));

    if (bestLabel !== -1) {
      const labelMask = new cv.Mat.zeros(labels.rows, labels.cols, cv.CV_8UC1);
      const maskData = labelMask.data;
      const labelData = labels.data32S;
      for (let i = 0; i < labelData.length; i++) {
        if (labelData[i] === bestLabel) {
          maskData[i] = 255;
        }
      }

      const componentContours = new cv.MatVector();
      const componentHierarchy = new cv.Mat();
      cv.findContours(labelMask, componentContours, componentHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      if (componentContours.size() > 0) {
        const bestContour = componentContours.get(0);
        const hull = new cv.Mat();
        cv.convexHull(bestContour, hull, false, true);
        const hullVector = new cv.MatVector();
        hullVector.push_back(hull);
        const red = new cv.Scalar(255, 0, 0, 255);
        const lineType = typeof cv.LINE_AA !== 'undefined' ? cv.LINE_AA : 8;
        cv.drawContours(working, hullVector, 0, red, thickness, lineType);
        hullVector.delete();
        hull.delete();
        bestContour.delete();
        hullDrawn = true;
      }
      componentHierarchy.delete();
      componentContours.delete();
      labelMask.delete();
    }

    if (!hullDrawn) {
      const fallbackContours = new cv.MatVector();
      const fallbackHierarchy = new cv.Mat();
      cv.findContours(remoteMask, fallbackContours, fallbackHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      const aspectTarget = 0.32;
      const extentTarget = 0.62;
      const areaTarget = 0.055;
      let bestIndex = -1;
      let fallbackScore = -Infinity;

      for (let i = 0; i < fallbackContours.size(); i++) {
        const contour = fallbackContours.get(i);
        const area = cv.contourArea(contour);
        if (area <= 0) {
          contour.delete();
          continue;
        }

        const rect = cv.boundingRect(contour);
        if (rect.width <= 0 || rect.height <= 0) {
          contour.delete();
          continue;
        }

        const areaFraction = area / frameArea;
        const aspectRatio = rect.width / rect.height;
        const extent = area / (rect.width * rect.height);

        if (areaFraction < 0.008 || areaFraction > 0.22) {
          contour.delete();
          continue;
        }
        if (aspectRatio < 0.18 || aspectRatio > 0.62) {
          contour.delete();
          continue;
        }
        if (extent < 0.35) {
          contour.delete();
          continue;
        }

        const aspectPenalty = Math.abs(Math.log((aspectRatio + Number.EPSILON) / aspectTarget));
        const extentPenalty = Math.abs(extent - extentTarget);
        const areaPenalty = Math.abs(areaFraction - areaTarget);
        const score =
          area -
          frameArea * (0.006 * aspectPenalty + 0.005 * extentPenalty + 0.004 * areaPenalty);

        if (score > fallbackScore) {
          fallbackScore = score;
          bestIndex = i;
        }

        contour.delete();
      }

      if (bestIndex !== -1) {
        const bestContour = fallbackContours.get(bestIndex);
        const hull = new cv.Mat();
        cv.convexHull(bestContour, hull, false, true);
        const hullVector = new cv.MatVector();
        hullVector.push_back(hull);
        const red = new cv.Scalar(255, 0, 0, 255);
        const lineType = typeof cv.LINE_AA !== 'undefined' ? cv.LINE_AA : 8;
        cv.drawContours(working, hullVector, 0, red, thickness, lineType);
        hullVector.delete();
        hull.delete();
        bestContour.delete();
        hullDrawn = true;
      }

      if (!hullDrawn) {
        console.warn('No contour detected for the remote outline.');
      }

      fallbackHierarchy.delete();
      fallbackContours.delete();
    }

    cv.imshow('remote-output-canvas', working);
  } catch (error) {
    console.error('OpenCV remote processing failed:', error);
  } finally {
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

if (remoteImg) {
  if (remoteImg.complete && remoteImg.naturalWidth !== 0) {
    remoteImageReady = true;
  } else {
    remoteImg.addEventListener('load', () => {
      remoteImageReady = true;
      tryProcessRemote();
    }, { once: true });
    remoteImg.addEventListener('error', () => {
      console.error('Failed to load the remote input image.');
    }, { once: true });
  }
} else {
  console.warn('Remote image element was not found.');
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
    tryProcessRemote();
    return true;
  }

  cv.onRuntimeInitialized = () => {
    cvRuntimeReady = true;
    tryProcessMouse();
    tryProcessCoaster();
    tryProcessRemote();
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

if (remoteImageReady) {
  tryProcessRemote();
}
