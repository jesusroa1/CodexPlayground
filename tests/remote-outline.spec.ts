import { expect, test } from '@playwright/test';

test('renders a red outline around the remote sample', async ({ page }) => {
  await page.goto('/index.html');

  await page.waitForFunction(
    'typeof remoteProcessed !== "undefined" && remoteProcessed === true',
    { timeout: 60_000 }
  );

  await page.waitForTimeout(500);

  const metrics = await page.evaluate(() => {
    const canvas = document.getElementById('remote-output-canvas') as HTMLCanvasElement | null;
    const ctx = canvas?.getContext('2d');

    if (!canvas || !ctx) {
      throw new Error('Remote output canvas is not available');
    }

    const { width, height } = canvas;
    const pixels = ctx.getImageData(0, 0, width, height).data;

    let redCount = 0;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    let maxHorizontalRun = 0;
    let maxVerticalRun = 0;
    const firstRedByRow = new Array(height).fill(-1);
    const lastRedByRow = new Array(height).fill(-1);
    const firstRedByCol = new Array(width).fill(-1);
    const lastRedByCol = new Array(width).fill(-1);

    for (let y = 0; y < height; y++) {
      let run = 0;
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const r = pixels[idx];
        const g = pixels[idx + 1];
        const b = pixels[idx + 2];
        const isRed = r > 200 && g < 60 && b < 60;

        if (isRed) {
          redCount += 1;
          run += 1;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
          if (firstRedByRow[y] === -1) firstRedByRow[y] = x;
          lastRedByRow[y] = x;
          if (firstRedByCol[x] === -1) firstRedByCol[x] = y;
          lastRedByCol[x] = y;
        } else {
          if (run > maxHorizontalRun) {
            maxHorizontalRun = run;
          }
          run = 0;
        }
      }
      if (run > maxHorizontalRun) {
        maxHorizontalRun = run;
      }
    }

    for (let x = 0; x < width; x++) {
      let run = 0;
      for (let y = 0; y < height; y++) {
        const idx = (y * width + x) * 4;
        const r = pixels[idx];
        const g = pixels[idx + 1];
        const b = pixels[idx + 2];
        const isRed = r > 200 && g < 60 && b < 60;

        if (isRed) {
          run += 1;
        } else {
          if (run > maxVerticalRun) {
            maxVerticalRun = run;
          }
          run = 0;
        }
      }
      if (run > maxVerticalRun) {
        maxVerticalRun = run;
      }
    }

    const boundingWidth = maxX >= minX ? maxX - minX + 1 : 0;
    const boundingHeight = maxY >= minY ? maxY - minY + 1 : 0;
    const boundingArea = boundingWidth * boundingHeight;
    const canvasArea = width * height;
    const boundingAreaRatio = canvasArea > 0 ? boundingArea / canvasArea : 0;
    const coverageRatio = boundingArea > 0 ? redCount / boundingArea : 0;

    let rowsWithOutline = 0;
    let colsWithOutline = 0;

    if (redCount > 0) {
      for (let y = minY; y <= maxY; y++) {
        if (firstRedByRow[y] !== -1 && lastRedByRow[y] !== -1) {
          rowsWithOutline += 1;
        }
      }

      for (let x = minX; x <= maxX; x++) {
        if (firstRedByCol[x] !== -1 && lastRedByCol[x] !== -1) {
          colsWithOutline += 1;
        }
      }
    }

    return {
      width,
      height,
      redCount,
      boundingWidth,
      boundingHeight,
      boundingArea,
      boundingAreaRatio,
      coverageRatio,
      minX,
      maxX,
      minY,
      maxY,
      rowsWithOutline,
      colsWithOutline,
      maxHorizontalRun,
      maxVerticalRun,
    };
  });

  expect(metrics.width).toBeGreaterThan(0);
  expect(metrics.height).toBeGreaterThan(0);

  expect(metrics.redCount).toBeGreaterThan(300);

  expect(metrics.boundingWidth).toBeGreaterThan(20);
  expect(metrics.boundingHeight).toBeGreaterThan(70);

  expect(metrics.boundingAreaRatio).toBeGreaterThan(0.02);
  expect(metrics.boundingAreaRatio).toBeLessThan(0.12);

  expect(metrics.coverageRatio).toBeGreaterThan(0.1);
  expect(metrics.coverageRatio).toBeLessThan(0.45);

  if (metrics.boundingHeight > 0) {
    expect(metrics.rowsWithOutline).toBeGreaterThanOrEqual(Math.floor(metrics.boundingHeight * 0.85));
  }

  if (metrics.boundingWidth > 0) {
    expect(metrics.colsWithOutline).toBeGreaterThanOrEqual(Math.floor(metrics.boundingWidth * 0.85));
  }

  expect(metrics.minX).toBeGreaterThanOrEqual(50);
  expect(metrics.maxX).toBeLessThanOrEqual(metrics.width - 50);
  expect(metrics.minY).toBeGreaterThanOrEqual(40);
  expect(metrics.maxY).toBeLessThanOrEqual(metrics.height - 40);

  expect(metrics.maxHorizontalRun).toBeLessThanOrEqual(metrics.boundingWidth * 3);
  expect(metrics.maxVerticalRun).toBeLessThanOrEqual(metrics.boundingHeight * 3);
});
