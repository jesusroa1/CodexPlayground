import { expect, test } from '@playwright/test';

test('renders a tight red outline around the coaster sample', async ({ page }) => {
  await page.goto('/index.html');

  await page.waitForFunction(
    'typeof coasterProcessed !== "undefined" && coasterProcessed === true',
    { timeout: 60_000 }
  );

  await page.waitForTimeout(500);

  const metrics = await page.evaluate(() => {
    const canvas = document.getElementById('coaster-output-canvas') as HTMLCanvasElement;
    const ctx = canvas?.getContext('2d');

    if (!canvas || !ctx) {
      throw new Error('Coaster output canvas is not available');
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
    let maxRowLeadingGap = 0;
    let maxRowTrailingGap = 0;
    let maxColLeadingGap = 0;
    let maxColTrailingGap = 0;

    if (redCount > 0) {
      for (let y = minY; y <= maxY; y++) {
        const first = firstRedByRow[y];
        const last = lastRedByRow[y];
        if (first !== -1 && last !== -1) {
          rowsWithOutline += 1;
          if (first - minX > maxRowLeadingGap) {
            maxRowLeadingGap = first - minX;
          }
          if (maxX - last > maxRowTrailingGap) {
            maxRowTrailingGap = maxX - last;
          }
        }
      }

      for (let x = minX; x <= maxX; x++) {
        const first = firstRedByCol[x];
        const last = lastRedByCol[x];
        if (first !== -1 && last !== -1) {
          colsWithOutline += 1;
          if (first - minY > maxColLeadingGap) {
            maxColLeadingGap = first - minY;
          }
          if (maxY - last > maxColTrailingGap) {
            maxColTrailingGap = maxY - last;
          }
        }
      }
    }

    return {
      width,
      height,
      redCount,
      minX,
      minY,
      maxX,
      maxY,
      boundingWidth,
      boundingHeight,
      boundingArea,
      boundingAreaRatio,
      coverageRatio,
      maxHorizontalRun,
      maxVerticalRun,
      rowsWithOutline,
      colsWithOutline,
      maxRowLeadingGap,
      maxRowTrailingGap,
      maxColLeadingGap,
      maxColTrailingGap,
    };
  });

  expect(metrics.width).toBe(220);
  expect(metrics.height).toBe(293);

  expect(metrics.redCount).toBeGreaterThanOrEqual(1080);
  expect(metrics.redCount).toBeLessThanOrEqual(1180);

  expect(metrics.boundingWidth).toBeGreaterThanOrEqual(70);
  expect(metrics.boundingWidth).toBeLessThanOrEqual(78);
  expect(metrics.boundingHeight).toBeGreaterThanOrEqual(70);
  expect(metrics.boundingHeight).toBeLessThanOrEqual(78);

  expect(metrics.boundingAreaRatio).toBeGreaterThan(0.075);
  expect(metrics.boundingAreaRatio).toBeLessThan(0.09);
  expect(metrics.coverageRatio).toBeGreaterThan(0.19);
  expect(metrics.coverageRatio).toBeLessThan(0.24);

  expect(metrics.rowsWithOutline).toBeGreaterThanOrEqual(70);
  expect(metrics.colsWithOutline).toBeGreaterThanOrEqual(70);

  expect(metrics.maxHorizontalRun).toBeLessThanOrEqual(50);
  expect(metrics.maxVerticalRun).toBeLessThanOrEqual(45);

  expect(metrics.minX).toBeGreaterThanOrEqual(78);
  expect(metrics.minX).toBeLessThanOrEqual(88);
  expect(metrics.maxX).toBeGreaterThanOrEqual(148);
  expect(metrics.maxX).toBeLessThanOrEqual(158);
  expect(metrics.minY).toBeGreaterThanOrEqual(72);
  expect(metrics.minY).toBeLessThanOrEqual(85);
  expect(metrics.maxY).toBeGreaterThanOrEqual(143);
  expect(metrics.maxY).toBeLessThanOrEqual(154);

  expect(metrics.maxRowLeadingGap).toBeLessThanOrEqual(12);
  expect(metrics.maxRowTrailingGap).toBeLessThanOrEqual(12);
  expect(metrics.maxColLeadingGap).toBeLessThanOrEqual(12);
  expect(metrics.maxColTrailingGap).toBeLessThanOrEqual(12);
});
