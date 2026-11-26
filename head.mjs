// @ts-check

/**
 * Generiše 5 temena pravog 4-simpleksa u R⁴.
 * @returns {number[][]} 5 vektora u R⁴
 */
export function fiveCellR4() {
  const ROOT = [1, Math.SQRT2, Math.sqrt(3), 2, Math.sqrt(5)];
  const n = 5;
  /** @type {number[][]} */
  const verts = [];

  for (let i = 0; i < n; i++) {
    const v = new Array(4).fill(0);
    for (let j = 0; j < 4; j++) {
      v[j] = (i === j) ? 1 : -0.25;
    }
    const scale = Math.sqrt(ROOT[i]);
    for (let j = 0; j < 4; j++) v[j] *= scale;
    verts.push(v);
  }

  // Centriranje i normalizacija na jedinicu
  const c = ;
  for (const v of verts) for (let j = 0; j < 4; j++) c[j] += v[j];
  for (let j = 0; j < 4; j++) c[j] /= n;
  for (const v of verts) for (let j = 0; j < 4; j++) v[j] -= c[j];

  for (const v of verts) {
    let norm = Math.sqrt(v.reduce((acc, val) => acc + val * val, 0)) || 1;
    for (let j = 0; j < 4; j++) v[j] /= norm;
  }

  return verts;
}

/**
 * Stereografska projekcija sa S³ u R³ (sa polom u e₄).
 * @param {number[]} p4 Vektor iz R⁴
 * @returns {number[]} [x, y, z] u R³
 */
export function stereographic(p4) {
  const w = p4;
  const d = 1 - w;
  if (Math.abs(d) < 1e-12) return [0, 0, 1e6];
  return [p4 / d, p4 / d, p4 / d];
}

/**
 * Bezdimenzionalna koherencija bisektora (ρ = 1 označava savršeno poklapanje prozorčića).
 * @param {number} ax
 * @param {number} by
 * @returns {number}
 */
export function bisectorCoherence(ax, by) {
  const denominator = ax + by;
  if (Math.abs(denominator) < 1e-12) return 0;
  return (4 * ax * by) / (denominator * denominator);
}
