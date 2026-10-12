package com.thesis.arlocalizer.level3_pgpnp

import kotlin.math.*

/** Petites utilitaires d'algebre pour la representation et l'optimisation de poses SE(3), en Kotlin pur. */
object PoseMath {

    /** Rodrigues : vecteur rotation w (R^3, angle = ||w||, axe = w/||w||) -> matrice de rotation 3x3 (row-major). */
    fun rodrigues(w: DoubleArray): DoubleArray {
        val theta = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
        if (theta < 1e-12) {
            return doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        }
        val kx = w[0] / theta; val ky = w[1] / theta; val kz = w[2] / theta
        val c = cos(theta); val s = sin(theta); val t = 1 - c
        return doubleArrayOf(
            t * kx * kx + c, t * kx * ky - s * kz, t * kx * kz + s * ky,
            t * kx * ky + s * kz, t * ky * ky + c, t * ky * kz - s * kx,
            t * kx * kz - s * ky, t * ky * kz + s * kx, t * kz * kz + c
        )
    }

    /** Compose une petite perturbation exponentielle delta_w a une rotation courante R (mise a jour multiplicative a droite). */
    fun composeRotation(r: DoubleArray, deltaW: DoubleArray): DoubleArray {
        val dR = rodrigues(deltaW)
        return matMul3(r, dR)
    }

    fun matMul3(a: DoubleArray, b: DoubleArray): DoubleArray {
        val out = DoubleArray(9)
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += a[i * 3 + k] * b[k * 3 + j]
            out[i * 3 + j] = s
        }
        return out
    }

    fun matVec3(m: DoubleArray, v: DoubleArray): DoubleArray = doubleArrayOf(
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2]
    )

    /** Projection perspective pi(K, R, t, X) -> (u,v) pixels. K = (f, cx, cy), pas de distorsion. */
    fun project(fx: Double, fy: Double, cx: Double, cy: Double, r: DoubleArray, t: DoubleArray, x: DoubleArray): DoubleArray {
        val p = matVec3(r, x)
        val xc = p[0] + t[0]; val yc = p[1] + t[1]; val zc = p[2] + t[2]
        val zSafe = if (abs(zc) < 1e-6) 1e-6 else zc
        return doubleArrayOf(fx * xc / zSafe + cx, fy * yc / zSafe + cy)
    }

    /**
     * Resout un systeme lineaire symetrique A*x = b (dimension n) par elimination de Gauss-Jordan avec pivot partiel.
     * Utilise a chaque iteration de Gauss-Newton (matrice normale H*dx = -g).
     */
    fun solveLinearSystem(aIn: Array<DoubleArray>, bIn: DoubleArray): DoubleArray? {
        val n = bIn.size
        val a = Array(n) { i -> aIn[i].copyOf() }
        val b = bIn.copyOf()

        for (col in 0 until n) {
            var pivotRow = col
            var maxVal = abs(a[col][col])
            for (r in col + 1 until n) {
                if (abs(a[r][col]) > maxVal) { maxVal = abs(a[r][col]); pivotRow = r }
            }
            if (maxVal < 1e-14) return null
            if (pivotRow != col) {
                val tmp = a[col]; a[col] = a[pivotRow]; a[pivotRow] = tmp
                val tb = b[col]; b[col] = b[pivotRow]; b[pivotRow] = tb
            }
            val pivot = a[col][col]
            for (r in 0 until n) {
                if (r == col) continue
                val factor = a[r][col] / pivot
                if (factor == 0.0) continue
                for (c in col until n) a[r][c] -= factor * a[col][c]
                b[r] -= factor * b[col]
            }
        }
        return DoubleArray(n) { i -> b[i] / a[i][i] }
    }

    /** Centre de camera C (monde) a partir de la pose extrinseque (R, t) : C = -R^T * t */
    fun cameraCenterFromPose(r: DoubleArray, t: DoubleArray): DoubleArray {
        val rt = doubleArrayOf(r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]) // transpose
        val rtT = matVec3(rt, t)
        return doubleArrayOf(-rtT[0], -rtT[1], -rtT[2])
    }

    /** Translation extrinseque t a partir du centre de camera C (monde) : t = -R * C */
    fun extrinsicTranslationFromCameraCenter(r: DoubleArray, c: DoubleArray): DoubleArray {
        val rc = matVec3(r, c)
        return doubleArrayOf(-rc[0], -rc[1], -rc[2])
    }
}
