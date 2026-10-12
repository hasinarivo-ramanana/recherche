package com.thesis.arlocalizer.level3_pgpnp

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Diagonalisation d'une matrice symetrique n x n par la methode de Jacobi (rotations
 * successives annulant les elements hors-diagonale). Utilise par EPnPSolver pour extraire
 * l'espace nul approximatif de M^T*M (n=12) et pour la SVD 3x3 (via H^T*H) du recalage
 * rigide de Kabsch -- pas de dependance a une bibliotheque d'algebre lineaire externe
 * (LAPACK/EJML), coherent avec le reste du module (PoseMath.solveLinearSystem en Gauss-Jordan).
 */
object JacobiEigen {

    /**
     * @return Pair(valeurs propres (taille n, ordre CROISSANT), vecteurs propres
     *         (matrice n x n, colonne j = vecteur propre associe a la valeur propre j)).
     */
    fun symmetricEigenDecomposition(aIn: Array<DoubleArray>, sweeps: Int = 60): Pair<DoubleArray, Array<DoubleArray>> {
        val n = aIn.size
        val a = Array(n) { i -> aIn[i].copyOf() }
        val v = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

        repeat(sweeps) {
            for (p in 0 until n - 1) {
                for (q in p + 1 until n) {
                    if (abs(a[p][q]) < 1e-15) continue

                    val theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                    val t = if (theta == 0.0) 1.0 else {
                        val sgn = if (theta > 0) 1.0 else -1.0
                        sgn / (abs(theta) + sqrt(theta * theta + 1.0))
                    }
                    val c = 1.0 / sqrt(t * t + 1.0)
                    val s = t * c

                    val app = a[p][p]; val aqq = a[q][q]; val apq = a[p][q]
                    a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
                    a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
                    a[p][q] = 0.0; a[q][p] = 0.0

                    for (i in 0 until n) {
                        if (i != p && i != q) {
                            val aip = a[i][p]; val aiq = a[i][q]
                            a[i][p] = c * aip - s * aiq; a[p][i] = a[i][p]
                            a[i][q] = s * aip + c * aiq; a[q][i] = a[i][q]
                        }
                    }
                    for (i in 0 until n) {
                        val vip = v[i][p]; val viq = v[i][q]
                        v[i][p] = c * vip - s * viq
                        v[i][q] = s * vip + c * viq
                    }
                }
            }
        }

        val eigenvalues = DoubleArray(n) { a[it][it] }
        val order = (0 until n).sortedBy { eigenvalues[it] }
        val sortedEigenvalues = DoubleArray(n) { eigenvalues[order[it]] }
        val sortedEigenvectors = Array(n) { i -> DoubleArray(n) { j -> v[i][order[j]] } }
        return Pair(sortedEigenvalues, sortedEigenvectors)
    }
}