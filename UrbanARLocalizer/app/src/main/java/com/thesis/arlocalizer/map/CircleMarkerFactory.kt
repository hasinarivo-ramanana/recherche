package com.thesis.arlocalizer.map

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.Drawable

/**
 * Genere des marqueurs circulaires colores (sans dependre d'assets image) pour distinguer les
 * 3 niveaux de la cascade sur la carte OSM :
 *   Niveau 1 (APR)        -> rouge
 *   Niveau 2 (YOLO+SIG)   -> bleu
 *   Niveau 3 (PG-PnP)     -> vert
 */
object CircleMarkerFactory {

    val COLOR_LEVEL1 = Color.RED
    val COLOR_LEVEL2 = Color.BLUE
    val COLOR_LEVEL3 = Color.rgb(0, 170, 0)

    private val cache = mutableMapOf<Int, Drawable>()

    fun forColor(context: Context, colorInt: Int, diameterDp: Float = 18f): Drawable {
        return cache.getOrPut(colorInt) {
            val density = context.resources.displayMetrics.density
            val diameterPx = (diameterDp * density).toInt().coerceAtLeast(4)
            val bitmap = Bitmap.createBitmap(diameterPx, diameterPx, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)

            val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colorInt; style = Paint.Style.FILL }
            val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 2f * density
            }
            val radius = diameterPx / 2f
            canvas.drawCircle(radius, radius, radius - strokePaint.strokeWidth / 2, fillPaint)
            canvas.drawCircle(radius, radius, radius - strokePaint.strokeWidth / 2, strokePaint)

            BitmapDrawable(context.resources, bitmap)
        }
    }

    fun forLevel(context: Context, level: Int): Drawable = when (level) {
        1 -> forColor(context, COLOR_LEVEL1)
        2 -> forColor(context, COLOR_LEVEL2)
        else -> forColor(context, COLOR_LEVEL3)
    }
}
