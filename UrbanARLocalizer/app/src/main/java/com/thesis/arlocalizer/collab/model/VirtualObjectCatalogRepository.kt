package com.thesis.arlocalizer.collab.model

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.google.gson.JsonParser
import java.util.UUID

/**
 * Catalogue des objets virtuels geolocalises, combine de deux sources :
 *
 *  1) Un fichier GeoJSON STATIQUE (assets/virtual_objects.geojson), analogue a sig_data.geojson pour
 *     les amers de localisation -- c'est le catalogue "de depart", edite hors-ligne (ex. via
 *     geojson.io comme pour le SIG du Niveau 2).
 *
 *  2) Une base SQLite locale MUTABLE (objets ajoutes depuis l'app, "mode placement"), qui persiste
 *     entre sessions et se synchronise avec les autres utilisateurs via CollabSyncRepository
 *     (message "catalog_add"). Chaque utilisateur qui place un objet l'ajoute a sa propre base ET le
 *     diffuse ; chaque utilisateur qui recoit une annonce l'ajoute aussi a sa base locale, pour que
 *     le catalogue reste disponible meme hors connexion au redemarrage de l'app.
 *
 * Format GeoJSON attendu (Point), proprietes :
 * {
 *   "id": "chaise_bibliotheque_01",
 *   "type": "chaise",                  // doit correspondre a ModelCatalog.allTypes()
 *   "label": "Chaise - Bibliotheque",
 *   "rotation_y_deg": 0,
 *   "color_argb": -1
 * }
 */
class VirtualObjectCatalogRepository private constructor(
    private val staticObjects: List<VirtualObjectDefinition>,
    private val dbHelper: CatalogDbHelper
) {

    companion object {
        fun load(context: Context, geojsonAssetName: String = "virtual_objects.geojson"): VirtualObjectCatalogRepository {
            val staticObjects = loadGeoJson(context, geojsonAssetName)
            val dbHelper = CatalogDbHelper(context)
            return VirtualObjectCatalogRepository(staticObjects, dbHelper)
        }

        private fun loadGeoJson(context: Context, assetName: String): List<VirtualObjectDefinition> {
            return try {
                val text = context.assets.open(assetName).bufferedReader().use { it.readText() }
                val root = JsonParser.parseString(text).asJsonObject
                val features = root.getAsJsonArray("features")
                features.map { el ->
                    val f = el.asJsonObject
                    val coords = f.getAsJsonObject("geometry").getAsJsonArray("coordinates")
                    val props = f.getAsJsonObject("properties")
                    VirtualObjectDefinition(
                        objectId = props.get("id")?.asString ?: UUID.randomUUID().toString(),
                        objectType = props.get("type")?.asString ?: ModelCatalog.TYPE_CHAISE,
                        longitude = coords[0].asDouble,
                        latitude = coords[1].asDouble,
                        label = props.get("label")?.asString ?: "",
                        defaultRotationYDeg = props.get("rotation_y_deg")?.asFloat ?: 0f,
                        defaultColorArgb = props.get("color_argb")?.asInt ?: -1
                    )
                }
            } catch (e: Exception) {
                emptyList() // fichier absent ou invalide : catalogue statique vide, on continue avec SQLite seul
            }
        }
    }

    /** Catalogue combine : statique + additions locales (SQLite), dedoublonne par objectId. */
    fun all(): List<VirtualObjectDefinition> {
        val local = dbHelper.getAll()
        val byId = LinkedHashMap<String, VirtualObjectDefinition>()
        staticObjects.forEach { byId[it.objectId] = it }
        local.forEach { byId[it.objectId] = it } // les additions locales priment en cas de doublon d'id
        return byId.values.toList()
    }

    /** Ajoute un objet place depuis l'app (mode placement) a la base locale. A appeler AVANT diffusion reseau. */
    fun addLocalObject(def: VirtualObjectDefinition) {
        dbHelper.insert(def)
    }

    /** Enregistre un objet recu d'un autre utilisateur (message catalog_add) dans la base locale. */
    fun addRemoteObject(def: VirtualObjectDefinition) {
        dbHelper.insert(def)
    }

    fun findById(objectId: String): VirtualObjectDefinition? = all().firstOrNull { it.objectId == objectId }
}

/** SQLite minimal pour les additions de catalogue faites depuis l'app. */
private class CatalogDbHelper(context: Context) : SQLiteOpenHelper(context, "virtual_objects_local.db", null, 1) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE objects (
                objectId TEXT PRIMARY KEY,
                objectType TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                label TEXT,
                defaultRotationYDeg REAL,
                defaultColorArgb INTEGER
            )
            """.trimIndent()
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS objects")
        onCreate(db)
    }

    fun insert(def: VirtualObjectDefinition) {
        val values = ContentValues().apply {
            put("objectId", def.objectId)
            put("objectType", def.objectType)
            put("latitude", def.latitude)
            put("longitude", def.longitude)
            put("label", def.label)
            put("defaultRotationYDeg", def.defaultRotationYDeg)
            put("defaultColorArgb", def.defaultColorArgb)
        }
        writableDatabase.insertWithOnConflict("objects", null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun getAll(): List<VirtualObjectDefinition> {
        val result = mutableListOf<VirtualObjectDefinition>()
        val cursor = readableDatabase.rawQuery("SELECT * FROM objects", null)
        cursor.use {
            while (it.moveToNext()) {
                result.add(
                    VirtualObjectDefinition(
                        objectId = it.getString(it.getColumnIndexOrThrow("objectId")),
                        objectType = it.getString(it.getColumnIndexOrThrow("objectType")),
                        latitude = it.getDouble(it.getColumnIndexOrThrow("latitude")),
                        longitude = it.getDouble(it.getColumnIndexOrThrow("longitude")),
                        label = it.getString(it.getColumnIndexOrThrow("label")) ?: "",
                        defaultRotationYDeg = it.getFloat(it.getColumnIndexOrThrow("defaultRotationYDeg")),
                        defaultColorArgb = it.getInt(it.getColumnIndexOrThrow("defaultColorArgb"))
                    )
                )
            }
        }
        return result
    }
}
