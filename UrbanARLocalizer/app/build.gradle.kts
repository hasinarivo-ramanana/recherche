plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.thesis.arlocalizer"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.thesis.arlocalizercrdtv2"
        minSdk = 26          // requis par ARCore
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        viewBinding = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // Les modeles / donnees SIG / nuage de points ne doivent pas etre compresses
    androidResources {
        noCompress += listOf("tflite", "ply", "db", "geojson", "glb")
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            pickFirsts += "**/libc++_shared.so"
        }
    }
}

dependencies {
    // WebSocket client pour le canal de synchronisation CRDT temps reel (fonctionnalite collaborative RA)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // CameraX (capture d'image en continu pour la cascade de localisation)
    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")

    // Carte OpenStreetMap (visualisation de l'historique de localisation, cf. Fig. "OSM visualization")
    implementation("org.osmdroid:osmdroid-android:6.1.20")
    implementation("androidx.preference:preference-ktx:1.2.1") // requis par la configuration osmdroid

    // TensorFlow Lite (inference MobileNet-APR et YOLOv10, modeles fournis par l'utilisateur)
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")

    // ARCore + rendu (SceneView encapsule ARCore + Filament, remplace Sceneform)
    implementation("io.github.sceneview:arsceneview:2.2.1")
    implementation("com.google.ar:core:1.44.0") // acces direct a Session/Frame/Pose pour la cascade de localisation

    // Lecture du fichier database.db de COLMAP (descripteurs SIFT, format SQLite)
    implementation("androidx.sqlite:sqlite-ktx:2.4.0")
    implementation("androidx.sqlite:sqlite-framework:2.4.0")

    // Parsing GeoJSON du SIG
    implementation("com.google.code.gson:gson:2.11.0")

    testImplementation("junit:junit:4.13.2")

    // OpenCV (EPnP, RANSAC, SIFT/ORB, Levenberg-Marquardt) :
    // Ajouter le module local `opencv` genere par le SDK Android OpenCV 4.x
    // (voir README.md, section "Integration OpenCV") puis decommenter :
    implementation(project(":opencv"))
}
