plugins {
    id("com.android.application") version "8.5.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
    id("com.android.library") version "8.5.0" apply false
}

subprojects {
    val baseBuildDir = "/tmp/UrbanARLocalizer-build"
    layout.buildDirectory.set(file("${baseBuildDir}/${project.path.replace(":", "/")}"))
}
