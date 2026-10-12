#!/usr/bin/env python3
"""
server.py - Noeud edge unique pour l'application AR
=====================================================
Fusionne en UN SEUL process FastAPI (un seul port) :

  1) POST /localize   -- REST, requete/reponse unique. Recoit une photo (+ cap/heading,
                          focale optionnelle) envoyee a chaque clic sur "Localiser ma
                          position", execute la cascade niv1/2/3 (deportee ici car le
                          niveau 3 en Kotlin presentait une erreur de plusieurs centaines
                          de metres), renvoie {level, lat, lon, yaw_deg, ...} en JSON.

  2) WS   /ws          -- WebSocket, logique de fusion CRDT identique a l'ancien
                          relay_server.py (object_state + catalog_add), pour la phase
                          collaborative une fois l'utilisateur a moins de 10 m de la cible.

Pourquoi un seul serveur : la localisation est un aller-retour ponctuel (REST convient
mieux, cf. discussion), la collab est un flux temps reel (WebSocket) -- mais rien
n'empeche de les servir depuis le meme process/port, ce que FastAPI/Starlette permet
nativement (pas besoin de la lib `websockets` en dependance directe, uvicorn s'en charge).

Lancement :
    uvicorn server:app --host 0.0.0.0 --port 8765

Cote Android, une seule adresse a saisir (ex: http://192.168.1.42:8765) sert de base a
la fois pour POST http://<ip>:8765/localize et pour ws://<ip>:8765/ws.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

import config
from localization_service import LocalizationService, save_upload_to_tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [edge] %(message)s")
log = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Etat global partage (identique en substance a relay_server.py)
# ---------------------------------------------------------------------------
object_states: Dict[str, Dict[str, Any]] = {}
catalog_additions: Dict[str, Dict[str, Any]] = {}
CLIENTS: Set[WebSocket] = set()

service: Optional[LocalizationService] = None
# La cascade (TFLite + OpenCV/SIFT + RANSAC) est bloquante et CPU-bound : on la sort
# de la boucle asyncio (qui sert aussi les clients WebSocket) via un threadpool dedie,
# sinon un /localize en cours geleriat la diffusion CRDT en cours.
_executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, _executor
    from concurrent.futures import ThreadPoolExecutor
    _executor = ThreadPoolExecutor(max_workers=2)

    log.info("Chargement de la cascade de localisation (peut prendre du temps)...")
    service = LocalizationService(
        apr_model=config.DEFAULT_PATHS["apr_model"],
        apr_norm=config.DEFAULT_PATHS["apr_norm_params"],
        yolo_model=config.DEFAULT_PATHS["yolo_model"],
        sig_geojson=config.DEFAULT_PATHS["sig_geojson"],
        colmap_points3d=config.DEFAULT_PATHS["colmap_points3d"],
        colmap_images=config.DEFAULT_PATHS["colmap_images"],
        colmap_cameras=config.DEFAULT_PATHS["colmap_cameras"],
        colmap_db=config.DEFAULT_PATHS["colmap_database"],
        gps_csv=config.DEFAULT_PATHS["gps_csv"],
        cache_dir=config.DEFAULT_PATHS["cache_dir"],
    )
    log.info("Cascade prete, serveur pret a repondre sur /localize et /ws.")
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="AR Localizer Edge Node", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 1) REST -- localisation en cascade (declenchee a chaque clic bouton)
# ---------------------------------------------------------------------------
@app.post("/localize")
async def localize(
    image: UploadFile = File(...),
    heading_deg: Optional[float] = Form(None),
    focal_mm: Optional[float] = Form(None),
    sensor_width_mm: Optional[float] = Form(None),
):
    if service is None:
        return JSONResponse(status_code=503, content={"error": "Service non initialise"})

    raw = await image.read()
    tmp_path = save_upload_to_tempfile(raw, suffix=os.path.splitext(image.filename or "img.jpg")[1] or ".jpg")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _executor, service.localize, tmp_path, heading_deg, focal_mm, sensor_width_mm
        )
    except Exception as e:
        log.exception("Echec de la cascade de localisation")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return {
        "level": result.level,
        "lat": result.lat,
        "lon": result.lon,
        "yaw_deg": result.yaw_deg,
        "pitch_deg": result.pitch_deg,
        "roll_deg": result.roll_deg,
        "sigma_pos_m": result.sigma_pos_m,
        "level1": result.level1,
        "level2": result.level2,
        "level3": result.level3,
        "elapsed_ms": result.elapsed_ms,
        "detail": result.detail,
    }


# ---------------------------------------------------------------------------
# 2) WebSocket -- collaboration CRDT (reprend telle quelle la logique de relay_server.py)
# ---------------------------------------------------------------------------
def _hlc_key(ts: Dict[str, Any]):
    return (ts["physical"], ts["logical"], ts["replicaId"])


def _precedes(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a["priority"] != b["priority"]:
        return a["priority"] < b["priority"]
    return _hlc_key(a["timestamp"]) < _hlc_key(b["timestamp"])


def _merge_object_state(current: Optional[Dict[str, Any]], incoming: Dict[str, Any]) -> Dict[str, Any]:
    if current is None:
        return incoming
    return incoming if _precedes(current, incoming) else current


async def _broadcast(message: str):
    stale = []
    for ws in CLIENTS:
        try:
            await ws.send_text(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        CLIENTS.discard(ws)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    CLIENTS.add(websocket)
    log.info(f"Client WS connecte (total={len(CLIENTS)})")

    for definition in catalog_additions.values():
        await websocket.send_text(json.dumps(definition))
    for state in object_states.values():
        await websocket.send_text(json.dumps(state))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                incoming = json.loads(raw)
                msg_type = incoming.get("type", "object_state")
                object_id = incoming["objectId"]
            except (json.JSONDecodeError, KeyError):
                log.warning("Message WS invalide ignore")
                continue

            if msg_type == "catalog_add":
                catalog_additions[object_id] = incoming
                log.info(f"catalogue: nouvel objet id={object_id} type={incoming.get('objectType')}")
                await _broadcast(json.dumps(incoming))
                continue

            current = object_states.get(object_id)
            merged = _merge_object_state(current, incoming)
            object_states[object_id] = merged
            await _broadcast(json.dumps(merged))
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(websocket)
        log.info(f"Client WS deconnecte (total={len(CLIENTS)})")
