#!/usr/bin/env python3
"""
relay_server.py — Noeud edge (serveur relais WebSocket) pour l'application collaborative RA
==============================================================================================

Adapte du serveur relais de la demonstration CRDT (UrbanLocalizerAR-CRDTDemoV2/server/relay_server.py).
Deux differences :

  1) Le message "anchor" (ARCore Cloud Anchor) est SUPPRIME : le referentiel spatial commun est
     desormais fourni independamment par la cascade de localisation de chaque telephone
     (PoseAnchorManager), pas par une coordination d'hebergement/resolution d'ancre entre
     dispositifs co-presents. Ce serveur n'a plus besoin de connaitre quoi que ce soit sur des ancres.

  2) Un nouveau type de message "catalog_add" est relaye et persiste : il permet a un utilisateur en
     "mode placement" d'ajouter un nouvel objet geolocalise (chaise/table/ordinateur/...) au catalogue
     partage, diffuse en temps reel a tous les autres utilisateurs connectes (et rejoue aux nouvelles
     connexions, comme pour les etats "object_state").

Le message "object_state" (position/rotation/couleur d'un objet, fusionne par priorite puis HLC)
est INCHANGE dans sa logique de fusion : le serveur reste agnostique du schema applicatif exact
du payload (il ne regarde que "priority" et "timestamp"), donc l'ajout de champs cote Kotlin
(offsetX/Y/Z, rotationYDeg, colorArgb) ne necessite AUCUNE modification ici.

Dependances : `pip install -r requirements.txt` (bibliotheque `websockets`).

Lancement :
    python relay_server.py [--host 0.0.0.0] [--port 8765]
"""

import argparse
import asyncio
import json
import logging
from typing import Any, Dict, Set

try:
    import websockets
except ImportError:
    raise SystemExit(
        "Le paquet 'websockets' est requis. Installez-le avec :\n"
        "    pip install -r requirements.txt"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [edge] %(message)s")
log = logging.getLogger("relay")

# Dernier etat CRDT connu par objectId (manipulation : position/rotation/couleur).
object_states: Dict[str, Dict[str, Any]] = {}

# Catalogue des objets ajoutes dynamiquement depuis l'app (mode placement), par objectId.
# Independant de object_states : un objet peut exister dans le catalogue sans jamais avoir ete
# manipule (auquel cas il conserve son etat par defaut cote client, cf. ObjectManipulationState.default).
catalog_additions: Dict[str, Dict[str, Any]] = {}

CLIENTS: Set["websockets.WebSocketServerProtocol"] = set()


def hlc_key(ts: Dict[str, Any]):
    """Cle de comparaison lexicographique (physical, logical, replicaId)."""
    return (ts["physical"], ts["logical"], ts["replicaId"])


def precedes(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Ordre total <= pour les etats d'objet : priorite d'abord, HLC en depart."""
    if a["priority"] != b["priority"]:
        return a["priority"] < b["priority"]
    return hlc_key(a["timestamp"]) < hlc_key(b["timestamp"])


def merge_object_state(current: Dict[str, Any] | None, incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Operateur de fusion (max au sens de l'ordre total ci-dessus)."""
    if current is None:
        return incoming
    return incoming if precedes(current, incoming) else current


async def broadcast(message: str):
    stale = []
    for ws in CLIENTS:
        try:
            await ws.send(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        CLIENTS.discard(ws)


async def handler(websocket):
    CLIENTS.add(websocket)
    peer = websocket.remote_address
    log.info(f"Client connecte: {peer} (total={len(CLIENTS)})")

    # Synchronise le nouvel arrivant avec le catalogue complet ET le dernier etat connu de chaque
    # objet manipule, pour qu'il demarre a jour sans rejouer tout l'historique.
    for definition in catalog_additions.values():
        await websocket.send(json.dumps(definition))
    for state in object_states.values():
        await websocket.send(json.dumps(state))

    try:
        async for raw in websocket:
            try:
                incoming = json.loads(raw)
                msg_type = incoming.get("type", "object_state")
                object_id = incoming["objectId"]
            except (json.JSONDecodeError, KeyError):
                log.warning(f"Message invalide ignore depuis {peer}")
                continue

            if msg_type == "catalog_add":
                # Pas de fusion CRDT necessaire ici : un ajout de catalogue ne "concurrence" rien,
                # on l'enregistre simplement (dernier recu l'emporte en cas de meme objectId,
                # ce qui ne devrait arriver que si un utilisateur re-place volontairement un objet).
                catalog_additions[object_id] = incoming
                log.info(
                    f"catalogue: nouvel objet id={object_id} type={incoming.get('objectType')} "
                    f"lat={incoming.get('latitude')} lon={incoming.get('longitude')}"
                )
                await broadcast(json.dumps(incoming))
                continue

            current = object_states.get(object_id)
            merged = merge_object_state(current, incoming)
            object_states[object_id] = merged

            winner = "distant" if merged is incoming else "conserve (local deja superieur)"
            log.info(
                f"objet={object_id} priorite={incoming.get('priority')} "
                f"replica={incoming.get('originReplica')} -> fusion: {winner}"
            )

            # Rediffuse l'etat FUSIONNE (et non le message brut recu) a tous les clients, y compris
            # l'emetteur, pour garantir que toutes les repliques convergent vers le meme etat.
            await broadcast(json.dumps(merged))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(websocket)
        log.info(f"Client deconnecte: {peer} (total={len(CLIENTS)})")


async def main():
    parser = argparse.ArgumentParser(description="Noeud edge - relais WebSocket CRDT (objets RA geolocalises)")
    parser.add_argument("--host", default="0.0.0.0", help="Adresse d'ecoute (defaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port d'ecoute (defaut: 8765)")
    args = parser.parse_args()

    log.info(f"Noeud edge en ecoute sur ws://{args.host}:{args.port}")
    log.info("Sur les smartphones, utilisez ws://<IP-DE-CETTE-MACHINE>:%d", args.port)
    async with websockets.serve(handler, args.host, args.port):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
