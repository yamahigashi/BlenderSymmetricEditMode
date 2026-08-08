# SPDX-License-Identifier: GPL-3.0-or-later

"""Pass-through overlays for supported native mesh-cut invocation routes."""

from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass

import bpy

from ._types import KeymapEvent, KeymapEventLike, KeymapFingerprint, KeymapIdentity, NativeRoute
from .operators import TOOL_PROFILES

INTERCEPT_OPERATOR = "mesh.ydd_symmetric_edit_intercept"
CONNECT_OPERATOR = "mesh.ydd_symmetric_edit_connect"
MERGE_MENU = "YSE_MT_merge"
DELETE_MENU = "YSE_MT_delete"
NATIVE_DELETE_MENU = "VIEW3D_MT_edit_mesh_delete"
TOOL_KEYMAP_NAME = TOOL_PROFILES["KNIFE"].tool_idnames[0]
TOOL_KEYMAP_NAMES = frozenset(tool_idname for profile in TOOL_PROFILES.values() for tool_idname in profile.tool_idnames)
OPERATOR_TOOL_KINDS = {profile.keymap_operator: profile.kind for profile in TOOL_PROFILES.values()}

_OWN_OPERATOR_IDS = frozenset({INTERCEPT_OPERATOR, CONNECT_OPERATOR})
_WATCH_INTERVAL = 1.0
_RETRY_INTERVAL = 0.25


_REGISTERED_ITEMS: list[tuple[object, KeymapEventLike]] = []
_ROUTES_BY_KEY: dict[str, NativeRoute] = {}
_FINGERPRINT: KeymapFingerprint | None = None
_DELETE_FINGERPRINT: tuple[tuple[str, str, str, KeymapEvent], ...] | None = None
_HAS_DELETE_ROUTES = False
_ENABLED = False
_RUNNING = False


@dataclass(frozen=True, slots=True)
class DeleteMenuRoute:
    """One scanned native delete-menu binding to clone into the add-on config."""

    keymap_name: str
    space_type: str
    region_type: str
    event: KeymapEvent

    @property
    def keymap_identity(self) -> KeymapIdentity:
        return KeymapIdentity(
            name=self.keymap_name,
            space_type=self.space_type,
            region_type=self.region_type,
        )


def _window_manager():
    return getattr(bpy.context, "window_manager", None)


def _event_signature(item: KeymapEventLike) -> KeymapEvent:
    return KeymapEvent(
        type=str(item.type),
        value=str(item.value),
        any=bool(item.any),
        # shift/ctrl/alt/oskey/hyper are three-valued (-1 'Any', 0, 1) on
        # Blender's KeyMapItem. Casting to bool would collapse -1 ('Any')
        # into True (required), silently turning an "Any" native binding
        # into a "must hold this modifier" intercept registration.
        shift=int(item.shift),
        ctrl=int(item.ctrl),
        alt=int(item.alt),
        oskey=int(item.oskey),
        # Hyper was added after Blender 4.4. Missing means the modifier cannot
        # be part of that version's keymap, which is equivalent to 0 (False).
        hyper=int(getattr(item, "hyper", 0)),
        key_modifier=str(item.key_modifier),
        direction=str(item.direction),
        repeat=bool(item.repeat),
    )


def _event_arguments(event: KeymapEvent) -> dict[str, object]:
    arguments = {
        "type": event.type,
        "value": event.value,
    }

    if event.any:
        arguments["any"] = True
    else:
        for field in ("shift", "ctrl", "alt", "oskey", "hyper"):
            value = getattr(event, field)
            if value:
                arguments[field] = value

    if event.key_modifier != "NONE":
        arguments["key_modifier"] = event.key_modifier
    if event.direction != "ANY":
        arguments["direction"] = event.direction
    if event.repeat:
        arguments["repeat"] = True
    return arguments


def _make_route_key(
    keymap_name: str,
    space_type: str,
    region_type: str,
    native_operator: str,
    event: KeymapEvent,
) -> str:
    payload = repr((keymap_name, space_type, region_type, native_operator, event)).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=16).hexdigest()
    return f"yse:{digest}"


def _find_keymap(key_config, identity: KeymapIdentity):
    return next(
        (
            keymap
            for keymap in key_config.keymaps
            if keymap.name == identity.name
            and keymap.space_type == identity.space_type
            and keymap.region_type == identity.region_type
            and not keymap.is_modal
        ),
        None,
    )


def _native_routes(window_manager) -> tuple[list[NativeRoute], KeymapFingerprint]:
    key_config = window_manager.keyconfigs.user
    active_config = window_manager.keyconfigs.active
    if key_config is None or active_config is None:
        return [], KeymapFingerprint(active_config_name="", routes=())

    routes = []
    for keymap in key_config.keymaps:
        if keymap.is_modal:
            continue

        items_by_event: dict[KeymapEvent, list[KeymapEventLike]] = {}
        for item in keymap.keymap_items:
            if not item.active or item.idname not in OPERATOR_TOOL_KINDS:
                continue
            event = _event_signature(item)
            items_by_event.setdefault(event, []).append(item)

        for event, matching_items in items_by_event.items():
            native_operators = tuple(dict.fromkeys(item.idname for item in matching_items))
            # Two different supported operators on the exact same physical
            # event cannot be identified by a PASS_THROUGH pre-hook.  Skipping
            # that unusual ambiguous route is safer than preparing the wrong
            # topology session before Blender resolves operator polling.
            if len(native_operators) != 1:
                continue
            native_operator = native_operators[0]

            route_key = _make_route_key(
                keymap.name,
                keymap.space_type,
                keymap.region_type,
                native_operator,
                event,
            )
            routes.append(
                NativeRoute(
                    keymap_name=keymap.name,
                    space_type=keymap.space_type,
                    region_type=keymap.region_type,
                    is_tool=keymap.name in TOOL_KEYMAP_NAMES,
                    native_operator=native_operator,
                    tool_kind=OPERATOR_TOOL_KINDS[native_operator],
                    event=event,
                    route_key=route_key,
                )
            )

    fingerprint = KeymapFingerprint(
        active_config_name=active_config.name,
        routes=tuple(routes),
    )
    return routes, fingerprint


def _delete_menu_routes(
    window_manager,
) -> tuple[list[DeleteMenuRoute], tuple[tuple[str, str, str, KeymapEvent], ...]]:
    """Scan user keymaps for active delete-menu call_menu bindings.

    Does not touch ``_native_routes`` / ``OPERATOR_TOOL_KINDS`` (those only
    match operator idnames).  Physical events are cloned via ``_event_signature``.
    """

    key_config = window_manager.keyconfigs.user
    if key_config is None:
        return [], ()

    routes: list[DeleteMenuRoute] = []
    seen: set[tuple[str, str, str, KeymapEvent]] = set()
    for keymap in key_config.keymaps:
        if keymap.is_modal:
            continue
        for item in keymap.keymap_items:
            if not item.active:
                continue
            if item.idname != "wm.call_menu":
                continue
            if getattr(item.properties, "name", "") != NATIVE_DELETE_MENU:
                continue
            event = _event_signature(item)
            identity = (keymap.name, keymap.space_type, keymap.region_type, event)
            if identity in seen:
                continue
            seen.add(identity)
            routes.append(
                DeleteMenuRoute(
                    keymap_name=keymap.name,
                    space_type=keymap.space_type,
                    region_type=keymap.region_type,
                    event=event,
                )
            )

    fingerprint = tuple((route.keymap_name, route.space_type, route.region_type, route.event) for route in routes)
    return routes, fingerprint


def _is_owned_item(item) -> bool:
    if item.idname in _OWN_OPERATOR_IDS:
        return True
    if item.idname != "wm.call_menu":
        return False
    menu_name = getattr(item.properties, "name", "")
    return menu_name in {MERGE_MENU, DELETE_MENU}


def _remove_owned_items(key_config) -> None:
    if key_config is None:
        return
    for keymap in key_config.keymaps:
        owned_items = tuple(item for item in keymap.keymap_items if _is_owned_item(item))
        for item in owned_items:
            try:
                keymap.keymap_items.remove(item)
            except (ReferenceError, RuntimeError):
                pass


def _register_routes(window_manager, routes: list[NativeRoute]) -> None:
    addon_config = window_manager.keyconfigs.addon
    if addon_config is None:
        raise RuntimeError("Blender's add-on key configuration is unavailable")

    addon_keymaps = {}
    for route in routes:
        keymap = addon_keymaps.get(route.keymap_identity)
        if keymap is None:
            keymap = addon_config.keymaps.new(
                name=route.keymap_name,
                space_type=route.space_type,
                region_type=route.region_type,
                modal=False,
                tool=route.is_tool,
            )
            addon_keymaps[route.keymap_identity] = keymap

        item = keymap.keymap_items.new(
            INTERCEPT_OPERATOR,
            head=True,
            **_event_arguments(route.event),
        )
        try:
            item.properties.route_key = route.route_key
        except Exception:
            keymap.keymap_items.remove(item)
            raise
        item.active = _ENABLED
        _REGISTERED_ITEMS.append((keymap, item))


def _register_replay_keymaps(window_manager) -> None:
    addon_config = window_manager.keyconfigs.addon
    if addon_config is None:
        raise RuntimeError("Blender's add-on key configuration is unavailable")

    keymap = addon_config.keymaps.new(
        name="Mesh",
        space_type="EMPTY",
        region_type="WINDOW",
        modal=False,
    )
    connect = keymap.keymap_items.new(
        CONNECT_OPERATOR,
        type="J",
        value="PRESS",
        head=True,
    )
    connect.active = _ENABLED
    _REGISTERED_ITEMS.append((keymap, connect))

    merge_menu = keymap.keymap_items.new(
        "wm.call_menu",
        type="M",
        value="PRESS",
        head=True,
    )
    try:
        merge_menu.properties.name = MERGE_MENU
    except Exception:
        keymap.keymap_items.remove(merge_menu)
        raise
    merge_menu.active = _ENABLED
    _REGISTERED_ITEMS.append((keymap, merge_menu))


def _register_delete_menu_keymaps(window_manager, routes: list[DeleteMenuRoute]) -> None:
    """Register head=True YSE_MT_delete bindings for every scanned native event."""

    addon_config = window_manager.keyconfigs.addon
    if addon_config is None:
        raise RuntimeError("Blender's add-on key configuration is unavailable")

    addon_keymaps = {}
    for route in routes:
        keymap = addon_keymaps.get(route.keymap_identity)
        if keymap is None:
            keymap = addon_config.keymaps.new(
                name=route.keymap_name,
                space_type=route.space_type,
                region_type=route.region_type,
                modal=False,
            )
            addon_keymaps[route.keymap_identity] = keymap

        item = keymap.keymap_items.new(
            "wm.call_menu",
            head=True,
            **_event_arguments(route.event),
        )
        try:
            item.properties.name = DELETE_MENU
        except Exception:
            keymap.keymap_items.remove(item)
            raise
        item.active = _ENABLED
        _REGISTERED_ITEMS.append((keymap, item))


def _rebuild(
    window_manager,
    routes: list[NativeRoute],
    delete_routes: list[DeleteMenuRoute],
) -> None:
    global _HAS_DELETE_ROUTES

    addon_config = window_manager.keyconfigs.addon
    if addon_config is None:
        raise RuntimeError("Blender's add-on key configuration is unavailable")

    _remove_owned_items(addon_config)
    _REGISTERED_ITEMS.clear()
    _ROUTES_BY_KEY.clear()
    _HAS_DELETE_ROUTES = False

    try:
        _register_routes(window_manager, routes)
        _register_replay_keymaps(window_manager)
        _register_delete_menu_keymaps(window_manager, delete_routes)
    except Exception:
        _remove_owned_items(addon_config)
        _REGISTERED_ITEMS.clear()
        _HAS_DELETE_ROUTES = False
        window_manager.keyconfigs.update()
        raise

    _ROUTES_BY_KEY.update((route.route_key, route) for route in routes)
    _HAS_DELETE_ROUTES = bool(delete_routes)
    window_manager.keyconfigs.update()


def _refresh(*, force: bool = False) -> bool:
    global _FINGERPRINT, _DELETE_FINGERPRINT

    window_manager = _window_manager()
    if window_manager is None:
        return False
    if (
        window_manager.keyconfigs.user is None
        or window_manager.keyconfigs.active is None
        or window_manager.keyconfigs.addon is None
    ):
        return False

    window_manager.keyconfigs.update()
    routes, fingerprint = _native_routes(window_manager)
    delete_routes, delete_fingerprint = _delete_menu_routes(window_manager)
    if force or fingerprint != _FINGERPRINT or delete_fingerprint != _DELETE_FINGERPRINT:
        _rebuild(window_manager, routes, delete_routes)
        _FINGERPRINT = fingerprint
        _DELETE_FINGERPRINT = delete_fingerprint
    return True


def _watch_keymaps():
    if not _RUNNING or not _ENABLED:
        return None
    try:
        if not _refresh():
            return _RETRY_INTERVAL
    except Exception:
        traceback.print_exc()
        return _RETRY_INTERVAL
    return _WATCH_INTERVAL


def _ensure_watcher() -> None:
    if not bpy.app.timers.is_registered(_watch_keymaps):
        bpy.app.timers.register(
            _watch_keymaps,
            first_interval=_RETRY_INTERVAL,
            persistent=True,
        )


def route_is_current(route_key: str) -> bool:
    """Return whether an intercept still precedes the native route it cloned."""

    if not _RUNNING or not _ENABLED:
        return False
    route = _ROUTES_BY_KEY.get(route_key)
    if route is None:
        return False

    window_manager = _window_manager()
    if window_manager is None or window_manager.keyconfigs.user is None:
        return False
    keymap = _find_keymap(
        window_manager.keyconfigs.user,
        route.keymap_identity,
    )
    if keymap is None:
        return False

    return any(
        item.active and item.idname == route.native_operator and _event_signature(item) == route.event
        for item in keymap.keymap_items
    )


def route_tool_kind(route_key: str) -> str | None:
    """Return the operation family captured by a currently registered route."""

    route = _ROUTES_BY_KEY.get(route_key)
    return route.tool_kind if route is not None else None


def has_delete_routes() -> bool:
    """Return whether the latest scan registered one or more delete-menu routes."""

    return _HAS_DELETE_ROUTES


def sync(enabled: bool) -> None:
    """Apply the persistent toggle without changing Blender's native KMI."""

    global _ENABLED

    _ENABLED = bool(enabled)
    if not _RUNNING:
        return

    window_manager = _window_manager()
    if not _ENABLED:
        for _keymap, item in tuple(_REGISTERED_ITEMS):
            try:
                item.active = False
            except (ReferenceError, RuntimeError):
                pass
        if window_manager is not None:
            window_manager.keyconfigs.update()
        return

    try:
        _refresh(force=True)
    except Exception:
        traceback.print_exc()
    _ensure_watcher()


def register(*, enabled: bool = False) -> None:
    global _ENABLED, _FINGERPRINT, _DELETE_FINGERPRINT, _HAS_DELETE_ROUTES, _RUNNING

    _RUNNING = True
    _ENABLED = bool(enabled)
    _FINGERPRINT = None
    _DELETE_FINGERPRINT = None
    _HAS_DELETE_ROUTES = False
    _REGISTERED_ITEMS.clear()
    _ROUTES_BY_KEY.clear()

    window_manager = _window_manager()
    if window_manager is not None:
        # Clean remnants from an interrupted script reload.  Never remove the
        # shared KeyMap itself: other add-ons may also own items in it.
        _remove_owned_items(window_manager.keyconfigs.user)
        _remove_owned_items(window_manager.keyconfigs.addon)
        window_manager.keyconfigs.update()

    if _ENABLED:
        try:
            _refresh(force=True)
        except Exception:
            traceback.print_exc()
        _ensure_watcher()


def unregister() -> None:
    global _ENABLED, _FINGERPRINT, _DELETE_FINGERPRINT, _HAS_DELETE_ROUTES, _RUNNING

    _RUNNING = False
    _ENABLED = False
    if bpy.app.timers.is_registered(_watch_keymaps):
        bpy.app.timers.unregister(_watch_keymaps)

    window_manager = _window_manager()
    if window_manager is not None:
        # Match Blender's add-on keymap helper: remove possible customized
        # copies from the resolved user config as well as our add-on sources.
        _remove_owned_items(window_manager.keyconfigs.user)
        _remove_owned_items(window_manager.keyconfigs.addon)
        window_manager.keyconfigs.update()

    _REGISTERED_ITEMS.clear()
    _ROUTES_BY_KEY.clear()
    _FINGERPRINT = None
    _DELETE_FINGERPRINT = None
    _HAS_DELETE_ROUTES = False
