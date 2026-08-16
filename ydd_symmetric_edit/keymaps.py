# SPDX-License-Identifier: GPL-3.0-or-later

"""Pass-through overlays for supported native mesh-cut invocation routes."""

from __future__ import annotations

import hashlib
import traceback
from dataclasses import dataclass

import bpy

from . import gizmo_adopt, session_state
from ._types import KeymapEvent, KeymapEventLike, KeymapFingerprint, KeymapIdentity, NativeRoute
from .operators import EXTRUDE_TOOL_KINDS, TOOL_PROFILES

INTERCEPT_OPERATOR = "mesh.ydd_symmetric_edit_intercept"
CONNECT_OPERATOR = "mesh.ydd_symmetric_edit_connect"
DISSOLVE_MODE_OPERATOR = "mesh.ydd_symmetric_edit_dissolve_mode"
EXTRUDE_MENU_OPENER = "mesh.ydd_symmetric_edit_extrude_menu"
MERGE_MENU = "YSE_MT_merge"
DELETE_MENU = "YSE_MT_delete"
EXTRUDE_MENU = "YSE_MT_extrude"
NATIVE_CONNECT_OPERATOR = "mesh.vert_connect_path"
NATIVE_MERGE_MENU = "VIEW3D_MT_edit_mesh_merge"
NATIVE_DELETE_MENU = "VIEW3D_MT_edit_mesh_delete"
NATIVE_EXTRUDE_MENU = "VIEW3D_MT_edit_mesh_extrude"
NATIVE_DISSOLVE_MODE = "mesh.dissolve_mode"
TOOL_KEYMAP_NAME = TOOL_PROFILES["KNIFE"].tool_idnames[0]
TOOL_KEYMAP_NAMES = frozenset(tool_idname for profile in TOOL_PROFILES.values() for tool_idname in profile.tool_idnames)
OPERATOR_TOOL_KINDS = {profile.keymap_operator: profile.kind for profile in TOOL_PROFILES.values()}

_OWN_OPERATOR_IDS = frozenset({INTERCEPT_OPERATOR, CONNECT_OPERATOR, DISSOLVE_MODE_OPERATOR, EXTRUDE_MENU_OPENER})
_WATCH_INTERVAL = 1.0
_RETRY_INTERVAL = 0.25


_REGISTERED_ITEMS: list[tuple[object, KeymapEventLike]] = []
_ROUTES_BY_KEY: dict[str, NativeRoute] = {}
_FINGERPRINT: KeymapFingerprint | None = None
_DeleteRouteFingerprint = tuple[str, str, str, KeymapEvent, bool, tuple[tuple[str, object], ...]]
_DELETE_FINGERPRINT: tuple[_DeleteRouteFingerprint, ...] | None = None
_DISSOLVE_FINGERPRINT: tuple[_DeleteRouteFingerprint, ...] | None = None
_ReplayRouteFingerprint = tuple[str, str, str, KeymapEvent, bool]
_ReplayFingerprint = tuple[tuple[_ReplayRouteFingerprint, ...], tuple[_ReplayRouteFingerprint, ...]]
_REPLAY_FINGERPRINT: _ReplayFingerprint | None = None
_ExtrudeMenuFingerprint = tuple[str, str, str, KeymapEvent, str]
_EXTRUDE_MENU_FINGERPRINT: tuple[_ExtrudeMenuFingerprint, ...] | None = None
_EXTRUDE_MENU_ROUTES_BY_KEY: dict[str, ExtrudeMenuRoute] = {}
_HAS_DELETE_ROUTES = False
_HAS_EXTRUDE_MENU_ROUTES = False
_ENABLED = False
_RUNNING = False


@dataclass(frozen=True, slots=True)
class DeleteMenuRoute:
    """One scanned native delete-menu / dissolve_mode binding to clone."""

    keymap_name: str
    space_type: str
    region_type: str
    event: KeymapEvent
    is_tool: bool = False
    # Explicitly-set native KMI properties (dissolve_mode use_verts etc.).
    properties: tuple[tuple[str, object], ...] = ()

    @property
    def keymap_identity(self) -> KeymapIdentity:
        return KeymapIdentity(
            name=self.keymap_name,
            space_type=self.space_type,
            region_type=self.region_type,
        )


@dataclass(frozen=True, slots=True)
class ReplayKeymapRoute:
    """One native Connect operator or Merge menu binding to clone."""

    keymap_name: str
    space_type: str
    region_type: str
    event: KeymapEvent
    is_tool: bool = False

    @property
    def keymap_identity(self) -> KeymapIdentity:
        return KeymapIdentity(
            name=self.keymap_name,
            space_type=self.space_type,
            region_type=self.region_type,
        )


@dataclass(frozen=True, slots=True)
class ExtrudeMenuRoute:
    """One scanned native extrude-menu call_menu binding to clone with an opener."""

    keymap_name: str
    space_type: str
    region_type: str
    event: KeymapEvent
    menu_name: str = NATIVE_EXTRUDE_MENU
    is_tool: bool = False
    route_key: str = ""

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


def _capture_set_kmi_properties(item) -> tuple[tuple[str, object], ...]:
    """Return (identifier, value) for properties explicitly set on a KMI."""

    props = getattr(item, "properties", None)
    if props is None:
        return ()
    captured: list[tuple[str, object]] = []
    try:
        for prop in props.bl_rna.properties:
            identifier = prop.identifier
            if identifier == "rna_type":
                continue
            if not props.is_property_set(identifier):
                continue
            value = getattr(props, identifier)
            # Fingerprint / seen-set require hashable, stable values. Macro
            # child pointers stringify with a changing address and must not
            # participate in route identity (G2 only needs scalar KMI props).
            if isinstance(value, (bool, int, float, str)):
                captured.append((identifier, value))
    except Exception:
        traceback.print_exc()
        return ()
    return tuple(captured)


def _route_fingerprint(route: DeleteMenuRoute) -> _DeleteRouteFingerprint:
    return (
        route.keymap_name,
        route.space_type,
        route.region_type,
        route.event,
        route.is_tool,
        route.properties,
    )


def _replay_route_fingerprint(route: ReplayKeymapRoute) -> _ReplayRouteFingerprint:
    return (
        route.keymap_name,
        route.space_type,
        route.region_type,
        route.event,
        route.is_tool,
    )


def _extrude_menu_fingerprint(route: ExtrudeMenuRoute) -> _ExtrudeMenuFingerprint:
    return (
        route.keymap_name,
        route.space_type,
        route.region_type,
        route.event,
        route.menu_name,
    )


def _is_native_extrude_call_menu(item) -> bool:
    return item.idname == "wm.call_menu" and getattr(item.properties, "name", "") == NATIVE_EXTRUDE_MENU


def _native_routes(window_manager) -> tuple[list[NativeRoute], KeymapFingerprint]:
    key_config = window_manager.keyconfigs.user
    active_config = window_manager.keyconfigs.active
    if key_config is None or active_config is None:
        return [], KeymapFingerprint(active_config_name="", routes=())

    routes = []
    for keymap in key_config.keymaps:
        if keymap.is_modal:
            continue

        legacy_by_event: dict[KeymapEvent, list[KeymapEventLike]] = {}
        extrude_by_event: dict[KeymapEvent, list[KeymapEventLike]] = {}
        call_menu_events: set[KeymapEvent] = set()
        for item in keymap.keymap_items:
            if not item.active:
                continue
            event = _event_signature(item)
            if _is_native_extrude_call_menu(item):
                call_menu_events.add(event)
                continue
            if item.idname not in OPERATOR_TOOL_KINDS:
                continue
            tool_kind = OPERATOR_TOOL_KINDS[item.idname]
            if tool_kind in EXTRUDE_TOOL_KINDS:
                extrude_by_event.setdefault(event, []).append(item)
            else:
                legacy_by_event.setdefault(event, []).append(item)

        for event, matching_items in legacy_by_event.items():
            if event in call_menu_events:
                continue
            native_operators = tuple(dict.fromkeys(item.idname for item in matching_items))
            # Two different supported operators on the exact same physical
            # event cannot be identified by a PASS_THROUGH pre-hook.  Skipping
            # that unusual ambiguous route is safer than preparing the wrong
            # topology session before Blender resolves operator polling.
            if len(native_operators) != 1:
                continue
            native_operator = native_operators[0]
            routes.append(
                NativeRoute(
                    keymap_name=keymap.name,
                    space_type=keymap.space_type,
                    region_type=keymap.region_type,
                    is_tool=keymap.name in TOOL_KEYMAP_NAMES,
                    native_operator=native_operator,
                    tool_kind=OPERATOR_TOOL_KINDS[native_operator],
                    event=event,
                    route_key=_make_route_key(
                        keymap.name,
                        keymap.space_type,
                        keymap.region_type,
                        native_operator,
                        event,
                    ),
                )
            )

        for event, matching_items in extrude_by_event.items():
            if event in legacy_by_event:
                continue
            if event in call_menu_events:
                continue
            native_operators = tuple(dict.fromkeys(item.idname for item in matching_items))
            if len(native_operators) != 1:
                continue
            if len(matching_items) > 1:
                continue
            native_operator = native_operators[0]
            routes.append(
                NativeRoute(
                    keymap_name=keymap.name,
                    space_type=keymap.space_type,
                    region_type=keymap.region_type,
                    is_tool=keymap.name in TOOL_KEYMAP_NAMES,
                    native_operator=native_operator,
                    tool_kind=OPERATOR_TOOL_KINDS[native_operator],
                    event=event,
                    route_key=_make_route_key(
                        keymap.name,
                        keymap.space_type,
                        keymap.region_type,
                        native_operator,
                        event,
                    ),
                    kmi_properties=_capture_set_kmi_properties(matching_items[0]),
                )
            )

    fingerprint = KeymapFingerprint(
        active_config_name=active_config.name,
        routes=tuple(routes),
    )
    return routes, fingerprint


def _replay_keymap_routes(
    window_manager,
) -> tuple[list[ReplayKeymapRoute], list[ReplayKeymapRoute], _ReplayFingerprint]:
    """Scan user keymaps for active native Connect and Merge routes."""

    key_config = window_manager.keyconfigs.user
    if key_config is None:
        return [], [], ((), ())

    connect_routes: list[ReplayKeymapRoute] = []
    merge_routes: list[ReplayKeymapRoute] = []
    seen_connect: set[_ReplayRouteFingerprint] = set()
    seen_merge: set[_ReplayRouteFingerprint] = set()
    for keymap in key_config.keymaps:
        if keymap.is_modal:
            continue
        for item in keymap.keymap_items:
            if not item.active:
                continue

            is_connect = item.idname == NATIVE_CONNECT_OPERATOR
            is_merge = item.idname == "wm.call_menu" and getattr(item.properties, "name", "") == NATIVE_MERGE_MENU
            if not is_connect and not is_merge:
                continue

            route = ReplayKeymapRoute(
                keymap_name=keymap.name,
                space_type=keymap.space_type,
                region_type=keymap.region_type,
                event=_event_signature(item),
                is_tool=keymap.name in TOOL_KEYMAP_NAMES,
            )
            identity = _replay_route_fingerprint(route)
            if is_connect and identity not in seen_connect:
                seen_connect.add(identity)
                connect_routes.append(route)
            if is_merge and identity not in seen_merge:
                seen_merge.add(identity)
                merge_routes.append(route)

    # An event bound to both Connect and the Merge menu is ambiguous; give
    # up on both rather than let registration order decide (same fail-closed
    # stance as _native_routes).
    ambiguous = seen_connect & seen_merge
    if ambiguous:
        connect_routes = [route for route in connect_routes if _replay_route_fingerprint(route) not in ambiguous]
        merge_routes = [route for route in merge_routes if _replay_route_fingerprint(route) not in ambiguous]

    fingerprint = (
        tuple(_replay_route_fingerprint(route) for route in connect_routes),
        tuple(_replay_route_fingerprint(route) for route in merge_routes),
    )
    return connect_routes, merge_routes, fingerprint


def _delete_menu_routes(
    window_manager,
) -> tuple[list[DeleteMenuRoute], tuple[_DeleteRouteFingerprint, ...]]:
    """Scan user keymaps for active delete-menu call_menu bindings.

    Does not touch ``_native_routes`` / ``OPERATOR_TOOL_KINDS`` (those only
    match operator idnames).  Physical events are cloned via ``_event_signature``.
    """

    key_config = window_manager.keyconfigs.user
    if key_config is None:
        return [], ()

    routes: list[DeleteMenuRoute] = []
    seen: set[_DeleteRouteFingerprint] = set()
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
            route = DeleteMenuRoute(
                keymap_name=keymap.name,
                space_type=keymap.space_type,
                region_type=keymap.region_type,
                event=event,
                is_tool=keymap.name in TOOL_KEYMAP_NAMES,
            )
            identity = _route_fingerprint(route)
            if identity in seen:
                continue
            seen.add(identity)
            routes.append(route)

    fingerprint = tuple(_route_fingerprint(route) for route in routes)
    return routes, fingerprint


def _extrude_menu_routes(
    window_manager,
) -> tuple[list[ExtrudeMenuRoute], tuple[_ExtrudeMenuFingerprint, ...]]:
    """Scan user keymaps for active extrude-menu call_menu bindings.

    Fail-closed with intercept: a keymap+event that also has any
    ``OPERATOR_TOOL_KINDS`` KMI yields neither an opener nor an intercept.
    Uniqueness is per keymap identity + event (not global).
    """

    key_config = window_manager.keyconfigs.user
    if key_config is None:
        return [], ()

    routes: list[ExtrudeMenuRoute] = []
    seen: set[_ExtrudeMenuFingerprint] = set()
    for keymap in key_config.keymaps:
        if keymap.is_modal:
            continue
        for item in keymap.keymap_items:
            if not item.active:
                continue
            if not _is_native_extrude_call_menu(item):
                continue
            event = _event_signature(item)
            native, _opener, other, supported = _extrude_menu_event_census(keymap, event)
            if native != 1 or other != 0 or supported != 0:
                continue
            route = ExtrudeMenuRoute(
                keymap_name=keymap.name,
                space_type=keymap.space_type,
                region_type=keymap.region_type,
                event=event,
                menu_name=NATIVE_EXTRUDE_MENU,
                is_tool=keymap.name in TOOL_KEYMAP_NAMES,
                route_key=_make_route_key(
                    keymap.name,
                    keymap.space_type,
                    keymap.region_type,
                    NATIVE_EXTRUDE_MENU,
                    event,
                ),
            )
            identity = _extrude_menu_fingerprint(route)
            if identity in seen:
                continue
            seen.add(identity)
            routes.append(route)

    fingerprint = tuple(_extrude_menu_fingerprint(route) for route in routes)
    return routes, fingerprint


def _dissolve_mode_routes(
    window_manager,
) -> tuple[list[DeleteMenuRoute], tuple[_DeleteRouteFingerprint, ...]]:
    """Scan user keymaps for active mesh.dissolve_mode bindings (Ctrl+X / IC)."""

    key_config = window_manager.keyconfigs.user
    if key_config is None:
        return [], ()

    routes: list[DeleteMenuRoute] = []
    seen: set[_DeleteRouteFingerprint] = set()
    for keymap in key_config.keymaps:
        if keymap.is_modal:
            continue
        for item in keymap.keymap_items:
            if not item.active:
                continue
            if item.idname != NATIVE_DISSOLVE_MODE:
                continue
            event = _event_signature(item)
            properties = _capture_set_kmi_properties(item)
            route = DeleteMenuRoute(
                keymap_name=keymap.name,
                space_type=keymap.space_type,
                region_type=keymap.region_type,
                event=event,
                is_tool=keymap.name in TOOL_KEYMAP_NAMES,
                properties=properties,
            )
            identity = _route_fingerprint(route)
            if identity in seen:
                continue
            seen.add(identity)
            routes.append(route)

    fingerprint = tuple(_route_fingerprint(route) for route in routes)
    return routes, fingerprint


def _is_owned_item(item) -> bool:
    if item.idname in _OWN_OPERATOR_IDS:
        return True
    if item.idname != "wm.call_menu":
        return False
    menu_name = getattr(item.properties, "name", "")
    return menu_name in {MERGE_MENU, DELETE_MENU, EXTRUDE_MENU}


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


def _register_replay_keymaps(
    window_manager,
    connect_routes: list[ReplayKeymapRoute],
    merge_routes: list[ReplayKeymapRoute],
) -> None:
    addon_config = window_manager.keyconfigs.addon
    if addon_config is None:
        raise RuntimeError("Blender's add-on key configuration is unavailable")

    if not connect_routes:
        print("ydd Symmetric Edit: no Connect keymap route found; Connect mirroring not registered")
    if not merge_routes:
        print("ydd Symmetric Edit: no Merge keymap route found; Merge mirroring not registered")

    addon_keymaps = {}
    for operator_id, menu_name, routes in (
        (CONNECT_OPERATOR, "", connect_routes),
        ("wm.call_menu", MERGE_MENU, merge_routes),
    ):
        for route in routes:
            cache_key = (route.keymap_identity, route.is_tool)
            keymap = addon_keymaps.get(cache_key)
            if keymap is None:
                keymap = addon_config.keymaps.new(
                    name=route.keymap_name,
                    space_type=route.space_type,
                    region_type=route.region_type,
                    modal=False,
                    tool=route.is_tool,
                )
                addon_keymaps[cache_key] = keymap

            item = keymap.keymap_items.new(
                operator_id,
                head=True,
                **_event_arguments(route.event),
            )
            if menu_name:
                try:
                    item.properties.name = menu_name
                except Exception:
                    keymap.keymap_items.remove(item)
                    raise
            item.active = _ENABLED
            _REGISTERED_ITEMS.append((keymap, item))


def _register_delete_menu_keymaps(window_manager, routes: list[DeleteMenuRoute]) -> None:
    """Register head=True YSE_MT_delete bindings for every scanned native event."""

    addon_config = window_manager.keyconfigs.addon
    if addon_config is None:
        raise RuntimeError("Blender's add-on key configuration is unavailable")

    addon_keymaps = {}
    for route in routes:
        cache_key = (route.keymap_identity, route.is_tool)
        keymap = addon_keymaps.get(cache_key)
        if keymap is None:
            keymap = addon_config.keymaps.new(
                name=route.keymap_name,
                space_type=route.space_type,
                region_type=route.region_type,
                modal=False,
                tool=route.is_tool,
            )
            addon_keymaps[cache_key] = keymap

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


def _register_dissolve_mode_keymaps(window_manager, routes: list[DeleteMenuRoute]) -> None:
    """Register head=True dissolve_mode replacement for every scanned native event."""

    addon_config = window_manager.keyconfigs.addon
    if addon_config is None:
        raise RuntimeError("Blender's add-on key configuration is unavailable")

    addon_keymaps = {}
    for route in routes:
        cache_key = (route.keymap_identity, route.is_tool)
        keymap = addon_keymaps.get(cache_key)
        if keymap is None:
            keymap = addon_config.keymaps.new(
                name=route.keymap_name,
                space_type=route.space_type,
                region_type=route.region_type,
                modal=False,
                tool=route.is_tool,
            )
            addon_keymaps[cache_key] = keymap

        item = keymap.keymap_items.new(
            DISSOLVE_MODE_OPERATOR,
            head=True,
            **_event_arguments(route.event),
        )
        # Replay explicitly-set native KMI props (custom use_verts etc.).
        for prop_name, prop_value in route.properties:
            try:
                setattr(item.properties, prop_name, prop_value)
            except Exception:
                traceback.print_exc()
        item.active = _ENABLED
        _REGISTERED_ITEMS.append((keymap, item))


def _register_extrude_menu_keymaps(window_manager, routes: list[ExtrudeMenuRoute]) -> None:
    """Register head=True opener operators for every scanned native extrude menu."""

    addon_config = window_manager.keyconfigs.addon
    if addon_config is None:
        raise RuntimeError("Blender's add-on key configuration is unavailable")

    addon_keymaps = {}
    for route in routes:
        cache_key = (route.keymap_identity, route.is_tool)
        keymap = addon_keymaps.get(cache_key)
        if keymap is None:
            keymap = addon_config.keymaps.new(
                name=route.keymap_name,
                space_type=route.space_type,
                region_type=route.region_type,
                modal=False,
                tool=route.is_tool,
            )
            addon_keymaps[cache_key] = keymap

        item = keymap.keymap_items.new(
            EXTRUDE_MENU_OPENER,
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


def _rebuild(
    window_manager,
    routes: list[NativeRoute],
    connect_routes: list[ReplayKeymapRoute],
    merge_routes: list[ReplayKeymapRoute],
    delete_routes: list[DeleteMenuRoute],
    dissolve_routes: list[DeleteMenuRoute],
    extrude_menu_routes: list[ExtrudeMenuRoute],
) -> None:
    global _HAS_DELETE_ROUTES, _HAS_EXTRUDE_MENU_ROUTES

    addon_config = window_manager.keyconfigs.addon
    if addon_config is None:
        raise RuntimeError("Blender's add-on key configuration is unavailable")

    _remove_owned_items(addon_config)
    _REGISTERED_ITEMS.clear()
    _ROUTES_BY_KEY.clear()
    _EXTRUDE_MENU_ROUTES_BY_KEY.clear()
    _HAS_DELETE_ROUTES = False
    _HAS_EXTRUDE_MENU_ROUTES = False

    try:
        _register_routes(window_manager, routes)
        _register_replay_keymaps(window_manager, connect_routes, merge_routes)
        _register_delete_menu_keymaps(window_manager, delete_routes)
        _register_dissolve_mode_keymaps(window_manager, dissolve_routes)
        _register_extrude_menu_keymaps(window_manager, extrude_menu_routes)
    except Exception:
        _remove_owned_items(addon_config)
        _REGISTERED_ITEMS.clear()
        _HAS_DELETE_ROUTES = False
        _HAS_EXTRUDE_MENU_ROUTES = False
        _EXTRUDE_MENU_ROUTES_BY_KEY.clear()
        window_manager.keyconfigs.update()
        raise

    _ROUTES_BY_KEY.update((route.route_key, route) for route in routes)
    _EXTRUDE_MENU_ROUTES_BY_KEY.update((route.route_key, route) for route in extrude_menu_routes)
    _HAS_DELETE_ROUTES = bool(delete_routes)
    _HAS_EXTRUDE_MENU_ROUTES = bool(extrude_menu_routes)
    window_manager.keyconfigs.update()


def _refresh(*, force: bool = False) -> bool:
    global _FINGERPRINT, _REPLAY_FINGERPRINT, _DELETE_FINGERPRINT, _DISSOLVE_FINGERPRINT
    global _EXTRUDE_MENU_FINGERPRINT

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
    connect_routes, merge_routes, replay_fingerprint = _replay_keymap_routes(window_manager)
    delete_routes, delete_fingerprint = _delete_menu_routes(window_manager)
    dissolve_routes, dissolve_fingerprint = _dissolve_mode_routes(window_manager)
    extrude_menu_routes, extrude_menu_fingerprint = _extrude_menu_routes(window_manager)
    if (
        force
        or fingerprint != _FINGERPRINT
        or replay_fingerprint != _REPLAY_FINGERPRINT
        or delete_fingerprint != _DELETE_FINGERPRINT
        or dissolve_fingerprint != _DISSOLVE_FINGERPRINT
        or extrude_menu_fingerprint != _EXTRUDE_MENU_FINGERPRINT
    ):
        _rebuild(
            window_manager,
            routes,
            connect_routes,
            merge_routes,
            delete_routes,
            dissolve_routes,
            extrude_menu_routes,
        )
        _FINGERPRINT = fingerprint
        _REPLAY_FINGERPRINT = replay_fingerprint
        _DELETE_FINGERPRINT = delete_fingerprint
        _DISSOLVE_FINGERPRINT = dissolve_fingerprint
        _EXTRUDE_MENU_FINGERPRINT = extrude_menu_fingerprint
    return True


def _watch_keymaps():
    if not _RUNNING or not _ENABLED:
        _sync_gizmo_poll()
        return None
    try:
        if not _refresh():
            _sync_gizmo_poll()
            return _RETRY_INTERVAL
    except Exception:
        traceback.print_exc()
        _sync_gizmo_poll()
        return _RETRY_INTERVAL
    _sync_gizmo_poll()
    return _WATCH_INTERVAL


def _poll_gizmo_global():
    # The arm predicate walks every window's toolsystem, so at 20 Hz the fast
    # tick only reads the flag the 1s watcher maintains via _sync_gizmo_poll.
    if not _RUNNING or not _ENABLED or not session_state._GIZMO_POLL_ARMED:
        return None
    try:
        gizmo_adopt.poll_global()
    except Exception:
        traceback.print_exc()
    return gizmo_adopt.GIZMO_POLL_INTERVAL


def _sync_gizmo_poll() -> None:
    should_run = _RUNNING and _ENABLED and gizmo_adopt.arm_required()
    session_state._GIZMO_POLL_ARMED = should_run
    registered = bpy.app.timers.is_registered(_poll_gizmo_global)
    if should_run and not registered:
        gizmo_adopt.prime_onset_state()
        bpy.app.timers.register(
            _poll_gizmo_global,
            first_interval=gizmo_adopt.GIZMO_POLL_INTERVAL,
            persistent=True,
        )
    elif not should_run and registered:
        bpy.app.timers.unregister(_poll_gizmo_global)
        session_state._GIZMO_MODAL_POINTERS_BY_WINDOW.clear()


def _ensure_watcher() -> None:
    if not bpy.app.timers.is_registered(_watch_keymaps):
        bpy.app.timers.register(
            _watch_keymaps,
            first_interval=_RETRY_INTERVAL,
            persistent=True,
        )


def _unique_live_supported_kmi(route: NativeRoute):
    """Return the unique live supported KMI for ``route``, or ``None``.

    Contract v7 §4.2: on the saved route's keymap + event, count every active
    KMI whose ``idname`` is in ``OPERATOR_TOOL_KINDS`` (any kind). That total
    must be exactly 1, and that idname must equal the saved route.
    """

    window_manager = _window_manager()
    if window_manager is None or window_manager.keyconfigs.user is None:
        return None
    keymap = _find_keymap(
        window_manager.keyconfigs.user,
        route.keymap_identity,
    )
    if keymap is None:
        return None

    matching = [
        item
        for item in keymap.keymap_items
        if item.active and item.idname in OPERATOR_TOOL_KINDS and _event_signature(item) == route.event
    ]
    if len(matching) != 1:
        return None
    if matching[0].idname != route.native_operator:
        return None
    return matching[0]


def route_is_current(route_key: str) -> bool:
    """Return whether an intercept still precedes the native route it cloned."""

    if not _RUNNING or not _ENABLED:
        return False
    route = _ROUTES_BY_KEY.get(route_key)
    if route is None:
        return False

    item = _unique_live_supported_kmi(route)
    if item is None:
        return False
    if route.tool_kind in EXTRUDE_TOOL_KINDS:
        return _capture_set_kmi_properties(item) == route.kmi_properties
    return True


def route_tool_kind(route_key: str) -> str | None:
    """Return the operation family captured by a currently registered route."""

    route = _ROUTES_BY_KEY.get(route_key)
    return route.tool_kind if route is not None else None


def route_kmi_properties(route_key: str) -> tuple[tuple[str, object], ...]:
    """Return the KMI props captured when the route was registered."""

    route = _ROUTES_BY_KEY.get(route_key)
    return route.kmi_properties if route is not None else ()


def live_route_kmi_properties(route_key: str) -> tuple[tuple[str, object], ...] | None:
    """Return explicit props of the unique live KMI for an extrude route.

    ``None`` means the live KMI is missing or not unique.
    """

    route = _ROUTES_BY_KEY.get(route_key)
    if route is None:
        return None
    item = _unique_live_supported_kmi(route)
    if item is None:
        return None
    return _capture_set_kmi_properties(item)


def live_route_has_dissolve_and_intersect(route_key: str) -> bool:
    """G2 live check: True when the current KMI sets dissolve_and_intersect."""

    props = live_route_kmi_properties(route_key)
    if props is None:
        return False
    return any(name == "dissolve_and_intersect" and bool(value) for name, value in props)


def has_delete_routes() -> bool:
    """Return whether the latest scan registered one or more delete-menu routes."""

    return _HAS_DELETE_ROUTES


def has_extrude_menu_routes() -> bool:
    """Return whether the latest scan registered one or more extrude-menu openers."""

    return _HAS_EXTRUDE_MENU_ROUTES


def _extrude_menu_event_census(keymap, event: KeymapEvent) -> tuple[int, int, int, int]:
    """Return (target_native, opener, other_call_menu, supported) on keymap+event."""

    native = opener = other = supported = 0
    for item in keymap.keymap_items:
        if not item.active or _event_signature(item) != event:
            continue
        if item.idname in OPERATOR_TOOL_KINDS:
            supported += 1
        elif item.idname == EXTRUDE_MENU_OPENER:
            opener += 1
        elif item.idname == "wm.call_menu":
            if getattr(item.properties, "name", "") == NATIVE_EXTRUDE_MENU:
                native += 1
            else:
                other += 1
    return native, opener, other, supported


def extrude_menu_route_is_current(route_key: str) -> bool:
    """Live verify: user native 1 + addon opener 1, no other call-menu or supported ops."""

    if not _RUNNING or not _ENABLED:
        return False
    route = _EXTRUDE_MENU_ROUTES_BY_KEY.get(route_key)
    if route is None:
        return False

    window_manager = _window_manager()
    if window_manager is None or window_manager.keyconfigs.user is None:
        return False
    if window_manager.keyconfigs.addon is None:
        return False

    user_keymap = _find_keymap(window_manager.keyconfigs.user, route.keymap_identity)
    addon_keymap = _find_keymap(window_manager.keyconfigs.addon, route.keymap_identity)
    if user_keymap is None or addon_keymap is None:
        return False

    user_native, _user_opener, user_other, user_supported = _extrude_menu_event_census(user_keymap, route.event)
    _addon_native, addon_opener, addon_other, addon_supported = _extrude_menu_event_census(addon_keymap, route.event)
    return (
        user_native == 1
        and addon_opener == 1
        and user_other + addon_other == 0
        and user_supported + addon_supported == 0
    )


def sync(enabled: bool) -> None:
    """Apply the persistent toggle without changing Blender's native KMI."""

    global _ENABLED

    _ENABLED = bool(enabled)
    if not _RUNNING:
        return

    window_manager = _window_manager()
    if not _ENABLED:
        _sync_gizmo_poll()
        gizmo_adopt.clear_runtime_state()
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
    _sync_gizmo_poll()


def register(*, enabled: bool = False) -> None:
    global _ENABLED, _FINGERPRINT, _REPLAY_FINGERPRINT, _DELETE_FINGERPRINT, _DISSOLVE_FINGERPRINT
    global _EXTRUDE_MENU_FINGERPRINT
    global _HAS_DELETE_ROUTES, _HAS_EXTRUDE_MENU_ROUTES, _RUNNING

    _RUNNING = True
    _ENABLED = bool(enabled)
    _FINGERPRINT = None
    _REPLAY_FINGERPRINT = None
    _DELETE_FINGERPRINT = None
    _DISSOLVE_FINGERPRINT = None
    _EXTRUDE_MENU_FINGERPRINT = None
    _HAS_DELETE_ROUTES = False
    _HAS_EXTRUDE_MENU_ROUTES = False
    _REGISTERED_ITEMS.clear()
    _ROUTES_BY_KEY.clear()
    _EXTRUDE_MENU_ROUTES_BY_KEY.clear()

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
        _sync_gizmo_poll()


def unregister() -> None:
    global _ENABLED, _FINGERPRINT, _REPLAY_FINGERPRINT, _DELETE_FINGERPRINT, _DISSOLVE_FINGERPRINT
    global _EXTRUDE_MENU_FINGERPRINT
    global _HAS_DELETE_ROUTES, _HAS_EXTRUDE_MENU_ROUTES, _RUNNING

    _RUNNING = False
    _ENABLED = False
    if bpy.app.timers.is_registered(_watch_keymaps):
        bpy.app.timers.unregister(_watch_keymaps)
    if bpy.app.timers.is_registered(_poll_gizmo_global):
        bpy.app.timers.unregister(_poll_gizmo_global)
    gizmo_adopt.clear_runtime_state()

    window_manager = _window_manager()
    if window_manager is not None:
        # Match Blender's add-on keymap helper: remove possible customized
        # copies from the resolved user config as well as our add-on sources.
        _remove_owned_items(window_manager.keyconfigs.user)
        _remove_owned_items(window_manager.keyconfigs.addon)
        window_manager.keyconfigs.update()

    _REGISTERED_ITEMS.clear()
    _ROUTES_BY_KEY.clear()
    _EXTRUDE_MENU_ROUTES_BY_KEY.clear()
    _FINGERPRINT = None
    _REPLAY_FINGERPRINT = None
    _DELETE_FINGERPRINT = None
    _DISSOLVE_FINGERPRINT = None
    _EXTRUDE_MENU_FINGERPRINT = None
    _HAS_DELETE_ROUTES = False
    _HAS_EXTRUDE_MENU_ROUTES = False
