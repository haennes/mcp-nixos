"""Arbitrary-flake data source (source=<flake-ref>) for MCP-NixOS server.

When the ``source`` argument to the ``nix`` tool is not one of the KNOWN_SOURCES
(e.g. ``nixos``, ``home-manager``, ...), it is treated as an arbitrary nix flake
reference (e.g. ``nixpkgs``, ``github:owner/repo``) and handled by this module.
This follows the same convention ``flake-inputs`` uses for flake directories.

``search``, ``info``, ``browse``, and ``cache`` mirror the ``nixos`` source's
behavior but are scoped to the given flake's own outputs:

- **packages**: ``packages.<system>.*`` (via ``nix flake show --json``)
- **options**: evaluated from the flake's ``nixosModules.*`` and
  ``homeManagerModules.*`` through the Nix module system (``lib.evalModules``),
  cached per (ref, system) via ``FlakeRefCache`` and invalidated by store path
  (no TTL)
- ``store`` fans out to ls/read over the flake's materialized ``/nix/store`` tree

Security:

- ``_validate_flake_ref`` rejects empty refs, any whitespace, ``"`` / ``\\`` / ``$``
  (which would break the embedded eval expression), and local-path forms
  (``path:`` / ``file:`` / ``/``, ``./``, ``../``, ``~``, single-letter drives).
- Flake refs are always passed as a single argv element (no shell).
- Filesystem access goes through ``_validate_store_path`` and the battle-tested
  ``store`` module.

All ``nix`` invocations go through ``server._run_nix_command`` via the getter
pattern so tests can patch the command runner.
"""

import asyncio
import json
import os
import platform
import re
from collections.abc import Callable, Coroutine
from typing import Any

from ..utils import (
    _validate_store_path,
    error,
    score_option_match,
)
from .flake_inputs import _check_nix_available, _flatten_inputs
from .nixhub import _check_system_cache
from .store import _store_ls, _store_read

# Upper bound display for a browse listing of option matches.
_BROWSE_DISPLAY_LIMIT = 100

# Long timeout for flake evaluation (options catalogue, flake show).
_EVAL_TIMEOUT = 300

# Shared nixpkgs-resolution snippet used by both the fingerprint probe and the
# options evaluation. The flake's own `nixpkgs` input is preferred (with an
# unrealized-input fallback to the registered `nixpkgs` flake), so that an
# experimental flake can pin its library. Keeping it in one place guarantees the
# probe and the eval cannot drift apart.
_BASE_RESOLUTION_TEMPLATE = (
    '  f = builtins.getFlake "%(ref)s";\n'
    '  base = if builtins.hasAttr "nixpkgs" f.inputs\n'
    "    then (if builtins.hasAttr \"legacyPackages\" f.inputs.nixpkgs\n"
    '          then f.inputs.nixpkgs\n'
    '          else (builtins.tryEval (builtins.getFlake "nixpkgs")).value)\n'
    '    else (builtins.tryEval (builtins.getFlake "nixpkgs")).value;\n'
)


def _fingerprint_expr(ref: str) -> str:
    """Nix expression yielding ``{ flake, lib }`` store paths for a ref.

    ``flake`` is the flake's own materialized store path; ``lib`` is the store
    path of the ``nixpkgs`` the options evaluation will actually use. Together
    they pin the inputs that determine the contents of every cached value, so
    a cache entry can be invalidated purely by comparing store paths instead of
    by wall-clock TTL.
    """
    base = _BASE_RESOLUTION_TEMPLATE % {"ref": ref}
    return "let\n" + base + "in { flake = f.outPath; lib = base.outPath; }"


class FlakeRefCache:
    """Process-local cache for arbitrary-flake data, invalidated by store path.

    There is no time-based TTL. Entries are stamped with the flake's
    materialized store path (plus, for the module-options catalogue, the
    resolved ``nixpkgs`` ``base`` path) at build time, and every request
    re-probes those paths with one cheap ``nix eval``. The expensive step —
    ``nix flake show`` / ``nix flake archive`` / module-option evaluation —
    only re-runs when a stamped path changes.

    Because the probe re-resolves the ref exactly the way a fresh ``nix`` call
    would, cached data can never be staler than a fresh evaluation: if ``nix``
    serves a cached fetch the path is unchanged and so is our entry; if the
    flake (or the registry ``nixpkgs`` it falls back to) advances, the path
    changes and we rebuild. Probe and eval failures are surfaced to the caller
    as errors — we never silently serve stale data.
    """

    def __init__(self) -> None:
        self._system: str | None = None
        # ref -> (flake_store_path, show)
        self._show: dict[str, tuple[str, dict[str, Any]]] = {}
        # ref -> (flake_store_path, root, inputs)
        self._archives: dict[str, tuple[str, str, dict[str, str]]] = {}
        # (ref, system) -> (fingerprint, catalogue)
        self._options: dict[tuple[str, str], tuple[dict[str, str], list[dict[str, str]]]] = {}
        self._lock = asyncio.Lock()

    async def current_system(self) -> str:
        """The current system triple, computed once per process (machine-constant)."""
        if self._system is not None:
            return self._system
        async with self._lock:
            if self._system is not None:
                return self._system
            success, stdout, _stderr = await _run_nix(
                ["eval", "--impure", "--raw", "--expr", "builtins.currentSystem"], 30
            )
            self._system = stdout.strip() if success and stdout.strip() else f"{platform.machine()}-linux"
        return self._system

    async def _fingerprint(self, ref: str) -> tuple[bool, dict[str, str] | None, str]:
        """Current ``{ flake, lib }`` store paths for a ref."""
        success, stdout, stderr = await _run_nix(
            ["eval", "--impure", "--json", "--expr", _fingerprint_expr(ref)], _EVAL_TIMEOUT
        )
        if not success:
            return False, None, stderr.strip() or "flake fingerprint eval failed"
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            return False, None, f"Failed to parse flake fingerprint output: {e}"
        if not isinstance(data, dict) or not data.get("flake"):
            return False, None, "Flake evaluation returned no store path"
        return True, {"flake": str(data["flake"]), "lib": str(data.get("lib", ""))}, ""

    async def get_show(self, ref: str) -> tuple[bool, dict[str, Any] | None, str]:
        """``nix flake show --json`` output, cached until the flake path changes."""
        ok, fp, err = await self._fingerprint(ref)
        if not ok:
            return False, None, err
        assert fp is not None
        cached = self._show.get(ref)
        if cached is not None and cached[0] == fp["flake"]:
            return True, cached[1], ""
        async with self._lock:
            cached = self._show.get(ref)
            if cached is not None and cached[0] == fp["flake"]:
                return True, cached[1], ""
            success, stdout, stderr = await _run_nix(["flake", "show", "--json", ref], _EVAL_TIMEOUT)
            if not success:
                return False, None, stderr.strip() or "nix flake show failed"
            try:
                show = json.loads(stdout)
            except json.JSONDecodeError as e:
                return False, None, f"Failed to parse flake show output: {e}"
            self._show[ref] = (fp["flake"], show)
            return True, show, ""

    async def get_archive(self, ref: str) -> tuple[bool, str, dict[str, str], str]:
        """``nix flake archive --json`` (root path + inputs), cached by flake path."""
        ok, fp, err = await self._fingerprint(ref)
        if not ok:
            return False, "", {}, err
        assert fp is not None
        cached = self._archives.get(ref)
        if cached is not None and cached[0] == fp["flake"]:
            return True, cached[1], cached[2], ""
        async with self._lock:
            cached = self._archives.get(ref)
            if cached is not None and cached[0] == fp["flake"]:
                return True, cached[1], cached[2], ""
            success, stdout, stderr = await _run_nix(["flake", "archive", "--json", ref], _EVAL_TIMEOUT)
            if not success:
                return False, "", {}, stderr.strip() or "nix flake archive failed"
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError as e:
                return False, "", {}, f"Failed to parse archive output: {e}"
            if not isinstance(data, dict):
                return False, "", {}, "Flake archive output was not an object"
            root = str(data.get("path", ""))
            inputs = _flatten_inputs(data)
            self._archives[ref] = (fp["flake"], root, inputs)
            return True, root, inputs, ""

    async def get_options(
        self, ref: str, system: str
    ) -> tuple[bool, list[dict[str, str]] | None, str]:
        """The module-options catalogue, cached until the flake or its nixpkgs base moves."""
        ok, fp, err = await self._fingerprint(ref)
        if not ok:
            return False, None, err
        assert fp is not None
        key = (ref, system)
        cached = self._options.get(key)
        if cached is not None and cached[0] == fp:
            return True, cached[1], ""
        async with self._lock:
            cached = self._options.get(key)
            if cached is not None and cached[0] == fp:
                return True, cached[1], ""
            success, stdout, stderr = await _run_nix(
                ["eval", "--impure", "--json", "--expr", _options_expr(ref, system)], _EVAL_TIMEOUT
            )
            if not success:
                return False, None, stderr.strip() or "nix eval failed"
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError as e:
                return False, None, f"Failed to parse options output: {e}"
            if not isinstance(data, dict):
                return False, None, "Options evaluation returned an unexpected shape"
            catalogue = _flatten_catalogue(data)
            self._options[key] = (fp, catalogue)
            return True, catalogue, ""


flake_ref_cache = FlakeRefCache()

# Flake ref validation.
_LOCAL_SCHEMES = {"path", "file"}
_BARE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]*$")
_SCHEME_REF = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_ATTR_PATH = re.compile(r"^[A-Za-z0-9._\-]+$")


def _validate_flake_ref(ref: str) -> bool:
    """Check a user-supplied flake reference for shape problems.

    Only remote-style references are accepted: bare registry aliases
    (``nixpkgs``, ``home-manager``) and scheme-prefixed refs
    (``github:o/r``, ``git+https://...``, ``https://...``). Local-path refs
    and anything that could break the embedded ``builtins.getFlake "<ref>"``
    expression are rejected.
    """
    if not ref or not ref.strip():
        return False
    value = ref.strip()
    if any(ch.isspace() for ch in value):
        return False
    if any(ch in value for ch in ('"', "\\", "$")):
        return False
    match = _SCHEME_REF.match(value)
    if match is not None:
        scheme = match.group(1).lower()
        if scheme in _LOCAL_SCHEMES:
            return False
        if len(scheme) == 1:  # Windows drive letter, e.g. C:\\...
            return False
        return True
    return bool(_BARE_REF.match(value))


def _get_run_nix_command() -> Callable[..., Coroutine[Any, Any, tuple[bool, str, str]]]:
    """Get _run_nix_command from the server module, allowing test mocking."""
    # pylint: disable=import-outside-toplevel
    from .. import server

    return server._run_nix_command


async def _run_nix(args: list[str], timeout: int = 60) -> tuple[bool, str, str]:
    """Run a nix command through the (mockable) server runner."""
    if not _check_nix_available():
        return False, "", "nix not available"
    return await _get_run_nix_command()(args, None, timeout)


async def _current_system() -> str:
    """Return the current nix system triple, computed once per process.

    Falls back to ``<machine>-linux`` if the eval fails.
    """
    return await flake_ref_cache.current_system()


def _flake_available_systems(show: dict[str, Any]) -> list[str]:
    """List the systems the flake exposes packages for (from flake show JSON)."""
    packages = show.get("packages", {}) if isinstance(show, dict) else {}
    if not isinstance(packages, dict):
        return []
    return sorted(
        sys_name for sys_name, entries in packages.items() if isinstance(entries, dict) and entries
    )


# =============================================================================
# Flake data retrieval helpers
# =============================================================================


async def _flake_ref_metadata(ref: str) -> tuple[bool, dict[str, Any] | None, str]:
    """Fetch ``nix flake metadata --json`` for a ref."""
    success, stdout, stderr = await _run_nix(["flake", "metadata", "--json", ref])
    if not success:
        return False, None, stderr.strip() or "nix flake metadata failed"
    try:
        return True, json.loads(stdout), ""
    except json.JSONDecodeError as e:
        return False, None, f"Failed to parse metadata output: {e}"


async def _flake_ref_show(ref: str) -> tuple[bool, dict[str, Any], str]:
    """Fetch ``nix flake show --json`` for a ref (the flake's output tree).

    Cached until the flake's store path changes.
    """
    ok, show, err = await flake_ref_cache.get_show(ref)
    return (True, show, "") if ok and show is not None else (False, {}, err)


async def _flake_ref_archive(ref: str) -> tuple[bool, str, dict[str, str], str]:
    """Materialize the flake via ``nix flake archive --json``.

    Returns ``(ok, root_store_path, inputs, error_message)`` where ``inputs``
    maps flattened input names (e.g. ``nixpkgs``, ``flake-parts.nixpkgs-lib``)
    to their store paths. Cached until the flake's store path changes.
    """
    return await flake_ref_cache.get_archive(ref)


# =============================================================================
# Module options catalogue (evaluated via the Nix module system)
# =============================================================================

# Evaluates every nixosModules.* / homeManagerModules.* output as a module and
# extracts its option tree. `--impure` is required for `builtins.getFlake`.
# The flake's own `nixpkgs` input is preferred for lib/pkgs, falling back to
# the registered `nixpkgs` flake. Each module is wrapped in tryEval/deepSeq so
# a single module that fails to evaluate degrades to { error = ... } instead of
# killing the whole catalogue.
#
# `f` and `base` are resolved via the shared `_BASE_RESOLUTION_TEMPLATE`, which
# is the same expression used by the store-path fingerprint probe. The two
# cannot drift apart because they share a single source of truth, and the eval
# output does not re-emit the store paths — the cache is stamped from the
# already-validated probe result.

_OPTIONS_EXPR_BODY = """\
  sys = "%(system)s";
  lib = ((base.legacyPackages.${sys} or base.legacyPackages.x86_64-linux).lib);
  pkgs = (base.legacyPackages.${sys} or base.legacyPackages.x86_64-linux);
  shim = { _module.args.pkgs = pkgs; _module.args.nixpkgs = base; _module.args.system = sys; };
  optRec = o: {
    name = o.name;
    type = if o.type == null then "" else (o.type.description or toString o.type);
    description = o.description or "";
  };
  optsFor = mod: let
      e = lib.evalModules { modules = [ shim mod ]; };
      out = builtins.map (n: optRec e.options.${n}) (builtins.attrNames e.options);
    in builtins.tryEval (builtins.deepSeq out out);
  collect = kind: builtins.listToAttrs (builtins.map
    (name: let r = optsFor f.${kind}.${name};
      in { inherit name;
           value = if r.success then r.value else { error = "module failed to evaluate"; }; })
    (builtins.attrNames (if builtins.hasAttr kind f then f.${kind} else {})));
in {
  system = sys;
  nixosModules = if builtins.hasAttr "nixosModules" f then collect "nixosModules" else {};
  homeManagerModules = if builtins.hasAttr "homeManagerModules" f then collect "homeManagerModules" else {};
}
"""


def _options_expr(ref: str, system: str) -> str:
    """Nix expression evaluating every module's option tree for a ref and system."""
    base = _BASE_RESOLUTION_TEMPLATE % {"ref": ref}
    body = _OPTIONS_EXPR_BODY % {"system": system}
    return "let\n" + base + body


def _flatten_catalogue(data: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten the eval output into per-option records for search/info/browse."""
    result: list[dict[str, str]] = []
    kinds = (("nixosModules", "nixos"), ("homeManagerModules", "home-manager"))
    for kind_key, label in kinds:
        modules = data.get(kind_key, {}) if isinstance(data, dict) else {}
        if not isinstance(modules, dict):
            continue
        for mod_name, opts in modules.items():
            if not isinstance(opts, list):
                continue  # skipped/failed modules carry {"error": ...}
            for opt in opts:
                if not isinstance(opt, dict):
                    continue
                name = opt.get("name", "")
                if not name:
                    continue
                result.append(
                    {
                        "name": name,
                        "type": opt.get("type", ""),
                        "description": opt.get("description", ""),
                        "module": mod_name,
                        "kind": label,
                    }
                )
    return result


async def _flake_ref_options_catalogue(
    ref: str, system: str
) -> tuple[bool, list[dict[str, str]] | None, str]:
    """Build (cached) the module options catalogue for a ref and system."""
    return await flake_ref_cache.get_options(ref, system)


# =============================================================================
# Formatters for the flake's own packages
# =============================================================================


def _flake_package_names(show: dict[str, Any]) -> set[str]:
    """Collect package attribute names across all systems from flake show."""
    packages = show.get("packages", {}) if isinstance(show, dict) else {}
    names: set[str] = set()
    if not isinstance(packages, dict):
        return names
    for _sys_name, entries in packages.items():
        if not isinstance(entries, dict):
            continue
        names.update(entry for entry in entries if isinstance(entry, str))
    return names


def _format_package_search(ref: str, query: str, matches: list[tuple[int, str, str]]) -> str:
    """Render ranked package search results as plain text."""
    results = [f"Found {len(matches)} packages in flake '{ref}' matching '{query}':", ""]
    for _score, attr, pname in matches:
        results.append(f"* {pname}")
        results.append(f"  Attribute: {attr}")
        results.append("")
    return "\n".join(results).strip()


# =============================================================================
# Action implementations
# =============================================================================


async def _flake_ref_search_packages(ref: str, query: str, limit: int) -> str:
    """Search the flake's packages.<system>.* attribute names, ranked by score."""
    ok, show, err_msg = await _flake_ref_show(ref)
    if not ok:
        return error(err_msg, "FLAKE_ERROR")

    matches: list[tuple[int, str, str]] = []
    for attr in sorted(_flake_package_names(show)):
        score = score_option_match(attr, "", query)
        if score:
            matches.append((score, attr, attr))
    matches.sort(key=lambda match: (-match[0], match[1].casefold()))
    matches = matches[:limit]

    if not matches:
        return f"No packages found in flake '{ref}' matching '{query}'"
    return _format_package_search(ref, query, matches)


async def _flake_ref_search_options(ref: str, query: str, limit: int) -> str:
    """Search options across the flake's nixosModules and homeManagerModules."""
    system = await _current_system()
    ok, catalogue, err_msg = await _flake_ref_options_catalogue(ref, system)
    if not ok:
        return error(err_msg, "FLAKE_ERROR")
    if not catalogue:
        return f"No options found in flake '{ref}' matching '{query}'"

    matches: list[tuple[int, dict[str, str]]] = []
    for opt in catalogue:
        score = score_option_match(opt["name"], opt.get("description", ""), query)
        if score:
            matches.append((score, opt))
    matches.sort(key=lambda match: (-match[0], match[1]["name"].casefold()))
    matches = matches[:limit]

    if not matches:
        return f"No options found in flake '{ref}' matching '{query}'"
    results = [f"Found {len(matches)} options in flake '{ref}' matching '{query}':", ""]
    for _score, opt in matches:
        results.append(f"* {opt['name']}")
        results.append(f"  Module: {opt['kind']} ({opt['module']})")
        if opt.get("type"):
            results.append(f"  Type: {opt['type']}")
        if opt.get("description"):
            results.append(f"  {opt['description']}")
        results.append("")
    return "\n".join(results).strip()


async def _flake_ref_info(ref: str, query: str, info_type: str) -> str:
    """Get details for a package or option in an arbitrary flake."""
    if not _ATTR_PATH.match(query):
        return error(f"Invalid attribute/option path: {query!r}", "INVALID_FORMAT")
    if info_type == "package":
        return await _flake_ref_info_package(ref, query)
    return await _flake_ref_info_option(ref, query)


async def _flake_ref_info_package(ref: str, query: str) -> str:
    """Details for a package attr: name, description, store path, flake revision."""
    system = await _current_system()

    ok, show, err_msg = await _flake_ref_show(ref)
    if not ok:
        return error(err_msg, "FLAKE_ERROR")
    if query not in _flake_package_names(show):
        available = ", ".join(sorted(_flake_package_names(show))[:10])
        return error(
            f"Package '{query}' not found in flake '{ref}'. "
            f"Available (sample): {available}",
            "NOT_FOUND",
        )

    # NB: not a top-level `outPath` key — `nix eval` stringifies any resulting
    # attrset that has an `outPath` attribute down to the bare path, so the
    # extra name/description would be lost. `storePath` avoids that.
    detail_expr = 'p: { storePath = p.outPath; name = p.name or ""; description = p.meta.description or ""; }'
    success, stdout, stderr = await _run_nix(
        ["eval", "--impure", "--json", f"{ref}#packages.{system}.{query}", "--apply", detail_expr],
        _EVAL_TIMEOUT,
    )
    if not success:
        detail: dict[str, Any] = {}
    else:
        try:
            detail = json.loads(stdout) if isinstance(stdout, str) else {}
        except json.JSONDecodeError:
            detail = {}

    ok_meta, meta, _meta_err = await _flake_ref_metadata(ref)
    rev = meta.get("rev", "") if ok_meta and meta else ""
    last_modified = meta.get("lastModified", None) if ok_meta and meta else None

    results = [f"Package: {query}", f"Flake: {ref}"]
    if rev:
        results.append(f"Revision: {rev[:8]}")
    if last_modified:
        from datetime import UTC, datetime

        try:
            results.append(f"Updated: {datetime.fromtimestamp(int(last_modified), tz=UTC).strftime('%Y-%m-%d')}")
        except Exception:
            pass  # Omit the date rather than failing on a malformed timestamp
    if detail.get("name"):
        results.append(f"Name: {detail['name']}")
    if detail.get("description"):
        results.append(f"Description: {detail['description']}")
    if detail.get("storePath"):
        results.append(f"Store path: {detail['storePath']}")
    return "\n".join(results)


async def _flake_ref_info_option(ref: str, query: str) -> str:
    """Details for an option path defined by one of the flake's modules."""
    system = await _current_system()
    ok, catalogue, err_msg = await _flake_ref_options_catalogue(ref, system)
    if not ok:
        return error(err_msg, "FLAKE_ERROR")
    if catalogue is None:
        return error("Failed to build option catalogue", "FLAKE_ERROR")

    hits = [opt for opt in catalogue if opt["name"] == query]
    if not hits:
        suggestions = sorted(
            {opt["name"] for opt in catalogue if query in opt["name"]}, key=str.casefold
        )[:5]
        if suggestions:
            return f"Option '{query}' not found in flake '{ref}'. Similar: {', '.join(suggestions)}"
        return f"Option '{query}' not found in flake '{ref}'."

    results = [f"Option: {query}", f"Flake: {ref}"]
    for i, opt in enumerate(hits):
        if len(hits) > 1:
            results.append("")
            results.append(f"Defined by #{i + 1} ({opt['kind']}/{opt['module']}):")
        else:
            results.append(f"Module: {opt['kind']} ({opt['module']})")
        if opt.get("type"):
            results.append(f"Type: {opt['type']}")
        if opt.get("description"):
            results.append(f"Description: {opt['description']}")
    return "\n".join(results)


async def _flake_ref_browse(ref: str, query: str) -> str:
    """Walk the flake's combined module-option tree by prefix (or list categories)."""
    system = await _current_system()
    ok, catalogue, err_msg = await _flake_ref_options_catalogue(ref, system)
    if not ok:
        return error(err_msg, "FLAKE_ERROR")
    if catalogue is None:
        return error("Failed to build option catalogue", "FLAKE_ERROR")

    if not query:
        categories: dict[str, int] = {}
        for opt in catalogue:
            seg = opt["name"].split(".", 1)[0]
            categories[seg] = categories.get(seg, 0) + 1
        if not categories:
            return f"Flake '{ref}' exposes no module options to browse."
        results = [f"Flake '{ref}' option categories ({len(categories)} total):", ""]
        for cat, count in sorted(categories.items(), key=lambda item: (-item[1], item[0])):
            results.append(f"* {cat} ({count} options)")
        return "\n".join(results)

    matches = [opt for opt in catalogue if opt["name"] == query or opt["name"].startswith(query + ".")]
    if not matches:
        return f"No options found in flake '{ref}' with prefix '{query}'"
    results = [f"Flake '{ref}' options with prefix '{query}' ({len(matches):,} found):", ""]
    for opt in sorted(matches, key=lambda item: item["name"])[:_BROWSE_DISPLAY_LIMIT]:
        results.append(f"* {opt['name']}")
        if opt.get("type"):
            results.append(f"  Type: {opt['type']}")
        if opt.get("description"):
            results.append(f"  {opt['description']}")
        results.append("")
    if len(matches) > _BROWSE_DISPLAY_LIMIT:
        results.append(f"... and {len(matches) - _BROWSE_DISPLAY_LIMIT:,} more options")
    return "\n".join(results).strip()


async def _flake_ref_stats(ref: str) -> str:
    """Metadata counts for the flake (packages, module options, inputs)."""
    system = await _current_system()
    results = [f"Flake Statistics: {ref}", ""]

    ok_show, show, show_err = await _flake_ref_show(ref)
    if ok_show and show:
        packages = show.get("packages", {})
        if isinstance(packages, dict) and packages:
            results.append("Packages per system:")
            for sys_name, entries in sorted(packages.items()):
                if isinstance(entries, dict):
                    results.append(f"  {sys_name}: {len(entries):,}")
        results.append("")
        for output in ("checks", "devShells", "apps"):
            value = show.get(output, {})
            if isinstance(value, dict) and value:
                attrs = {
                    attr
                    for sys_entries in value.values()
                    if isinstance(sys_entries, dict)
                    for attr in sys_entries
                }
                results.append(f"{output.capitalize()}: {len(attrs)} attributes")
        results.append("")
    else:
        results.append(f"  Packages: unavailable ({show_err})")
        results.append("")

    ok_cat, catalogue, cat_err = await _flake_ref_options_catalogue(ref, system)
    if ok_cat and catalogue is not None:
        by_kind: dict[str, int] = {}
        for opt in catalogue:
            by_kind[opt["kind"]] = by_kind.get(opt["kind"], 0) + 1
        results.append(f"Module options: {len(catalogue):,}")
        for kind, count in sorted(by_kind.items()):
            results.append(f"  {kind}: {count:,}")
    else:
        results.append(f"  Module options: unavailable ({cat_err})")

    return "\n".join(results).strip()


async def _flake_ref_cache(ref: str, pkg: str, system: str) -> str:
    """Check the binary cache for one of the flake's package outputs."""
    if not _ATTR_PATH.match(pkg):
        return error(f"Invalid package name: {pkg!r}", "INVALID_FORMAT")
    sys_target = system or await _current_system()

    success, stdout, stderr = await _run_nix(
        ["eval", "--impure", "--raw", f"{ref}#packages.{sys_target}.{pkg}.outPath"], _EVAL_TIMEOUT
    )
    if not success:
        return error(
            f"Package '{pkg}' not found in flake '{ref}' for {sys_target}. {stderr.strip()}",
            "NOT_FOUND",
        )
    store_path = stdout.strip()
    if not store_path:
        return error("Failed to resolve package store path", "FLAKE_ERROR")

    lines = [f"Binary Cache Status: {pkg} ({ref})", ""]
    cache_lines = await asyncio.to_thread(_check_system_cache, {"system": sys_target, "store_path": store_path})
    lines.extend(cache_lines)
    return "\n".join(lines).strip()


async def _flake_ref_store(ref: str, query: str, op: str, limit: int) -> str:
    """List or read files in the materialized flake (or one of its inputs).

    Query grammar: ``""``/``.`` = flake root, ``sub/path`` = under the flake
    root, ``input:sub/path`` = inside a named input.
    """
    ok, root, inputs, err_msg = await _flake_ref_archive(ref)
    if not ok:
        return error(err_msg, "FLAKE_ERROR")
    if not root:
        return error("Flake produced no store path", "FLAKE_ERROR")

    if not query or query.strip() in ("", "."):
        target = root
    elif ":" in query:
        input_name, subpath = query.split(":", 1)
        if input_name not in inputs:
            available = ", ".join(sorted(inputs.keys())[:10])
            more = f" ... and {len(inputs) - 10} more" if len(inputs) > 10 else ""
            return error(f"Input '{input_name}' not found. Available: {available}{more}", "NOT_FOUND")
        target = os.path.join(inputs[input_name], subpath.lstrip("/"))
    else:
        target = os.path.join(root, query.lstrip("/"))

    if not _validate_store_path(target):
        return error("Invalid path: must stay within /nix/store/", "SECURITY_ERROR")

    if op == "ls":
        return await _store_ls(target, limit)
    return await _store_read(target, limit)
