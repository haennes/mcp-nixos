"""Tests for the arbitrary-flake data source (source=<flake-ref>).

Unit tests mock `server._run_nix_command` (via the getter used by the module)
and patch the process-local caches so each test is deterministic. The single
integration test at the bottom hits a real flake.
"""

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from mcp_nixos.server import DEFAULT_LINE_LIMIT, nix
from mcp_nixos.sources import flake_ref as flake_ref_module
from mcp_nixos.sources.flake_ref import (
    _current_system,
    _flake_available_systems,
    _flake_package_names,
    _flake_ref_archive,
    _flake_ref_browse,
    _flake_ref_cache,
    _flake_ref_info,
    _flake_ref_info_option,
    _flake_ref_metadata,
    _flake_ref_options_catalogue,
    _flake_ref_search_options,
    _flake_ref_search_packages,
    _flake_ref_show,
    _flake_ref_stats,
    _flake_ref_store,
    _flatten_catalogue,
    _validate_flake_ref,
)

nix_fn = getattr(nix, "fn", nix)

SHOW_PAYLOAD = {
    "packages": {
        "x86_64-linux": {"foo": {}, "bar": {}},
        "aarch64-linux": {"foo": {}, "baz": {}},
    }
}

OPTIONS_PAYLOAD = {
    "system": "x86_64-linux",
    "nixosModules": {
        "default": [
            {
                "name": "services.foo.enable",
                "type": "boolean",
                "description": "Enable the foo service.",
            }
        ]
    },
    "homeManagerModules": {
        "default": [
            {"name": "programs.bar.enable", "type": "boolean", "description": "Enable bar."}
        ]
    },
}

# The fingerprint probe is the store-path check that invalidates the caches. Its
# fake response uses a stable default so unchanged-tree tests reuse their cache
# entries; the "flake"/"lib" keys can be overridden via extra or the sequenced
# runner below to simulate the upstream flake (or its nixpkgs input) moving.
DEFAULT_FINGERPRINT = {"flake": "/nix/store/aaa-flake", "lib": "/nix/store/aaa-nixpkgs"}


@pytest.fixture(autouse=True)
def _clear_caches():
    flake_ref_module.flake_ref_cache._system = None
    flake_ref_module.flake_ref_cache._show.clear()
    flake_ref_module.flake_ref_cache._archives.clear()
    flake_ref_module.flake_ref_cache._options.clear()
    flake_ref_module.flake_ref_cache._lock = asyncio.Lock()
    yield


def _make_fake_run(extra: dict | None = None):
    """Return an async _run_nix_command stand-in dispatching on nix subcommands."""
    extra = extra or {}

    async def fake_run(args, cwd=None, timeout=60):
        joined = " ".join(args)
        if "flake show" in joined:
            return True, json.dumps(extra.get("show", SHOW_PAYLOAD)), ""
        if "flake metadata" in joined:
            return (
                True,
                json.dumps(
                    extra.get("metadata", {"rev": "abc1234abc1234", "lastModified": 1700000000})
                ),
                "",
            )
        if "flake archive" in joined:
            return True, json.dumps(extra.get("archive", {"path": "/nix/store/aaa", "inputs": {}})), ""
        if "--expr" in joined and "builtins.getFlake" in joined and "outPath" in joined:
            return True, json.dumps(extra.get("fingerprint", DEFAULT_FINGERPRINT)), ""
        if "builtins.currentSystem" in joined:
            return True, "x86_64-linux", ""
        if "--apply" in joined:
            return (
                True,
                json.dumps(
                    extra.get(
                        "detail",
                        {"storePath": "/nix/store/ccc-foo", "name": "foo", "description": "A foo"},
                    )
                ),
                "",
            )
        if "outPath" in joined:
            return True, "/nix/store/ccc-foo", ""
        if "--expr" in joined:
            return True, json.dumps(extra.get("options", OPTIONS_PAYLOAD)), ""
        return True, "", ""

    return fake_run


@contextmanager
def _patch_run(fake_run=None):
    """Patch nix availability + the command runner for the duration of a block."""
    runner = fake_run if fake_run is not None else _make_fake_run()
    with patch("mcp_nixos.sources.flake_ref._check_nix_available", return_value=True), patch(
        "mcp_nixos.server._run_nix_command", new=runner
    ):
        yield


def _make_counting_run(fingerprints: list[dict] | None = None, extra: dict | None = None):
    """Fake runner that sequences fingerprint probes and counts expensive evals.

    ``fingerprints`` is consumed one entry per fingerprint probe so tests can
    simulate the upstream flake (or its registered ``nixpkgs``) moving between
    calls. ``calls`` tallies the expensive step for each cache: ``show``,
    ``archive``, and ``options`` (the module-option eval).
    """
    fingerprints = fingerprints or []
    base = _make_fake_run(extra)
    index = 0
    calls = {"show": 0, "archive": 0, "options": 0}

    async def counting(args, cwd=None, timeout=60):
        nonlocal index
        joined = " ".join(args)
        if "--expr" in joined and "builtins.getFlake" in joined and "outPath" in joined:
            payload = fingerprints[index] if index < len(fingerprints) else DEFAULT_FINGERPRINT
            index += 1
            return True, json.dumps(payload), ""
        if "flake show" in joined:
            calls["show"] += 1
        elif "flake archive" in joined:
            calls["archive"] += 1
        elif "--expr" in joined and "builtins.currentSystem" not in joined:
            calls["options"] += 1
        return await base(args, cwd, timeout)

    return counting, calls


def _make_fragment_run(fragment: str, result: tuple[bool, str, str]):
    """Default fake runner, except a command containing ``fragment`` returns ``result``."""
    base = _make_fake_run()

    async def runner(args, cwd=None, timeout=60):
        if fragment in " ".join(args):
            return result
        return await base(args, cwd, timeout)

    return runner


def _make_fingerprint_run(result: tuple[bool, str, str]):
    """Default fake runner, except the store-path fingerprint probe returns ``result``."""
    base = _make_fake_run()

    async def runner(args, cwd=None, timeout=60):
        joined = " ".join(args)
        if "--expr" in joined and "builtins.getFlake" in joined and "outPath" in joined:
            return result
        return await base(args, cwd, timeout)

    return runner


async def _race_inner_lock(slow_fragment: str, work):
    """Run ``work()`` twice so the second call hits the post-lock cache re-check.

    The first call blocks while evaluating ``slow_fragment`` (holding the cache
    lock); the second call then clears its pre-lock check and blocks on the lock,
    so when the first finishes it exercises the double-checked single-flight
    return path inside the lock.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    base = _make_fake_run()

    async def runner(args, cwd=None, timeout=60):
        joined = " ".join(args)
        if slow_fragment in joined:
            started.set()
            await release.wait()
        return await base(args, cwd, timeout)

    with _patch_run(runner):
        first = asyncio.create_task(work())
        await started.wait()
        second = asyncio.create_task(work())
        await asyncio.sleep(0)
        release.set()
        r1, r2 = await asyncio.gather(first, second)
    return r1, r2


@pytest.mark.unit
class TestValidateFlakeRef:
    @pytest.mark.parametrize(
        "ref",
        [
            "nixpkgs",
            "home-manager",
            "impermanence",
            "github:owner/repo",
            "github:owner/repo/nixos-25.05",
            "gitlab:owner/repo",
            "gitea:owner/repo",
            "git+https://github.com/owner/repo",
            "https://github.com/owner/repo",
            "github:nix-community/home-manager?ref=master",
        ],
    )
    def test_accepts_remote_refs(self, ref):
        assert _validate_flake_ref(ref) is True

    @pytest.mark.parametrize(
        "ref",
        [
            "",
            "   ",
            "two words",
            "path:/tmp/flake",
            "file:/tmp/flake",
            "file:///tmp/flake",
            "./flake",
            "../flake",
            "/nix/store/abc",
            "~/flake",
            "C:\\flake",
            'github:"owner/repo"',
            "github:owner\\repo",
            "github:owner$repo",
        ],
    )
    def test_rejects_local_and_unsafe_refs(self, ref):
        assert _validate_flake_ref(ref) is False

    @pytest.mark.parametrize("ref", ["q:/foo", "Z:\\tmp", "n:/x"])
    def test_rejects_single_letter_drive_schemes(self, ref):
        assert _validate_flake_ref(ref) is False


@pytest.mark.unit
class TestFlakeRefHelpers:
    def test_flake_package_names_collects_across_systems(self):
        assert _flake_package_names(SHOW_PAYLOAD) == {"foo", "bar", "baz"}

    def test_flake_package_names_tolerates_non_dicts(self):
        assert _flake_package_names({"packages": ["x86_64-linux"]}) == set()

    def test_flake_package_names_tolerates_non_entries(self):
        assert _flake_package_names({"packages": {"x86_64-linux": "nope"}}) == set()

    def test_available_systems_sorted(self):
        assert _flake_available_systems(SHOW_PAYLOAD) == ["aarch64-linux", "x86_64-linux"]

    def test_available_systems_tolerates_non_dicts(self):
        assert _flake_available_systems({"packages": ["x86_64-linux"]}) == []

    def test_flatten_catalogue_skips_malformed_entries(self):
        data = {
            "system": "x86_64-linux",
            "nixosModules": "oops",
            "homeManagerModules": {
                "failed": {"error": "module failed to evaluate"},
                "strings": ["not-a-record"],
                "unnamed": [{"name": "", "type": "boolean"}],
            },
        }
        assert _flatten_catalogue(data) == []


@pytest.mark.unit
class TestFlakeRefSearch:
    @pytest.mark.asyncio
    async def test_search_packages(self):
        with _patch_run():
            result = await _flake_ref_search_packages("github:o/r", "foo", 20)
        assert "Found 1 packages" in result
        assert "* foo" in result
        assert "Attribute: foo" in result

    @pytest.mark.asyncio
    async def test_search_packages_no_matches(self):
        with _patch_run():
            result = await _flake_ref_search_packages("github:o/r", "zzzz", 20)
        assert "No packages found" in result

    @pytest.mark.asyncio
    async def test_search_packages_nix_unavailable(self):
        with patch("mcp_nixos.sources.flake_ref._check_nix_available", return_value=False):
            result = await _flake_ref_search_packages("github:o/r", "foo", 20)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_search_options(self):
        with _patch_run():
            result = await _flake_ref_search_options("github:o/r", "foo", 20)
        assert "services.foo.enable" in result
        assert "Module: nixos (default)" in result
        assert "Enable the foo service" in result

    @pytest.mark.asyncio
    async def test_search_options_flake_exposes_none(self):
        options = {"system": "x86_64-linux"}
        with _patch_run(_make_fake_run({"options": options})):
            result = await _flake_ref_search_options("github:o/r", "foo", 20)
        assert "No options found" in result


@pytest.mark.unit
class TestFlakeRefInfo:
    @pytest.mark.asyncio
    async def test_info_package(self):
        with _patch_run():
            result = await _flake_ref_info("github:o/r", "foo", "package")
        assert "Package: foo" in result
        assert "Flake: github:o/r" in result
        assert "Revision: abc1234" in result
        assert "Name: foo" in result
        assert "Description: A foo" in result
        assert "Store path: /nix/store/ccc-foo" in result

    @pytest.mark.asyncio
    async def test_info_package_not_found(self):
        extra = {
            "show": {
                "packages": {
                    "x86_64-linux": {"foo": {}},
                }
            }
        }
        with _patch_run(_make_fake_run(extra)):
            result = await _flake_ref_info("github:o/r", "nope", "package")
        assert "Package 'nope' not found" in result
        assert "foo" in result

    @pytest.mark.asyncio
    async def test_info_invalid_path_rejected(self):
        with _patch_run():
            result = await _flake_ref_info("github:o/r", "bad name../../", "package")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_info_option_single_module(self):
        with _patch_run():
            result = await _flake_ref_info("github:o/r", "services.foo.enable", "option")
        assert "Option: services.foo.enable" in result
        assert "Module: nixos (default)" in result
        assert "Type: boolean" in result

    @pytest.mark.asyncio
    async def test_info_option_defined_by_many_modules(self):
        options = {
            "system": "x86_64-linux",
            "nixosModules": {
                "a": [{"name": "services.foo.enable", "type": "boolean", "description": ""}],
                "b": [{"name": "services.foo.enable", "type": "boolean", "description": ""}],
            },
        }
        with _patch_run(_make_fake_run({"options": options})):
            result = await _flake_ref_info("github:o/r", "services.foo.enable", "option")
        assert "Defined by #1 (nixos/a)" in result
        assert "Defined by #2 (nixos/b)" in result

    @pytest.mark.asyncio
    async def test_info_option_not_found_suggests(self):
        options = {
            "system": "x86_64-linux",
            "nixosModules": {"default": [{"name": "services.foo.enable", "type": "boolean", "description": ""}]},
        }
        with _patch_run(_make_fake_run({"options": options})):
            result = await _flake_ref_info("github:o/r", "services.foo", "option")
        assert "Option 'services.foo' not found" in result
        assert "services.foo.enable" in result


@pytest.mark.unit
class TestFlakeRefBrowse:
    @pytest.mark.asyncio
    async def test_browse_categories(self):
        with _patch_run():
            result = await _flake_ref_browse("github:o/r", "")
        assert "option categories" in result
        assert "* services (1 options)" in result
        assert "* programs (1 options)" in result

    @pytest.mark.asyncio
    async def test_browse_prefix(self):
        with _patch_run():
            result = await _flake_ref_browse("github:o/r", "services")
        assert "with prefix 'services'" in result
        assert "services.foo.enable" in result


@pytest.mark.unit
class TestFlakeRefStats:
    @pytest.mark.asyncio
    async def test_stats(self):
        with _patch_run():
            result = await _flake_ref_stats("github:o/r")
        assert "Flake Statistics: github:o/r" in result
        assert "x86_64-linux: 2" in result
        assert "aarch64-linux: 2" in result
        assert "Module options: 2" in result
        assert "nixos: 1" in result

    @pytest.mark.asyncio
    async def test_stats_degrades_gracefully(self):
        with _patch_run(_make_fake_run({"show": {}, "options": {"system": "x86_64-linux"}})):
            result = await _flake_ref_stats("github:o/r")
        assert "Flake Statistics: github:o/r" in result
        assert "Module options: unavailable" not in result


@pytest.mark.unit
class TestFlakeRefCache:
    @pytest.mark.asyncio
    async def test_cache_happy_path(self):
        with _patch_run():
            with patch(
                "mcp_nixos.sources.flake_ref._check_system_cache",
                return_value=["x86_64-linux: yes"],
            ):
                result = await _flake_ref_cache("github:o/r", "foo", "")
        assert "Binary Cache Status: foo (github:o/r)" in result
        assert "x86_64-linux" in result

    @pytest.mark.asyncio
    async def test_cache_package_not_found(self):
        async def fake_run(args, cwd=None, timeout=60):
            if "builtins.currentSystem" in " ".join(args):
                return True, "x86_64-linux", ""
            return False, "", "error: attribute 'packages.x86_64-linux.nope' missing"

        with _patch_run(fake_run):
            result = await _flake_ref_cache("github:o/r", "nope", "")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_cache_invalid_name(self):
        with _patch_run():
            result = await _flake_ref_cache("github:o/r", "bad name/", "")
        assert "Error" in result


@pytest.mark.unit
class TestFlakeRefStore:
    async def _run_store(self, tmp_path, query, op, limit=100, root=None, inputs=None):
        root = root or str(tmp_path)
        with patch(
            "mcp_nixos.sources.flake_ref._flake_ref_archive",
            new=AsyncMock(return_value=(True, root, inputs or {}, "")),
        ):
            with patch("mcp_nixos.sources.flake_ref._validate_store_path", return_value=True):
                with patch("mcp_nixos.sources.store._validate_store_path", return_value=True):
                    return await _flake_ref_store("github:o/r", query, op, limit)

    @pytest.mark.asyncio
    async def test_store_ls_root_empty_query(self, tmp_path):
        (tmp_path / "README.md").write_text("hello\n")
        (tmp_path / "src").mkdir()
        result = await self._run_store(tmp_path, "", "ls")
        assert "Contents of" in result
        assert "README.md" in result
        assert "src/" in result

    @pytest.mark.asyncio
    async def test_store_ls_dot_is_root(self, tmp_path):
        (tmp_path / "README.md").write_text("hello\n")
        result = await self._run_store(tmp_path, ".", "ls")
        assert "README.md" in result

    @pytest.mark.asyncio
    async def test_store_ls_subpath(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.nix").write_text("{}")
        result = await self._run_store(tmp_path, "src", "ls")
        assert "main.nix" in result

    @pytest.mark.asyncio
    async def test_store_read_file(self, tmp_path):
        (tmp_path / "README.md").write_text("hello flakes\n")
        result = await self._run_store(tmp_path, "README.md", "read")
        assert "hello flakes" in result

    @pytest.mark.asyncio
    async def test_store_input_subpath(self, tmp_path):
        (tmp_path / "hello.nix").write_text("{}")
        inputs = {"nixpkgs": str(tmp_path)}
        result = await self._run_store(tmp_path, "nixpkgs:hello.nix", "read", inputs=inputs)
        assert "File:" in result

    @pytest.mark.asyncio
    async def test_store_unknown_input(self, tmp_path):
        result = await self._run_store(tmp_path, "no-such-input:foo", "ls")
        assert "Input 'no-such-input' not found" in result

    @pytest.mark.asyncio
    async def test_store_path_violation_rejected(self, tmp_path):
        with patch(
            "mcp_nixos.sources.flake_ref._flake_ref_archive",
            new=AsyncMock(return_value=(True, str(tmp_path), {}, "")),
        ):
            with patch("mcp_nixos.sources.flake_ref._validate_store_path", return_value=False):
                result = await _flake_ref_store("github:o/r", "", "ls", 100)
        assert "SECURITY_ERROR" in result


@pytest.mark.unit
class TestFlakeRefRouting:
    @pytest.mark.asyncio
    async def test_search_routes_by_type(self):
        with patch(
            "mcp_nixos.server._flake_ref_search_options", new=AsyncMock(return_value="opts")
        ) as m:
            result = await nix_fn(action="search", source="github:o/r", query="x", type="options")
        m.assert_called_once_with("github:o/r", "x", 20)
        assert result == "opts"

    @pytest.mark.asyncio
    async def test_search_invalid_type_rejected(self):
        result = await nix_fn(action="search", source="github:o/r", query="x", type="programs")
        assert "Error" in result
        assert "packages, options" in result

    @pytest.mark.asyncio
    async def test_info_defaults_to_package(self):
        with patch("mcp_nixos.server._flake_ref_info", new=AsyncMock(return_value="info")) as m:
            result = await nix_fn(action="info", source="gitlab:o/r", query="services.foo.enable")
        m.assert_called_once_with("gitlab:o/r", "services.foo.enable", "package")
        assert result == "info"

    @pytest.mark.asyncio
    async def test_browse_routes(self):
        with patch("mcp_nixos.server._flake_ref_browse", new=AsyncMock(return_value="b")) as m:
            result = await nix_fn(action="browse", source="nixpkgs", query="services")
        m.assert_called_once_with("nixpkgs", "services")
        assert result == "b"

    @pytest.mark.asyncio
    async def test_stats_routes(self):
        with patch("mcp_nixos.server._flake_ref_stats", new=AsyncMock(return_value="s")) as m:
            result = await nix_fn(action="stats", source="nixpkgs")
        m.assert_called_once_with("nixpkgs")
        assert result == "s"

    @pytest.mark.asyncio
    async def test_cache_routes(self):
        with patch("mcp_nixos.server._flake_ref_cache", new=AsyncMock(return_value="c")) as m:
            result = await nix_fn(action="cache", source="github:o/r", query="foo")
        m.assert_called_once_with("github:o/r", "foo", "")
        assert result == "c"

    @pytest.mark.asyncio
    async def test_store_empty_query_is_root_not_error(self):
        with patch("mcp_nixos.server._flake_ref_store", new=AsyncMock(return_value="root ls")) as m:
            result = await nix_fn(action="store", source="github:o/r", query="", type="ls")
        m.assert_called_once()
        assert m.call_args.args[:3] == ("github:o/r", "", "ls")
        assert result == "root ls"

    @pytest.mark.asyncio
    async def test_store_read_routes_with_default_promotion(self):
        with patch("mcp_nixos.server._flake_ref_store", new=AsyncMock(return_value="read")) as m:
            result = await nix_fn(action="store", source="github:o/r", query="README.md", type="read")
        m.assert_called_once()
        assert m.call_args.args == ("github:o/r", "README.md", "read", DEFAULT_LINE_LIMIT)
        assert result == "read"

    @pytest.mark.asyncio
    async def test_store_read_routes_explicit_limit(self):
        with patch("mcp_nixos.server._flake_ref_store", new=AsyncMock(return_value="read")) as m:
            result = await nix_fn(action="store", source="github:o/r", query="src/main.rs", type="read", limit=7)
        m.assert_called_once()
        assert m.call_args.args == ("github:o/r", "src/main.rs", "read", 7)
        assert result == "read"


@pytest.mark.unit
class TestFlakeRefStorePathCache:
    @pytest.mark.asyncio
    async def test_show_cached_until_flake_path_changes(self):
        runner, calls = _make_counting_run()
        with _patch_run(runner):
            r1 = await _flake_ref_search_packages("github:o/r", "foo", 20)
            r2 = await _flake_ref_search_packages("github:o/r", "foo", 20)
        assert calls["show"] == 1
        assert "Found 1 packages" in r1
        assert "Found 1 packages" in r2

    @pytest.mark.asyncio
    async def test_show_invalidated_on_path_change(self):
        f1 = {"flake": "/nix/store/flake-1", "lib": "/nix/store/lib-1"}
        f2 = {"flake": "/nix/store/flake-2", "lib": "/nix/store/lib-1"}
        runner, calls = _make_counting_run(fingerprints=[f1, f2])
        with _patch_run(runner):
            await _flake_ref_search_packages("github:o/r", "foo", 20)
            await _flake_ref_search_packages("github:o/r", "foo", 20)
        assert calls["show"] == 2

    @pytest.mark.asyncio
    async def test_archive_cached_until_flake_path_changes(self):
        runner, calls = _make_counting_run()
        with _patch_run(runner):
            ok1, root1, _inputs1, err1 = await _flake_ref_archive("github:o/r")
            ok2, root2, _inputs2, err2 = await _flake_ref_archive("github:o/r")
        assert calls["archive"] == 1
        assert ok1 and ok2
        assert root1 == root2 == "/nix/store/aaa"
        assert not err1 and not err2

    @pytest.mark.asyncio
    async def test_archive_invalidated_on_path_change(self):
        f1 = {"flake": "/nix/store/flake-1", "lib": "/nix/store/lib-1"}
        f2 = {"flake": "/nix/store/flake-2", "lib": "/nix/store/lib-1"}
        runner, calls = _make_counting_run(fingerprints=[f1, f2])
        with _patch_run(runner):
            await _flake_ref_archive("github:o/r")
            await _flake_ref_archive("github:o/r")
        assert calls["archive"] == 2

    @pytest.mark.asyncio
    async def test_options_cached_until_fingerprint_changes(self):
        runner, calls = _make_counting_run()
        with _patch_run(runner):
            r1 = await _flake_ref_search_options("github:o/r", "services.foo", 20)
            r2 = await _flake_ref_search_options("github:o/r", "services.foo", 20)
        assert calls["options"] == 1
        assert "services.foo.enable" in r1
        assert "services.foo.enable" in r2

    @pytest.mark.asyncio
    async def test_options_invalidated_on_flake_path_change(self):
        f1 = {"flake": "/nix/store/flake-1", "lib": "/nix/store/lib-1"}
        f2 = {"flake": "/nix/store/flake-2", "lib": "/nix/store/lib-1"}
        runner, calls = _make_counting_run(fingerprints=[f1, f2, f2])
        with _patch_run(runner):
            r1 = await _flake_ref_search_options("github:o/r", "services.foo", 20)
            r2 = await _flake_ref_search_options("github:o/r", "services.foo", 20)
            r3 = await _flake_ref_search_options("github:o/r", "services.foo", 20)
        assert calls["options"] == 2
        assert "services.foo.enable" in r1
        assert "services.foo.enable" in r2
        assert "services.foo.enable" in r3

    @pytest.mark.asyncio
    async def test_options_invalidated_on_registry_nixpkgs_change(self):
        # Same flake path but the registered nixpkgs moved (registry-fallback case).
        f1 = {"flake": "/nix/store/flake-1", "lib": "/nix/store/lib-1"}
        f2 = {"flake": "/nix/store/flake-1", "lib": "/nix/store/lib-2"}
        runner, calls = _make_counting_run(fingerprints=[f1, f2])
        with _patch_run(runner):
            await _flake_ref_search_options("github:o/r", "services.foo", 20)
            await _flake_ref_search_options("github:o/r", "services.foo", 20)
        assert calls["options"] == 2

    @pytest.mark.asyncio
    async def test_concurrent_options_misses_single_flight(self):
        runner, calls = _make_counting_run()
        with _patch_run(runner):
            r1, r2 = await asyncio.gather(
                _flake_ref_search_options("github:o/r", "services.foo", 20),
                _flake_ref_search_options("github:o/r", "services.foo", 20),
            )
        assert calls["options"] == 1
        assert "services.foo.enable" in r1
        assert "services.foo.enable" in r2


@pytest.mark.unit
class TestFlakeRefCacheEdgePaths:
    @pytest.mark.asyncio
    async def test_fingerprint_parse_failure(self):
        with _patch_run(_make_fingerprint_run((True, "not-json", ""))):
            ok, show, err = await _flake_ref_show("github:o/r")
        assert not ok
        assert show == {}
        assert "Failed to parse flake fingerprint output" in err

    @pytest.mark.asyncio
    async def test_fingerprint_no_store_path(self):
        with _patch_run(_make_fingerprint_run((True, json.dumps({"flake": ""}), ""))):
            ok, _show, err = await _flake_ref_show("github:o/r")
        assert not ok
        assert "Flake evaluation returned no store path" in err

    @pytest.mark.asyncio
    async def test_fingerprint_failure_propagates_to_all_caches(self):
        with _patch_run(_make_fingerprint_run((False, "", "fp boom"))):
            ok_a, _root, _inputs, err_a = await _flake_ref_archive("github:o/r")
            ok_o, _cat, err_o = await _flake_ref_options_catalogue("github:o/r", "x86_64-linux")
        assert not ok_a and not ok_o
        assert "fp boom" in err_a
        assert "fp boom" in err_o

    @pytest.mark.asyncio
    async def test_show_eval_failure(self):
        with _patch_run(_make_fragment_run("flake show", (False, "", "show boom"))):
            ok, _show, err = await _flake_ref_show("github:o/r")
        assert not ok
        assert "show boom" in err

    @pytest.mark.asyncio
    async def test_show_parse_failure(self):
        with _patch_run(_make_fragment_run("flake show", (True, "not-json", ""))):
            ok, _show, err = await _flake_ref_show("github:o/r")
        assert not ok
        assert "Failed to parse flake show output" in err

    @pytest.mark.asyncio
    async def test_archive_eval_failure(self):
        with _patch_run(_make_fragment_run("flake archive", (False, "", "archive boom"))):
            ok, _root, _inputs, err = await _flake_ref_archive("github:o/r")
        assert not ok
        assert "archive boom" in err

    @pytest.mark.asyncio
    async def test_archive_parse_failure(self):
        with _patch_run(_make_fragment_run("flake archive", (True, "not-json", ""))):
            ok, _root, _inputs, err = await _flake_ref_archive("github:o/r")
        assert not ok
        assert "Failed to parse archive output" in err

    @pytest.mark.asyncio
    async def test_archive_non_object(self):
        with _patch_run(_make_fragment_run("flake archive", (True, json.dumps([1]), ""))):
            ok, _root, _inputs, err = await _flake_ref_archive("github:o/r")
        assert not ok
        assert "was not an object" in err

    @pytest.mark.asyncio
    async def test_options_eval_failure(self):
        with _patch_run(_make_fragment_run("evalModules", (False, "", "eval boom"))):
            ok, _cat, err = await _flake_ref_options_catalogue("github:o/r", "x86_64-linux")
        assert not ok
        assert "eval boom" in err

    @pytest.mark.asyncio
    async def test_options_parse_failure(self):
        with _patch_run(_make_fragment_run("evalModules", (True, "not-json", ""))):
            ok, _cat, err = await _flake_ref_options_catalogue("github:o/r", "x86_64-linux")
        assert not ok
        assert "Failed to parse options output" in err

    @pytest.mark.asyncio
    async def test_options_unexpected_shape(self):
        with _patch_run(_make_fake_run({"options": []})):
            ok, _cat, err = await _flake_ref_options_catalogue("github:o/r", "x86_64-linux")
        assert not ok
        assert "unexpected shape" in err

    @pytest.mark.asyncio
    async def test_concurrent_current_system_rechecks_under_lock(self):
        r1, r2 = await _race_inner_lock("builtins.currentSystem", lambda: _current_system())
        assert r1 == "x86_64-linux"
        assert r2 == "x86_64-linux"

    @pytest.mark.asyncio
    async def test_concurrent_show_rechecks_under_lock(self):
        r1, r2 = await _race_inner_lock("flake show", lambda: _flake_ref_show("github:o/r"))
        assert r1[0] and r1[1].get("packages")
        assert r2[0] and r2[1].get("packages")

    @pytest.mark.asyncio
    async def test_concurrent_archive_rechecks_under_lock(self):
        r1, r2 = await _race_inner_lock("flake archive", lambda: _flake_ref_archive("github:o/r"))
        assert r1[0] and r1[1] == "/nix/store/aaa"
        assert r2[0] and r2[1] == "/nix/store/aaa"

    @pytest.mark.asyncio
    async def test_concurrent_options_rechecks_under_lock(self):
        r1, r2 = await _race_inner_lock(
            "evalModules", lambda: _flake_ref_search_options("github:o/r", "services.foo", 20)
        )
        assert "services.foo.enable" in r1
        assert "services.foo.enable" in r2


@pytest.mark.unit
class TestFlakeRefMetadata:
    @pytest.mark.asyncio
    async def test_metadata_failure(self):
        with _patch_run(_make_fragment_run("flake metadata", (False, "", "meta boom"))):
            ok, meta, err = await _flake_ref_metadata("github:o/r")
        assert not ok
        assert meta is None
        assert "meta boom" in err

    @pytest.mark.asyncio
    async def test_metadata_parse_failure(self):
        with _patch_run(_make_fragment_run("flake metadata", (True, "not-json", ""))):
            ok, _meta, err = await _flake_ref_metadata("github:o/r")
        assert not ok
        assert "Failed to parse metadata output" in err


@pytest.mark.unit
class TestFlakeRefActionEdgePaths:
    @pytest.mark.asyncio
    async def test_search_options_eval_failure(self):
        with _patch_run(_make_fragment_run("evalModules", (False, "", "eval boom"))):
            result = await _flake_ref_search_options("github:o/r", "foo", 20)
        assert "FLAKE_ERROR" in result
        assert "eval boom" in result

    @pytest.mark.asyncio
    async def test_search_options_no_matching_results(self):
        with _patch_run():
            result = await _flake_ref_search_options("github:o/r", "zzzz", 20)
        assert "No options found in flake 'github:o/r' matching 'zzzz'" in result

    @pytest.mark.asyncio
    async def test_info_package_show_failure(self):
        with _patch_run(_make_fragment_run("flake show", (False, "", "show boom"))):
            result = await _flake_ref_info("github:o/r", "foo", "package")
        assert "FLAKE_ERROR" in result
        assert "show boom" in result

    @pytest.mark.asyncio
    async def test_info_package_detail_eval_failure(self):
        with _patch_run(_make_fragment_run("--apply", (False, "", "detail boom"))):
            result = await _flake_ref_info("github:o/r", "foo", "package")
        assert "Package: foo" in result
        assert "Store path:" not in result

    @pytest.mark.asyncio
    async def test_info_package_detail_parse_failure(self):
        with _patch_run(_make_fragment_run("--apply", (True, "not-json", ""))):
            result = await _flake_ref_info("github:o/r", "foo", "package")
        assert "Package: foo" in result
        assert "Name:" not in result
        assert "Store path:" not in result

    @pytest.mark.asyncio
    async def test_info_package_malformed_date(self):
        extra = {"metadata": {"rev": "abc1234abc1234", "lastModified": "not-a-number"}}
        with _patch_run(_make_fake_run(extra)):
            result = await _flake_ref_info("github:o/r", "foo", "package")
        assert "Package: foo" in result
        assert "Updated:" not in result

    @pytest.mark.asyncio
    async def test_info_option_eval_failure(self):
        with _patch_run(_make_fragment_run("evalModules", (False, "", "eval boom"))):
            result = await _flake_ref_info("github:o/r", "services.foo.enable", "option")
        assert "FLAKE_ERROR" in result
        assert "eval boom" in result

    @pytest.mark.asyncio
    async def test_info_option_catalogue_none(self):
        with patch(
            "mcp_nixos.sources.flake_ref._current_system",
            new=AsyncMock(return_value="x86_64-linux"),
        ), patch(
            "mcp_nixos.sources.flake_ref._flake_ref_options_catalogue",
            new=AsyncMock(return_value=(True, None, "")),
        ):
            result = await _flake_ref_info_option("github:o/r", "services.foo.enable")
        assert "Failed to build option catalogue" in result

    @pytest.mark.asyncio
    async def test_info_option_not_found_no_suggestions(self):
        with _patch_run():
            result = await _flake_ref_info("github:o/r", "zzzzz", "option")
        assert "Option 'zzzzz' not found in flake 'github:o/r'." in result

    @pytest.mark.asyncio
    async def test_browse_eval_failure(self):
        with _patch_run(_make_fragment_run("evalModules", (False, "", "eval boom"))):
            result = await _flake_ref_browse("github:o/r", "services")
        assert "FLAKE_ERROR" in result
        assert "eval boom" in result

    @pytest.mark.asyncio
    async def test_browse_catalogue_none(self):
        with patch(
            "mcp_nixos.sources.flake_ref._current_system",
            new=AsyncMock(return_value="x86_64-linux"),
        ), patch(
            "mcp_nixos.sources.flake_ref._flake_ref_options_catalogue",
            new=AsyncMock(return_value=(True, None, "")),
        ):
            result = await _flake_ref_browse("github:o/r", "services")
        assert "Failed to build option catalogue" in result

    @pytest.mark.asyncio
    async def test_browse_no_options_to_browse(self):
        with _patch_run(_make_fake_run({"options": {"system": "x86_64-linux"}})):
            result = await _flake_ref_browse("github:o/r", "")
        assert "exposes no module options to browse" in result

    @pytest.mark.asyncio
    async def test_browse_no_matches(self):
        with _patch_run():
            result = await _flake_ref_browse("github:o/r", "zzzz")
        assert "No options found in flake 'github:o/r' with prefix 'zzzz'" in result

    @pytest.mark.asyncio
    async def test_browse_truncates_long_list(self):
        options = {
            "system": "x86_64-linux",
            "nixosModules": {
                "default": [
                    {"name": f"services.svc{i:03d}.enable", "type": "boolean", "description": ""}
                    for i in range(105)
                ]
            },
        }
        with _patch_run(_make_fake_run({"options": options})):
            result = await _flake_ref_browse("github:o/r", "services")
        assert "options with prefix 'services' (105 found)" in result
        assert "... and 5 more options" in result

    @pytest.mark.asyncio
    async def test_stats_counts_other_outputs(self):
        show = dict(SHOW_PAYLOAD)
        show["checks"] = {"x86_64-linux": {"build": {}}}
        show["devShells"] = {"x86_64-linux": {"default": {}}, "aarch64-linux": {"default": {}}}
        show["apps"] = {"x86_64-linux": {"cli": {}}}
        with _patch_run(_make_fake_run({"show": show})):
            result = await _flake_ref_stats("github:o/r")
        assert "Checks: 1 attributes" in result
        assert "Devshells: 1 attributes" in result
        assert "Apps: 1 attributes" in result

    @pytest.mark.asyncio
    async def test_stats_options_unavailable(self):
        with _patch_run(_make_fragment_run("evalModules", (False, "", "eval boom"))):
            result = await _flake_ref_stats("github:o/r")
        assert "Module options: unavailable (eval boom)" in result

    @pytest.mark.asyncio
    async def test_cache_empty_store_path(self):
        async def runner(args, cwd=None, timeout=60):
            joined = " ".join(args)
            if "outPath" in joined and "--expr" not in joined:
                return True, "", ""
            return await _make_fake_run()(args, cwd, timeout)

        with _patch_run(runner):
            result = await _flake_ref_cache("github:o/r", "foo", "")
        assert "Failed to resolve package store path" in result

    @pytest.mark.asyncio
    async def test_store_archive_failure(self):
        with _patch_run(_make_fragment_run("flake archive", (False, "", "archive boom"))):
            result = await _flake_ref_store("github:o/r", "", "ls", 100)
        assert "FLAKE_ERROR" in result
        assert "archive boom" in result

    @pytest.mark.asyncio
    async def test_store_empty_root(self):
        async def runner(args, cwd=None, timeout=60):
            joined = " ".join(args)
            if "flake archive" in joined:
                return True, json.dumps({"path": "", "inputs": {}}), ""
            return await _make_fake_run()(args, cwd, timeout)

        with _patch_run(runner):
            result = await _flake_ref_store("github:o/r", "", "ls", 100)
        assert "Flake produced no store path" in result


@pytest.mark.integration
@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_integration_search_packages_real_flake():
    """Live check against a real flake (requires nix locally).

    Regression for `--impure` placement: nix rejects `--impure eval` with
    ``unrecognised flag '--impure'``; the flag must follow the subcommand
    (``nix eval --impure ...``). A broken invocation surfaces as a FLAKE_ERROR
    here, so we assert on the success marker rather than mere non-emptiness.
    """
    result = await _flake_ref_search_packages("github:nix-community/home-manager", "xdg", 5)
    assert isinstance(result, str)
    assert "FLAKE_ERROR" not in result
    assert len(result) > 0


@pytest.mark.integration
@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_integration_info_package_real_flake():
    """Live details for a real flake package (covers the `--apply` eval path)."""
    result = await _flake_ref_info("github:nix-community/home-manager", "home-manager", "package")
    assert isinstance(result, str)
    assert "FLAKE_ERROR" not in result
    assert "Package: home-manager" in result
