#!/usr/bin/env python3
"""Build only the reviewed tutorial sources, then audit the complete publish directory.

No original _freeze, _site, or superseded fitted assets enter the build.
Only the Python standard library and Quarto are needed. An existing output is never
overwritten. This checks the publication boundary; it does not certify model fits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "final-product"
PAGES = ("index", "data", "xgboostlss", "pytorch", "comparison", "discussion")
RESULT_BLOCKS = {"xgboostlss.qmd": "MODEL", "pytorch.qmd": "MODEL",
                 "comparison.qmd": "COMPARISON", "discussion.qmd": "VALIDATION"}
HTML_PAGES = {f"{name}.html" for name in PAGES}
RESULT_SUMMARY = "assets/forecast/summary.json"
FORECAST_FIGURES = ("assets/forecast/train-test-diagnostics.png",
                   "assets/forecast/validation-diagnostics.png",
                   "assets/forecast/xgb-train-G1_stacf.png",
                   "assets/forecast/gnn-train-G1_stacf.png",
                   "assets/forecast/test-discrimination.png",
                   "assets/forecast/validation-discrimination.png")
MAP_RESOURCES = ("assets/forecast/validation-maps.js", "assets/forecast/maps/index.json",
                 "assets/forecast/maps/counties.json",
                 *(f"assets/forecast/maps/horizon-{h:02d}.bin" for h in range(1, 13)))
REPRO_RESOURCES = ("assets/reproduction/manifest.json", "assets/reproduction/tutorial-source.zip",
                   "assets/reproduction/xgb-predictions.zip", "assets/reproduction/gnn-predictions.zip")
RESOURCES = ("styles.css", "references.bib", "assets/fig/gcn_lstm_mechanism.gif",
             RESULT_SUMMARY, *FORECAST_FIGURES, *MAP_RESOURCES, *REPRO_RESOURCES)
INPUTS = ("_quarto.yml", *(f"{name}.qmd" for name in PAGES), *RESOURCES)
MODEL_DIRS = {"xgb_forecast_safe_20260907": "output/xgb_forecast_safe_20260907",
              "gnn_forecast_safe_20260907": "output/model/forecast_safe_20260907"}
EXPECTED_ROWS = {"train": 4699296, "test": 559440, "validation": 1566432}
TERRAIN = {"aspect_cos", "aspect_sin", "elev", "slope"}
SITE_URL = "https://bendevries.github.io/continental-us-counties-fire/"
QUARTO_FALLBACK = "/usr/lib/rstudio/resources/app/bin/quarto/bin/quarto"
SITE_ROOT_FILES = HTML_PAGES | {"styles.css", "search.json", "sitemap.xml", "robots.txt"}
LIBRARY_EXTENSIONS = {".js", ".css", ".woff", ".woff2", ".ttf", ".eot", ".svg", ".map"}
# These identify known withheld evidence, including copied output hidden in comments
# or search records. The file allowlist also blocks arbitrary old fitted assets.
WITHHELD = re.compile(
    r"(?:forecast-selection|conditional-selection|inla)\.(?:qmd|html)"
    r"|assets/sim/|sim_fitted_|sim_(?:xgb|torch|inla)_(?:cells|window|forecast|fitted)"
    r"|_freeze/|output/(?!xgb_forecast_safe_20260907|model/forecast_safe_20260907)"
    r"|fig-(?:fc-|cond-|xgb-(?:fc|cond)-|torch-(?:fc|cond)-|inla-fc-)"
    r"|-0\.(?:24189|23854|21334|21197|248510|242308)\b"
    r"|\boverall winner\b|wins it honestly|SIGNIFICANCE DIFFERS",
    re.IGNORECASE,
)


class ReleaseError(ValueError):
    """A publication boundary or local resource check failed."""


def check_text(text: str, label: str) -> None:
    match = WITHHELD.search(unescape(text).replace("−", "-"))
    if match:
        raise ReleaseError(f"{label}: withheld content {match.group(0)!r}")


def check_result_summary(path: Path, source_blocks: bool = False) -> None:
    """Admit only the newly regenerated forecast examples, without prediction bands."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("experiment") != "forecast_safe_20260907" or data.get("prediction_intervals") is not False:
        raise ReleaseError("Result summary must describe the new forecast experiment without prediction intervals")
    specs = data.get("model_specs", {})
    if set(specs) != set(MODEL_DIRS) or set(data.get("results", {})) != set(MODEL_DIRS):
        raise ReleaseError("Result summary model allowlist mismatch")
    for model, directory in MODEL_DIRS.items():
        spec = specs[model]
        cfg = spec.get("config", {})
        if (spec.get("directory") != directory or cfg.get("forecast_safe") is not True
                or cfg.get("lookback") != 48 or cfg.get("horizon") != 12):
            raise ReleaseError(f"{model}: wrong forecast configuration or source")
        if model.startswith("xgb"):
            if (set(cfg.get("raw_cov_names", [])) != TERRAIN
                    or cfg.get("climatology") is not False or cfg.get("clim_oof") is not False
                    or cfg.get("n_nbr_pcs") != 0 or cfg.get("lc_categories") != [0]
                    or cfg.get("reduction") != "none"):
                raise ReleaseError("XGBoost result includes a forbidden predictor path")
        elif (cfg.get("n_pca") is not None or cfg.get("static_bypass") is not False
              or cfg.get("teacher_forcing") is not False or cfg.get("lc_embed_dim") != 0):
            raise ReleaseError("GNN result includes a forbidden predictor path")
        splits = data["results"][model]
        if set(splits) != set(EXPECTED_ROWS):
            raise ReleaseError(f"{model}: missing result split")
        for split, count in EXPECTED_ROWS.items():
            result = splits[split]
            if result.get("rows") != count:
                raise ReleaseError(f"{model}/{split}: incorrect shared row count")
            for score in ("mean_nll", "mean_crps"):
                if type(result.get(score)) not in (int, float) or not math.isfinite(result[score]):
                    raise ReleaseError(f"{model}/{split}: missing or nonfinite {score}")
            if {"beta_coverage", "max_beta_coverage_deviation", "q05", "q95", "prediction_quantiles"} & set(result):
                raise ReleaseError("Prediction intervals/coverage are outside this release")
    for field in ("protocol_sha256", "input_audit_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", data.get(field, "")):
            raise ReleaseError(f"Missing forecast provenance {field}")
    for split, count in EXPECTED_ROWS.items():
        match = data.get("matched_evaluation_rows", {}).get(split, {})
        if match.get("exact_full_keys_and_outcomes") is not True or match.get("rows") != count:
            raise ReleaseError(f"Unmatched forecast evidence on {split}")
    sources = {item.get("path"): item.get("sha256", "") for item in data.get("sources", [])}
    required = [f"{directory}/predictions_{split}.parquet" for directory in MODEL_DIRS.values()
                for split in EXPECTED_ROWS]
    required += [f"{directory}/config.json" for directory in MODEL_DIRS.values()]
    if any(not re.fullmatch(r"[0-9a-f]{64}", sources.get(name, "")) for name in required):
        raise ReleaseError("Forecast evidence lacks required prediction/config source hashes")
    base = path.parents[2]
    assets = data.get("assets_sha256", {})
    if set(assets) != set(FORECAST_FIGURES):
        raise ReleaseError("Forecast figure hash allowlist mismatch")
    for relative, expected in assets.items():
        figure = base / relative
        if not figure.is_file() or hashlib.sha256(figure.read_bytes()).hexdigest() != expected:
            raise ReleaseError(f"Stale or missing forecast figure: {relative}")
    hashes = data.get("result_blocks_sha256", {})
    if set(hashes) != set(RESULT_BLOCKS) or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()):
        raise ReleaseError("Forecast result-block hash allowlist mismatch")
    if source_blocks:
        for page, kind in RESULT_BLOCKS.items():
            text = (base / page).read_text(encoding="utf-8")
            begin, end = f"<!-- BEGIN FORECAST {kind} RESULTS -->", f"<!-- END FORECAST {kind} RESULTS -->"
            if text.count(begin) != 1 or text.count(end) != 1:
                raise ReleaseError(f"Missing/duplicate generated result marker: {page}")
            body = text.split(begin, 1)[1].split(end, 1)[0].strip()
            if hashlib.sha256(body.encode("utf-8")).hexdigest() != hashes[page]:
                raise ReleaseError(f"Stale generated result block: {page}")


def check_supplements(base: Path) -> None:
    index = json.loads((base / "assets/forecast/maps/index.json").read_text())
    if index.get("shape") != [42, 5, 3108] or index.get("fields") != [
            "xgb_p_occ", "xgb_mu", "observed_fraction", "gnn_p_occ", "gnn_mu"]:
        raise ReleaseError("Unexpected validation map layout")
    geometry = base / "assets/forecast/maps/counties.json"
    if (not geometry.is_file()
            or hashlib.sha256(geometry.read_bytes()).hexdigest() != index.get("geometry_sha256")):
        raise ReleaseError("Missing or changed validation county geometry")
    horizons = index.get("horizons", [])
    if [entry.get("horizon") for entry in horizons] != list(range(1, 13)):
        raise ReleaseError("Missing validation map horizon")
    for entry in horizons:
        name = f"horizon-{entry['horizon']:02d}.bin"
        path = base / "assets/forecast/maps" / name
        if (entry.get("file") != name or not path.is_file()
                or path.stat().st_size != 42 * 5 * 3108 * 4
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256")):
            raise ReleaseError(f"Missing or changed validation map: {name}")
    manifest = json.loads((base / REPRO_RESOURCES[0]).read_text())
    archives = manifest.get("archives", {})
    if set(archives) != {Path(name).name for name in REPRO_RESOURCES[1:]}:
        raise ReleaseError("Reproduction archive allowlist mismatch")
    for name, record in archives.items():
        path = base / "assets/reproduction" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            raise ReleaseError(f"Missing or changed reproduction archive: {name}")
    with zipfile.ZipFile(base / "assets/reproduction/tutorial-source.zip") as archive:
        # The downloadable chapters and display code must describe this rendered edition.
        shared = ("_quarto.yml", "styles.css", "assets/forecast/validation-maps.js", "assets/forecast/maps/counties.json")
        shared += tuple(f"{page}.qmd" for page in PAGES)
        for relative in shared:
            local = base / relative
            if local.is_file() and archive.read(f"tutorial-writeup/final-product/{relative}") != local.read_bytes():
                raise ReleaseError(f"Reproduction source differs from this edition: {relative}")


def check_sources(source: Path) -> None:
    for relative in INPUTS:
        path = source / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ReleaseError(f"Missing or symlinked release input: {path}")
        if path.suffix not in {".qmd", ".yml", ".css"}:
            continue
        text = path.read_text(encoding="utf-8")
        check_text(text, relative)
        if path.suffix in {".qmd", ".yml"}:
            if re.search(r"^\s*(?:`{3,}|~{3,})\s*\{", text, re.MULTILINE):
                raise ReleaseError(f"{relative}: executable code fence in prose-only release")
            if re.search(r"^\s*(?:engine|jupyter|knitr)\s*:", text, re.MULTILINE):
                raise ReleaseError(f"{relative}: execution engine in prose-only release")
            if re.search(r"^\s*freeze\s*:\s*(?:true|auto)\b", text, re.MULTILINE):
                raise ReleaseError(f"{relative}: frozen execution is forbidden in this release")
            if "{{<" in text:
                raise ReleaseError(f"{relative}: unreviewed shortcode/include")
    check_result_summary(source / RESULT_SUMMARY, source_blocks=True)
    check_supplements(source)


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name == "id" or (tag == "a" and name == "name"):
                self.ids.add(value)
            if name in {"src", "href", "poster"} or (tag == "object" and name == "data"):
                self.references.append(value)
            if name == "srcset" and not value.lstrip().startswith("data:"):
                self.references.extend(item.strip().split()[0] for item in value.split(",") if item.strip())


def local_target(site: Path, owner: Path, reference: str) -> tuple[Path, str] | None:
    parsed = urlsplit(reference.strip())
    canonical = urlsplit(SITE_URL)
    if parsed.netloc:
        if parsed.netloc.lower() != canonical.netloc or not parsed.path.startswith(canonical.path):
            return None
        relative = parsed.path[len(canonical.path):]
        target = site / unquote(relative)
    elif parsed.scheme:
        return None
    elif parsed.path.startswith("/"):
        relative = parsed.path
        if relative.startswith(canonical.path):
            relative = relative[len(canonical.path):]
        target = site / unquote(relative.lstrip("/"))
    elif not parsed.path:
        target = owner
    else:
        target = owner.parent / unquote(parsed.path)
    target = target.resolve()
    if not target.is_relative_to(site.resolve()):
        raise ReleaseError(f"{owner.name}: reference escapes publish directory: {reference}")
    if target.is_dir():
        target /= "index.html"
    return target, unquote(parsed.fragment)


def check_site(site: Path) -> dict[str, int]:
    site = site.resolve()
    if not site.is_dir():
        raise ReleaseError(f"No rendered site: {site}")
    files = [path for path in site.rglob("*") if path.is_file()]
    for path in site.rglob("*"):
        if path.is_symlink():
            raise ReleaseError(f"Symlink in publish directory: {path}")
    actual_pages = {path.relative_to(site).as_posix() for path in files if path.suffix == ".html"}
    if actual_pages != HTML_PAGES:
        raise ReleaseError(f"HTML allowlist mismatch: missing={sorted(HTML_PAGES - actual_pages)}, extra={sorted(actual_pages - HTML_PAGES)}")
    parsers: dict[Path, PageParser] = {}
    references: list[tuple[Path, str]] = []
    for path in files:
        relative = path.relative_to(site).as_posix()
        is_library = relative.startswith("site_libs/") and path.suffix in LIBRARY_EXTENSIONS
        if relative not in SITE_ROOT_FILES and relative not in RESOURCES[2:] and not is_library:
            raise ReleaseError(f"Unapproved published resource: {relative}")
        if path.suffix in {".html", ".json", ".xml", ".txt", ".css", ".js", ".svg", ".map"}:
            text = path.read_text(encoding="utf-8")
            if relative != RESULT_SUMMARY:
                check_text(text, relative)
            else:
                check_result_summary(path)
        if path.suffix == ".html":
            parser = PageParser()
            parser.feed(text)
            parsers[path] = parser
            references.extend((path, reference) for reference in parser.references)
            if re.search(r"class=[\"'][^\"']*quarto-unresolved-ref", text):
                raise ReleaseError(f"Unresolved Quarto cross-reference: {relative}")
        if path.suffix == ".css":
            references.extend((path, item.strip(" \t\n\"'")) for item in re.findall(r"url\(([^)]+)\)", text))
    if not (site / RESULT_SUMMARY).is_file():
        raise ReleaseError("Missing regenerated forecast summary")
    check_supplements(site)
    search_path = site / "search.json"
    if not search_path.is_file():
        raise ReleaseError("Missing search.json")
    entries = json.loads(search_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ReleaseError("search.json must contain page records")
    search_pages = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("href"), str):
            raise ReleaseError("Malformed search record")
        target = local_target(site, search_path, entry["href"])
        if target is None or target[0].name not in HTML_PAGES:
            raise ReleaseError(f"Unapproved search target: {entry['href']}")
        search_pages.add(target[0].name)
        references.append((search_path, entry["href"]))
    if search_pages != HTML_PAGES:
        raise ReleaseError(f"Search index missing pages: {sorted(HTML_PAGES - search_pages)}")
    sitemap = site / "sitemap.xml"
    if sitemap.is_file():
        for element in ET.parse(sitemap).iter():
            if element.tag.rsplit("}", 1)[-1] == "loc" and element.text:
                references.append((sitemap, element.text))
    checked = 0
    for owner, reference in references:
        target = local_target(site, owner, reference)
        if target is None:
            continue
        path, fragment = target
        if not path.is_file():
            raise ReleaseError(f"{owner.relative_to(site)}: missing local target {reference!r}")
        if fragment and path.suffix == ".html" and fragment not in parsers[path].ids:
            raise ReleaseError(f"{owner.relative_to(site)}: missing fragment {reference!r}")
        checked += 1
    return {"pages": len(actual_pages), "files": len(files), "local_references": checked}


def find_quarto(explicit: str | None) -> str:
    quarto = explicit or shutil.which("quarto")
    if quarto is None and Path(QUARTO_FALLBACK).is_file():
        quarto = QUARTO_FALLBACK
    if quarto is None:
        raise ReleaseError("Quarto is required; install Quarto 1.9.38 or pass --quarto PATH")
    return quarto


def check_manifest(site: Path) -> None:
    """If this is a staged release, reject changes since its successful build."""
    manifest_path = site.parent / "release-manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {path.relative_to(site).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in site.rglob("*") if path.is_file()}
    if actual != manifest.get("published_sha256"):
        raise ReleaseError("Published files differ from the checked release manifest; rebuild into a new directory")
    for relative in INPUTS:
        path = site.parent / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("inputs_sha256", {}).get(relative):
            raise ReleaseError(f"Staged input differs from the checked release manifest: {relative}")


def build(output: Path | None, quarto: str | None) -> Path:
    check_sources(SOURCE)
    executable = find_quarto(quarto)
    if output is not None and output.exists():
        raise ReleaseError(f"Output already exists; choose a new directory: {output}")
    with tempfile.TemporaryDirectory(prefix="burned-area-tutorial-") as temporary:
        stage = Path(temporary)
        for relative in INPUTS:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / relative, target)
        check_sources(stage)
        subprocess.run([executable, "render", str(stage), "--no-execute", "--output-dir", "_site"], check=True)
        report = check_site(stage / "_site")
        if output is None:
            parent = HERE / "release-builds"
            parent.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = Path(tempfile.mkdtemp(prefix=f"release-{stamp}-", dir=parent))
        else:
            output.mkdir(parents=True, exist_ok=False)
        # Retain a minimal Quarto project under the repository for gh-pages' Git
        # discovery; publish --no-render copies only its audited _site directory.
        for relative in INPUTS:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage / relative, target)
        shutil.copytree(stage / "_site", output / "_site")
        manifest = {
            "scope": "Fresh leakage-safe XGBoost/GNN forecasts; no INLA, conditional results or prediction intervals",
            "quarto": subprocess.check_output([executable, "--version"], text=True).strip(),
            "checks": report,
            "inputs_sha256": {relative: hashlib.sha256((stage / relative).read_bytes()).hexdigest() for relative in INPUTS},
            "published_sha256": {path.relative_to(output / "_site").as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((output / "_site").rglob("*")) if path.is_file()},
        }
        (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Checked {report['pages']} pages and {report['local_references']} local references.")
        print(f"Staged project: {output.resolve()}")
        print(f"Publish directory: {(output / '_site').resolve()}")
        return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new staged-project directory; HTML goes in its _site child")
    parser.add_argument("--quarto", help="Quarto executable (automatically discovered when omitted)")
    parser.add_argument("--check", type=Path, metavar="SITE", help="audit an existing rendered _site directory without building")
    args = parser.parse_args()
    try:
        if args.check:
            report = check_site(args.check)
            check_manifest(args.check)
            print(json.dumps(report, indent=2))
        else:
            build(args.output, args.quarto)
    except (ReleaseError, OSError, subprocess.CalledProcessError, json.JSONDecodeError,
            ET.ParseError, zipfile.BadZipFile, KeyError) as error:
        print(f"Release check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
