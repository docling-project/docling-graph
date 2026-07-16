"""
OWL/RDFS/SKOS ontology -> TemplateSpec draft compiler (rdflib).

A pure compiler (design §5.1): BFS from the root class over object-property
ranges, bounded by ``depth``/``include``/``exclude``, then a fixed mapping
table turns each construct into draft models/fields/enums. **Zero LLM calls.**

Mapping summary (each row a separable branch, unit-tested in
``tests/unit/templategen/test_ontology_owl.py``):

- ``owl:Class``/``rdfs:Class`` -> ModelSpec; ``rdfs:comment``/``skos:definition``
  -> docstring (raw — the linter budgets to 240); ``rdfs:label`` -> "Also called".
- ``rdfs:subClassOf`` -> flattened: children copy ancestor properties; abstract
  parents (no direct properties, >= 2 children) are dropped with edges fanned
  out per child; siblings get "NOT a <sibling>" IS-NOT docstring clauses.
- ``owl:DatatypeProperty`` -> property field via :data:`XSD_SCALAR_MAP`.
- ``owl:ObjectProperty`` -> edge field with ``normalize_edge_label(localname)``.
- ``owl:FunctionalProperty`` -> single-valued; everything else defaults to a
  list (open-world: repeatable unless restricted).
- ``owl:Restriction`` max(Qualified)Cardinality: n=1 -> single optional;
  n>1 -> list + the documented ``max_instances = n`` on the target (the
  linter's ``repair_draft`` doubles exactly once) + "At most n per document."
  docstring sentence. ``minCardinality >= 1`` -> description note only, never
  a required field.
- ``owl:hasKey`` -> identity (first 2, gap beyond); datatype
  ``owl:InverseFunctionalProperty`` -> identity; else the heuristic ladder
  (preferring ``dcterms:identifier``/``schema:identifier``); else demotion to
  component + ``missing_identity`` gap. Identity-less root -> synthesized
  ``document_reference`` + gap.
- ``owl:oneOf`` / SKOS concept schemes -> EnumSpec (prefLabel members,
  altLabel synonyms; always ``include_other=True`` — the generated normalizer
  must never reject, R17); SKOS-only input raises :class:`SkosOnlyOntologyError`.
- Class reachable from >= 2 object properties -> later occurrences
  ``reference=True``; canonical home = the inbound path whose source declares
  the most datatype properties, tie-broken by shallowest depth. Exception
  (seller/buyer multi-role shape): all-single edges from ONE source class
  stay full.
- ``owl:equivalentClass`` collapsed to one representative; blank restriction
  nodes are consumed, never emitted.
- ``skos:example``/``vann:example`` -> field examples.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field as dataclass_field, replace as dataclass_replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from docling_graph.templategen.naming import (
    normalize_edge_label,
    sanitize_class_name,
    sanitize_field_name,
    to_snake_case,
)
from docling_graph.templategen.ontology import (
    MIN_CARDINALITY_NOTE,
    cardinality_sentence,
    class_passes_filters,
    document_reference_field,
    identity_ladder_rank,
    placeholder_docstring,
    require_optional_dependency,
    unique_name,
)
from docling_graph.templategen.spec import SpecGap

if TYPE_CHECKING:
    from rdflib import Graph
    from rdflib.term import Node, URIRef

XSD_SCALAR_MAP: dict[str, str] = {
    "string": "str",
    "normalizedString": "str",
    "token": "str",
    "anyURI": "str",
    "langString": "str",
    "integer": "int",
    "int": "int",
    "long": "int",
    "short": "int",
    "byte": "int",
    "nonNegativeInteger": "int",
    "positiveInteger": "int",
    "unsignedLong": "int",
    "unsignedInt": "int",
    "unsignedShort": "int",
    "decimal": "float",
    "float": "float",
    "double": "float",
    "boolean": "bool",
    "date": "date",
    "dateTime": "datetime",
    "dateTimeStamp": "datetime",
}
"""XSD datatype local name -> SPEC scalar type; anything unmapped is ``str``."""

_TYPE_WIDENING: dict[frozenset[str], str] = {
    frozenset({"int", "float"}): "float",
}
"""Type-conflict lattice for flattened same-name properties: ``int``/``float``
widen to ``float``; any other disagreement widens to ``str`` (never narrows)."""

_PREFERRED_IDENTITY_IRIS: frozenset[str] = frozenset(
    {
        "http://purl.org/dc/terms/identifier",
        "http://schema.org/identifier",
        "https://schema.org/identifier",
    }
)

_VANN_EXAMPLE = "http://purl.org/vocab/vann/example"

_MAX_IS_NOT_SIBLINGS = 3
"""Cap on "NOT a <sibling>" clauses per docstring (the rest add noise, not routing)."""


class SkosOnlyOntologyError(ValueError):
    """Raised for SKOS-only vocabularies: enums extracted, but no structure.

    Carries the compiled :class:`~docling_graph.templategen.spec.EnumSpec`
    drafts on ``.enums`` so callers can still persist the vocabularies.
    """

    def __init__(self, message: str, enums: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.enums = enums


@dataclass
class _PropInfo:
    """One usable ontology property, pre-classified."""

    iri: URIRef
    local: str
    kind: str  # "datatype" | "object"
    range_iri: URIRef | None
    functional: bool
    inverse_functional: bool
    preferred_identity: bool
    description: str
    examples: list[str] = dataclass_field(default_factory=list)
    scalar_override: str | None = None
    """Set when hierarchy flattening widened conflicting datatype ranges."""


def spec_draft_from_owl(
    path: str | Path,
    *,
    root: str | None = None,
    depth: int = 4,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[SpecGap]]:
    """Compile an OWL/RDFS/SKOS file into a loose TemplateSpec draft + gaps.

    Args:
        path: Ontology file (any serialization rdflib can parse: ttl/rdf/owl/...).
        root: Root class as full IRI, CURIE (resolved against the graph's
            namespaces), or unique local name. ``None`` auto-detects the single
            class that is no object property's range.
        depth: Maximum BFS depth from the root over object-property ranges.
        include: Glob patterns over class local names; non-matching classes
            (root excepted) are pruned.
        exclude: Glob patterns over class local names to drop (wins over include).

    Raises:
        ImportError: rdflib is not installed (``pip install 'docling-graph[templategen]'``).
        SkosOnlyOntologyError: The file is a SKOS vocabulary with no classes.
        ValueError: Unparseable file, unresolvable/ambiguous root.
    """
    rdflib = require_optional_dependency("rdflib", purpose="OWL/RDFS/SKOS ontology compilation")
    graph = rdflib.Graph()
    try:
        graph.parse(str(path))
    except Exception as exc:
        raise ValueError(f"rdflib could not parse '{path}': {exc}") from exc
    return _OwlCompiler(
        graph,
        source=Path(path),
        root=root,
        depth=depth,
        include=include,
        exclude=exclude,
    ).compile()


class _OwlCompiler:
    """Single-use compiler: one rdflib graph -> one draft dict + gaps."""

    def __init__(
        self,
        graph: Graph,
        *,
        source: Path,
        root: str | None,
        depth: int,
        include: Sequence[str] | None,
        exclude: Sequence[str] | None,
    ) -> None:
        self._g = graph
        self._source = source
        self._root_arg = root
        self._depth = depth
        self._include = include
        self._exclude = exclude
        self._notes: list[str] = []
        self._gaps: list[SpecGap] = []
        # Populated during compile():
        self._classes: set[URIRef] = set()
        self._rep: dict[URIRef, URIRef] = {}
        self._eq_group: dict[URIRef, list[URIRef]] = {}
        self._props_by_class: dict[URIRef, list[_PropInfo]] = {}
        self._parents: dict[URIRef, set[URIRef]] = {}
        self._children: dict[URIRef, set[URIRef]] = {}
        self._restrictions: dict[tuple[URIRef, URIRef], dict[str, int]] = {}
        self._enum_classes: dict[URIRef, dict[str, Any]] = {}
        self._enums_used: dict[URIRef, str] = {}
        self._enum_drafts: list[dict[str, Any]] = []
        self._closure: dict[URIRef, int] = {}
        self._dropped_abstract: set[URIRef] = set()
        self._emitted: dict[URIRef, str] = {}
        self._taken_names: set[str] = set()
        self._bounds: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def compile(self) -> tuple[dict[str, Any], list[SpecGap]]:
        self._harvest_classes()
        self._index_properties()
        self._collapse_equivalents()
        self._index_hierarchy()
        self._index_restrictions()
        self._detect_enum_classes()
        root_class = self._resolve_root()
        self._walk_closure(root_class)
        self._drop_abstract_parents(root_class)
        self._assign_model_names()

        models = [self._build_model(c, root_class) for c in self._emitted]
        self._apply_cardinality_bounds(models)
        self._mark_multi_path_references(models)

        draft = {
            "module_docstring": self._module_docstring(root_class),
            "root": self._emitted[root_class],
            "enums": self._enum_drafts,
            "models": models,
            "needs_root_list_dedup": [],
            "generator": {
                "format": "owl",
                "source": str(self._source),
                "root": str(root_class),
                "notes": self._notes,
            },
        }
        return draft, self._gaps

    # ------------------------------------------------------------------ #
    # Harvesting
    # ------------------------------------------------------------------ #

    def _harvest_classes(self) -> None:
        from rdflib import URIRef as _URIRef
        from rdflib.namespace import OWL, RDF, RDFS, SKOS

        for class_type in (OWL.Class, RDFS.Class):
            for subject in self._g.subjects(RDF.type, class_type):
                if isinstance(subject, _URIRef):
                    self._classes.add(subject)

        if not self._classes:
            schemes = list(self._g.subjects(RDF.type, SKOS.ConceptScheme))
            concepts = list(self._g.subjects(RDF.type, SKOS.Concept))
            if schemes or concepts:
                raise SkosOnlyOntologyError(
                    "The ontology declares SKOS vocabularies but no OWL/RDFS classes. "
                    "SKOS schemes provide vocabularies, not structure — combine them "
                    "with a structural ontology or generate from example documents "
                    "with 'docling-graph template from-docs'.",
                    enums=self._build_scheme_enums(),
                )
            raise ValueError(f"No owl:Class / rdfs:Class declarations found in '{self._source}'.")

    def _index_properties(self) -> None:
        from rdflib import URIRef as _URIRef
        from rdflib.namespace import OWL, RDF, RDFS, SKOS

        candidates: set[URIRef] = set()
        for prop_type in (
            OWL.DatatypeProperty,
            OWL.ObjectProperty,
            OWL.FunctionalProperty,
            OWL.InverseFunctionalProperty,
            RDF.Property,
        ):
            for subject in self._g.subjects(RDF.type, prop_type):
                if isinstance(subject, _URIRef):
                    candidates.add(subject)

        for prop in sorted(candidates, key=str):
            if (prop, RDF.type, OWL.AnnotationProperty) in self._g:
                continue
            ranges = sorted(
                (o for o in self._g.objects(prop, RDFS.range) if isinstance(o, _URIRef)),
                key=str,
            )
            kind, range_iri = self._classify_property(prop, ranges)
            if kind is None:
                continue

            domains = self._domains_of(prop)
            if not domains:
                self._notes.append(
                    f"Property '{_local(prop)}' has no rdfs:domain; skipped "
                    "(attaching everywhere would produce noise fields)."
                )
                continue

            description = self._annotation(prop, RDFS.comment, SKOS.definition)
            examples = self._examples_of(prop)
            info = _PropInfo(
                iri=prop,
                local=_local(prop),
                kind=kind,
                range_iri=range_iri,
                functional=(prop, RDF.type, OWL.FunctionalProperty) in self._g,
                inverse_functional=(prop, RDF.type, OWL.InverseFunctionalProperty) in self._g,
                preferred_identity=self._is_preferred_identity(prop),
                description=description,
                examples=examples,
            )
            for domain in domains:
                self._props_by_class.setdefault(domain, []).append(info)

    def _classify_property(
        self, prop: URIRef, ranges: list[URIRef]
    ) -> tuple[str | None, URIRef | None]:
        from rdflib.namespace import OWL, RDF

        explicit_datatype = (prop, RDF.type, OWL.DatatypeProperty) in self._g
        explicit_object = (prop, RDF.type, OWL.ObjectProperty) in self._g

        class_ranges = [r for r in ranges if r in self._classes]
        xsd_ranges = [r for r in ranges if self._is_literal_range(r)]

        if explicit_object:
            if class_ranges:
                return "object", class_ranges[0]
            self._notes.append(
                f"Object property '{_local(prop)}' has no named class range; skipped."
            )
            return None, None
        if explicit_datatype:
            return "datatype", xsd_ranges[0] if xsd_ranges else None
        # Untyped rdf:Property / bare Functional: classify by range.
        if class_ranges:
            return "object", class_ranges[0]
        return "datatype", xsd_ranges[0] if xsd_ranges else None

    @staticmethod
    def _is_literal_range(range_iri: URIRef) -> bool:
        from rdflib.namespace import RDF, RDFS, XSD

        text = str(range_iri)
        return text.startswith(str(XSD)) or range_iri == RDF.langString or range_iri == RDFS.Literal

    def _domains_of(self, prop: URIRef) -> list[URIRef]:
        from rdflib import BNode, URIRef as _URIRef
        from rdflib.collection import Collection
        from rdflib.namespace import OWL, RDFS

        domains: list[URIRef] = []
        for domain in self._g.objects(prop, RDFS.domain):
            if isinstance(domain, _URIRef):
                domains.append(domain)
            elif isinstance(domain, BNode):
                union = self._g.value(domain, OWL.unionOf)
                if union is not None:
                    members = [m for m in Collection(self._g, union) if isinstance(m, _URIRef)]
                    domains.extend(members)
                    self._notes.append(
                        f"Property '{_local(prop)}' has a union domain; attached to "
                        f"{', '.join(sorted(_local(m) for m in members))}."
                    )
        return sorted(set(domains), key=str)

    def _examples_of(self, subject: URIRef) -> list[str]:
        from rdflib import URIRef as _URIRef
        from rdflib.namespace import SKOS

        values = set(self._g.objects(subject, SKOS.example))
        values |= set(self._g.objects(subject, _URIRef(_VANN_EXAMPLE)))
        return sorted(str(v) for v in values)[:5]

    def _is_preferred_identity(self, prop: URIRef) -> bool:
        from rdflib import URIRef as _URIRef
        from rdflib.namespace import RDFS

        if str(prop) in _PREFERRED_IDENTITY_IRIS:
            return True
        return any(
            isinstance(parent, _URIRef) and str(parent) in _PREFERRED_IDENTITY_IRIS
            for parent in self._g.objects(prop, RDFS.subPropertyOf)
        )

    # ------------------------------------------------------------------ #
    # Equivalence, hierarchy, restrictions, enums
    # ------------------------------------------------------------------ #

    def _collapse_equivalents(self) -> None:
        from rdflib import URIRef as _URIRef
        from rdflib.namespace import OWL

        parent_of: dict[URIRef, URIRef] = {c: c for c in self._classes}

        def find(c: URIRef) -> URIRef:
            while parent_of[c] != c:
                parent_of[c] = parent_of[parent_of[c]]
                c = parent_of[c]
            return c

        for a, b in self._g.subject_objects(OWL.equivalentClass):
            if (
                isinstance(a, _URIRef)
                and isinstance(b, _URIRef)
                and a in self._classes
                and b in self._classes
            ):
                parent_of[find(a)] = find(b)

        groups: dict[URIRef, list[URIRef]] = {}
        for c in self._classes:
            groups.setdefault(find(c), []).append(c)

        for members in groups.values():
            rep = max(
                members,
                key=lambda c: (len(self._props_by_class.get(c, [])), str(c)),
            )
            for member in members:
                self._rep[member] = rep
            self._eq_group[rep] = sorted(members, key=str)
            if len(members) > 1:
                merged: list[_PropInfo] = []
                seen: set[URIRef] = set()
                for member in self._eq_group[rep]:
                    for info in self._props_by_class.get(member, []):
                        if info.iri not in seen:
                            seen.add(info.iri)
                            merged.append(info)
                self._props_by_class[rep] = merged
                self._notes.append(
                    "Collapsed owl:equivalentClass group "
                    f"{{{', '.join(_local(m) for m in self._eq_group[rep])}}} "
                    f"into '{_local(rep)}'."
                )

    def _to_rep(self, c: URIRef) -> URIRef:
        return self._rep.get(c, c)

    def _index_hierarchy(self) -> None:
        from rdflib import URIRef as _URIRef
        from rdflib.namespace import RDFS

        for child, parent in self._g.subject_objects(RDFS.subClassOf):
            if not (isinstance(child, _URIRef) and isinstance(parent, _URIRef)):
                continue
            if child not in self._classes or parent not in self._classes:
                continue
            child_rep, parent_rep = self._to_rep(child), self._to_rep(parent)
            if child_rep == parent_rep:
                continue
            self._parents.setdefault(child_rep, set()).add(parent_rep)
            self._children.setdefault(parent_rep, set()).add(child_rep)

    def _ancestors(self, c: URIRef) -> list[URIRef]:
        """All ancestors of ``c``, nearest first, cycle-safe."""
        ordered: list[URIRef] = []
        seen: set[URIRef] = {c}
        frontier = sorted(self._parents.get(c, ()), key=str)
        while frontier:
            nxt: list[URIRef] = []
            for parent in frontier:
                if parent in seen:
                    continue
                seen.add(parent)
                ordered.append(parent)
                nxt.extend(sorted(self._parents.get(parent, ()), key=str))
            frontier = nxt
        return ordered

    def _index_restrictions(self) -> None:
        from rdflib import BNode, URIRef as _URIRef
        from rdflib.namespace import OWL, RDF, RDFS

        for subject, restriction in self._g.subject_objects(RDFS.subClassOf):
            if not isinstance(subject, _URIRef) or subject not in self._classes:
                continue
            if not isinstance(restriction, BNode):
                continue
            if (restriction, RDF.type, OWL.Restriction) not in self._g:
                continue
            on_property = self._g.value(restriction, OWL.onProperty)
            if not isinstance(on_property, _URIRef):
                continue

            exact = self._g.value(restriction, OWL.cardinality) or self._g.value(
                restriction, OWL.qualifiedCardinality
            )
            max_card = (
                self._g.value(restriction, OWL.maxCardinality)
                or self._g.value(restriction, OWL.maxQualifiedCardinality)
                or exact
            )
            min_card = (
                self._g.value(restriction, OWL.minCardinality)
                or self._g.value(restriction, OWL.minQualifiedCardinality)
                or exact
            )
            entry = self._restrictions.setdefault((self._to_rep(subject), on_property), {})
            max_value = _literal_int(max_card)
            min_value = _literal_int(min_card)
            if max_value is not None:
                entry["max"] = max_value
            if min_value is not None:
                entry["min"] = min_value

    def _restriction_for(self, c: URIRef, prop: URIRef) -> dict[str, int]:
        """Restriction bounds for ``(c, prop)``, own class first, then ancestors."""
        for cls in [c, *self._ancestors(c)]:
            entry = self._restrictions.get((cls, prop))
            if entry:
                return entry
        return {}

    def _detect_enum_classes(self) -> None:
        for c in sorted(self._classes, key=str):
            rep = self._to_rep(c)
            if rep in self._enum_classes:
                continue
            enum_info = self._one_of_enum(rep) or self._skos_backed_enum(rep)
            if enum_info is not None:
                self._enum_classes[rep] = enum_info

    def _one_of_enum(self, c: URIRef) -> dict[str, Any] | None:
        from rdflib import BNode
        from rdflib.collection import Collection
        from rdflib.namespace import OWL

        one_of = self._g.value(c, OWL.oneOf)
        if one_of is None:
            for equivalent in self._g.objects(c, OWL.equivalentClass):
                if isinstance(equivalent, BNode):
                    one_of = self._g.value(equivalent, OWL.oneOf)
                    if one_of is not None:
                        break
        if one_of is None:
            return None
        members: list[str] = []
        synonyms: dict[str, list[str]] = {}
        for individual in Collection(self._g, one_of):
            label = self._member_label(individual)
            members.append(label)
            alt = self._alt_labels(individual)
            if alt:
                synonyms[label] = alt
        if not members:
            return None
        # Even a provably total owl:oneOf catalog keeps the OTHER fallback:
        # documents quote values the ontology never listed (the class's own
        # altLabel synonyms included), and the generated _normalize_enum must
        # coerce-and-warn, never raise (R17, the never-reject law).
        return {"members": members, "synonyms": synonyms, "include_other": True}

    def _skos_backed_enum(self, c: URIRef) -> dict[str, Any] | None:
        from rdflib import URIRef as _URIRef
        from rdflib.namespace import RDF, SKOS

        instances = [i for i in self._g.subjects(RDF.type, c) if isinstance(i, _URIRef)]
        if not instances:
            return None
        schemes: set[Node] = set()
        for instance in instances:
            in_schemes = set(self._g.objects(instance, SKOS.inScheme))
            if not in_schemes:
                return None
            schemes |= in_schemes
        if len(schemes) != 1:
            return None
        members: list[str] = []
        synonyms: dict[str, list[str]] = {}
        for instance in sorted(instances, key=self._member_label):
            label = self._member_label(instance)
            members.append(label)
            alt = self._alt_labels(instance)
            if alt:
                synonyms[label] = alt
        return {"members": members, "synonyms": synonyms, "include_other": True}

    def _member_label(self, individual: Node) -> str:
        from rdflib.namespace import RDFS, SKOS

        for predicate in (SKOS.prefLabel, RDFS.label):
            value = self._g.value(individual, predicate)
            if value is not None:
                return str(value)
        return _local(individual)

    def _alt_labels(self, individual: Node) -> list[str]:
        from rdflib.namespace import SKOS

        return sorted(str(v) for v in self._g.objects(individual, SKOS.altLabel))

    def _build_scheme_enums(self) -> list[dict[str, Any]]:
        from rdflib.namespace import RDF, RDFS, SKOS

        enums: list[dict[str, Any]] = []
        taken: set[str] = set()
        for scheme in sorted(self._g.subjects(RDF.type, SKOS.ConceptScheme), key=str):
            label = self._g.value(scheme, RDFS.label) or self._g.value(scheme, SKOS.prefLabel)
            name = unique_name(sanitize_class_name(str(label) if label else _local(scheme)), taken)
            members: list[str] = []
            synonyms: dict[str, list[str]] = {}
            concepts = set(self._g.subjects(SKOS.inScheme, scheme))
            concepts |= set(self._g.subjects(SKOS.topConceptOf, scheme))
            for concept in sorted(concepts, key=self._member_label):
                member = self._member_label(concept)
                members.append(member)
                alt = self._alt_labels(concept)
                if alt:
                    synonyms[member] = alt
            if members:
                enums.append(
                    {
                        "name": name,
                        "members": members,
                        "synonyms": synonyms,
                        "include_other": True,
                    }
                )
        return enums

    # ------------------------------------------------------------------ #
    # Root resolution + closure
    # ------------------------------------------------------------------ #

    def _object_range_reps(self) -> set[URIRef]:
        targets: set[URIRef] = set()
        for infos in self._props_by_class.values():
            for info in infos:
                if info.kind == "object" and info.range_iri is not None:
                    targets.add(self._to_rep(info.range_iri))
        return targets

    def _resolve_root(self) -> URIRef:
        reps = sorted({self._to_rep(c) for c in self._classes}, key=str)
        if self._root_arg is None:
            ranged = self._object_range_reps()
            candidates = [c for c in reps if c not in ranged and c not in self._enum_classes]
            if len(candidates) == 1:
                return candidates[0]
            listing = ", ".join(_local(c) for c in candidates) or "none"
            raise ValueError(
                "Cannot auto-detect the root class (classes that are no object "
                f"property's range): candidates = {listing}. Pass --root explicitly."
            )

        iri = self._root_iri_from_arg(self._root_arg)
        rep = self._to_rep(iri)
        if rep not in {self._to_rep(c) for c in self._classes}:
            raise ValueError(
                f"Root '{self._root_arg}' resolved to <{iri}>, which is not a "
                "declared class in the ontology."
            )
        if rep in self._enum_classes:
            raise ValueError(
                f"Root '{self._root_arg}' is an enumerated class (owl:oneOf / SKOS "
                "vocabulary) — pick a structural class as the root."
            )
        return rep

    def _root_iri_from_arg(self, arg: str) -> URIRef:
        from rdflib import URIRef as _URIRef

        if "://" in arg:
            return _URIRef(arg)
        if ":" in arg:
            try:
                return self._g.namespace_manager.expand_curie(arg)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Cannot expand CURIE '{arg}' against the graph's namespaces: {exc}"
                ) from exc
        matches = sorted((c for c in self._classes if _local(c) == arg), key=str)
        if not matches:
            available = ", ".join(sorted({_local(c) for c in self._classes}))
            raise ValueError(f"No class with local name '{arg}'. Available classes: {available}.")
        if len(matches) > 1:
            listing = ", ".join(f"<{c}>" for c in matches)
            raise ValueError(
                f"Local name '{arg}' is ambiguous ({listing}); pass a full IRI or CURIE."
            )
        return matches[0]

    def _passes_filters(self, c: URIRef, *, is_root: bool = False) -> bool:
        return class_passes_filters(_local(c), self._include, self._exclude, is_root=is_root)

    def _walk_closure(self, root_class: URIRef) -> None:
        queue: deque[tuple[URIRef, int]] = deque([(root_class, 0)])
        while queue:
            current, current_depth = queue.popleft()
            if current in self._closure:
                continue
            self._closure[current] = current_depth

            # Subclass expansion: specializations enter at the same depth.
            for child in sorted(self._children.get(current, ()), key=str):
                if child not in self._closure and self._passes_filters(child):
                    queue.append((child, current_depth))

            if current_depth >= self._depth:
                continue
            for info in self._effective_props(current):
                if info.kind != "object" or info.range_iri is None:
                    continue
                target = self._to_rep(info.range_iri)
                if target in self._enum_classes or target in self._closure:
                    continue
                if self._passes_filters(target):
                    queue.append((target, current_depth + 1))

    def _effective_props(self, c: URIRef) -> list[_PropInfo]:
        """Flattened properties of ``c``: ancestors farthest-first, own last.

        Same-name collisions keep the nearest declaration; conflicting scalar
        types widen (``int``/``float`` -> ``float``, else ``str``), never narrow.
        """
        merged: dict[str, _PropInfo] = {}
        lineage = [*reversed(self._ancestors(c)), c]
        for cls in lineage:
            for info in sorted(self._props_by_class.get(cls, []), key=lambda i: i.local):
                previous = merged.get(info.local)
                if previous is not None and previous.kind == "datatype" == info.kind:
                    prev_type = _scalar_of(previous)
                    new_type = _scalar_of(info)
                    if prev_type != new_type:
                        widened = _TYPE_WIDENING.get(frozenset({prev_type, new_type}), "str")
                        self._notes.append(
                            f"Property '{info.local}' on '{_local(c)}' widened to "
                            f"'{widened}' ({prev_type} vs {new_type} across the hierarchy)."
                        )
                        info = dataclass_replace(info, scalar_override=widened)
                merged[info.local] = info
        return sorted(merged.values(), key=lambda i: i.local)

    def _drop_abstract_parents(self, root_class: URIRef) -> None:
        for c in self._closure:
            if c == root_class:
                continue
            children_in_closure = [
                child for child in self._children.get(c, ()) if child in self._closure
            ]
            if len(children_in_closure) >= 2 and not self._props_by_class.get(c):
                self._dropped_abstract.add(c)
                self._notes.append(
                    f"Dropped abstract parent '{_local(c)}' (no direct properties, "
                    f"{len(children_in_closure)} subclasses); edges fan out per subclass."
                )

    def _assign_model_names(self) -> None:
        ordered = sorted(self._closure, key=lambda c: (self._closure[c], str(c)))
        for c in ordered:
            if c in self._dropped_abstract or c in self._enum_classes:
                continue
            base = sanitize_class_name(_local(c))
            name = unique_name(base, self._taken_names)
            if name != base:
                self._notes.append(
                    f"Class <{c}> renamed to '{name}': '{base}' was already taken "
                    "by a distinct class. Rename in the SPEC or --exclude one of them."
                )
            self._emitted[c] = name

    # ------------------------------------------------------------------ #
    # Model building
    # ------------------------------------------------------------------ #

    def _build_model(self, c: URIRef, root_class: URIRef) -> dict[str, Any]:
        name = self._emitted[c]
        fields, prop_field_names = self._build_fields(c)
        identity = self._select_identity(c, name, fields, prop_field_names)

        kind = "root" if c == root_class else ("entity" if identity else "component")
        if c == root_class and not identity:
            fields.insert(0, document_reference_field())
            identity = ["document_reference"]
            self._gaps.append(
                SpecGap(
                    model=name,
                    field="document_reference",
                    kind="missing_identity",
                    note=(
                        "The ontology gives the root no identity; synthesized "
                        "'document_reference'. Replace with a real printed identifier."
                    ),
                )
            )
        elif not identity:
            self._gaps.append(
                SpecGap(
                    model=name,
                    kind="missing_identity",
                    note=(
                        "No owl:hasKey, InverseFunctionalProperty, or identity-like "
                        "property; demoted to component (never invent ids)."
                    ),
                )
            )

        for field_draft in fields:
            if field_draft["name"] in identity:
                field_draft["role"] = "identity"
                field_draft["is_list"] = False
                if len(field_draft.get("examples", [])) < 2:
                    self._gaps.append(
                        SpecGap(
                            model=name,
                            field=field_draft["name"],
                            kind="missing_examples",
                            note="Ontologies define shapes, not instances; add 2-5 "
                            "verbatim examples from a real document.",
                        )
                    )

        docstring = self._docstring_for(c, name)
        return {
            "name": name,
            "kind": kind,
            "docstring": docstring,
            "identity_fields": identity,
            "max_instances": None,
            "fields": fields,
            "canonical_home": None,
            "provenance": "ontology",
            "source_ref": str(c),
        }

    def _build_fields(self, c: URIRef) -> tuple[list[dict[str, Any]], dict[str, _PropInfo]]:
        fields: list[dict[str, Any]] = []
        prop_by_field: dict[str, _PropInfo] = {}
        used: set[str] = set()

        for info in self._effective_props(c):
            base_name = sanitize_field_name(info.local)
            field_name = base_name
            counter = 2
            while field_name in used:
                field_name = f"{base_name}_{counter}"
                counter += 1

            restriction = self._restriction_for(c, info.iri)
            max_card = restriction.get("max")
            min_card = restriction.get("min")
            is_list = not (info.functional or max_card == 1)
            description = info.description
            if min_card is not None and min_card >= 1:
                description = _join_sentences(description, MIN_CARDINALITY_NOTE)

            if info.kind == "datatype":
                if max_card is not None and max_card > 1:
                    description = _join_sentences(
                        description, f"At most {max_card} values per document."
                    )
                used.add(field_name)
                prop_by_field[field_name] = info
                fields.append(
                    {
                        "name": field_name,
                        "type": _scalar_of(info),
                        "is_list": is_list,
                        "description": description,
                        "examples": info.examples,
                        "role": "property",
                    }
                )
                continue

            target = self._to_rep(info.range_iri) if info.range_iri is not None else None
            if target is None:
                continue
            if target in self._enum_classes:
                enum_name = self._ensure_enum(target)
                used.add(field_name)
                fields.append(
                    {
                        "name": field_name,
                        "type": enum_name,
                        "is_list": is_list,
                        "description": description,
                        "role": "property",
                        "normalizer": "enum",
                    }
                )
                continue
            if target in self._dropped_abstract:
                children = self._emitted_descendants(target)
                if not children:
                    self._notes.append(
                        f"Edge '{_local(c)}.{field_name}' dropped: abstract target "
                        f"'{_local(target)}' contributes no emitted subclass "
                        "(all descendants pruned or non-structural)."
                    )
                    continue
                for child in children:
                    child_field = f"{field_name}_{to_snake_case(self._emitted[child])}"
                    used.add(child_field)
                    fields.append(self._edge_field(child_field, info, child, is_list, description))
                self._register_bound(target_children=children, max_card=max_card)
                continue
            if target not in self._emitted:
                self._notes.append(
                    f"Edge '{_local(c)}.{field_name}' dropped: target "
                    f"'{_local(target)}' is outside the closure (depth/include/exclude)."
                )
                continue

            used.add(field_name)
            fields.append(self._edge_field(field_name, info, target, is_list, description))
            if max_card is not None and max_card > 1:
                bound_name = self._emitted[target]
                self._bounds[bound_name] = max(self._bounds.get(bound_name, 0), max_card)

        return fields, prop_by_field

    def _emitted_descendants(self, target: URIRef) -> list[URIRef]:
        """Emitted subclasses a dropped abstract fans out to, transitively.

        A child that is itself a dropped abstract is expanded into ITS emitted
        descendants (grandchildren and deeper), so nested abstract hierarchies
        never orphan their concrete leaves.
        """
        found: set[URIRef] = set()
        seen: set[URIRef] = {target}
        stack: list[URIRef] = [target]
        while stack:
            current = stack.pop()
            for child in self._children.get(current, ()):
                if child in seen:
                    continue
                seen.add(child)
                if child in self._emitted:
                    found.add(child)
                elif child in self._dropped_abstract:
                    stack.append(child)
        return sorted(found, key=lambda child: self._emitted[child])

    def _register_bound(self, *, target_children: Iterable[URIRef], max_card: int | None) -> None:
        if max_card is None or max_card <= 1:
            return
        for child in target_children:
            bound_name = self._emitted[child]
            self._bounds[bound_name] = max(self._bounds.get(bound_name, 0), max_card)

    def _edge_field(
        self,
        field_name: str,
        info: _PropInfo,
        target: URIRef,
        is_list: bool,
        description: str,
    ) -> dict[str, Any]:
        return {
            "name": field_name,
            "type": self._emitted[target],
            "is_list": is_list,
            "description": description,
            "role": "edge",
            "edge_label": normalize_edge_label(info.local, target=_local(target)),
        }

    def _ensure_enum(self, target: URIRef) -> str:
        if target in self._enums_used:
            return self._enums_used[target]
        enum_info = self._enum_classes[target]
        name = unique_name(sanitize_class_name(_local(target)), self._taken_names)
        self._enums_used[target] = name
        self._enum_drafts.append({"name": name, **enum_info})
        return name

    def _select_identity(
        self,
        c: URIRef,
        name: str,
        fields: list[dict[str, Any]],
        prop_by_field: dict[str, _PropInfo],
    ) -> list[str]:
        datatype_fields = [f["name"] for f in fields if f["name"] in prop_by_field]

        # 1. owl:hasKey (own class first, then ancestors).
        key_props = self._has_key_props(c)
        if key_props:
            field_by_prop = {prop_by_field[fname].iri: fname for fname in datatype_fields}
            key_fields = [field_by_prop[p] for p in key_props if p in field_by_prop]
            if len(key_fields) > 2:
                self._gaps.append(
                    SpecGap(
                        model=name,
                        field=key_fields[2],
                        kind="missing_identity",
                        note=(
                            f"owl:hasKey lists {len(key_fields)} properties; only the "
                            "first 2 kept as identity (identity is 1-2 scalar fields)."
                        ),
                    )
                )
            if key_fields:
                return key_fields[:2]

        # 2. Datatype owl:InverseFunctionalProperty.
        ifp_fields = sorted(
            (fname for fname in datatype_fields if prop_by_field[fname].inverse_functional),
            key=lambda fname: (not prop_by_field[fname].preferred_identity, fname),
        )
        if ifp_fields:
            return [ifp_fields[0]]

        # 3. Heuristic ladder, preferring dcterms:identifier / schema:identifier.
        ranked = sorted(
            (
                (
                    not prop_by_field[fname].preferred_identity,
                    identity_ladder_rank(fname)
                    if identity_ladder_rank(fname) is not None
                    else len(fields) + 99,
                    fname,
                )
                for fname in datatype_fields
                if prop_by_field[fname].preferred_identity
                or identity_ladder_rank(fname) is not None
            ),
        )
        if ranked:
            return [ranked[0][2]]
        return []

    def _has_key_props(self, c: URIRef) -> list[URIRef]:
        from rdflib import URIRef as _URIRef
        from rdflib.collection import Collection
        from rdflib.namespace import OWL

        for cls in [c, *self._ancestors(c)]:
            for member in self._eq_group.get(cls, [cls]):
                key_list = self._g.value(member, OWL.hasKey)
                if key_list is not None:
                    return [p for p in Collection(self._g, key_list) if isinstance(p, _URIRef)]
        return []

    # ------------------------------------------------------------------ #
    # Docstrings + post passes
    # ------------------------------------------------------------------ #

    def _annotation(self, subject: URIRef, *predicates: Any) -> str:
        for member in self._eq_group.get(subject, [subject]):
            for predicate in predicates:
                value = self._g.value(member, predicate)
                if value is not None:
                    return str(value)
        return ""

    def _docstring_for(self, c: URIRef, name: str) -> str:
        from rdflib.namespace import RDFS, SKOS

        docstring = self._annotation(c, RDFS.comment, SKOS.definition)
        if not docstring:
            docstring = placeholder_docstring(name)
            self._gaps.append(
                SpecGap(
                    model=name,
                    kind="missing_docstring",
                    note="No rdfs:comment / skos:definition in the ontology.",
                )
            )

        labels = sorted(
            {
                str(label)
                for member in self._eq_group.get(c, [c])
                for label in self._g.objects(member, RDFS.label)
                if sanitize_class_name(str(label)) != name
            }
        )
        if labels:
            docstring = _join_sentences(docstring, f"Also called: {', '.join(labels)}.")

        siblings = sorted(
            {
                self._emitted[sibling]
                for parent in self._parents.get(c, ())
                for sibling in self._children.get(parent, ())
                if sibling != c and sibling in self._emitted
            }
        )
        for sibling in siblings[:_MAX_IS_NOT_SIBLINGS]:
            docstring = _join_sentences(docstring, f"NOT a {sibling}.")
        return docstring

    def _apply_cardinality_bounds(self, models: list[dict[str, Any]]) -> None:
        by_name = {m["name"]: m for m in models}
        for target_name, documented_max in sorted(self._bounds.items()):
            model = by_name.get(target_name)
            if model is None:
                continue
            if model["kind"] != "entity":
                self._notes.append(
                    f"Cardinality bound on '{target_name}' skipped: "
                    f"{model['kind']}s take no max_instances."
                )
                continue
            # The DOCUMENTED maximum: the linter's repair_draft doubles it once.
            model["max_instances"] = documented_max
            model["docstring"] = _join_sentences(
                model["docstring"], cardinality_sentence(documented_max)
            )

    def _mark_multi_path_references(self, models: list[dict[str, Any]]) -> None:
        """R10 pre-pass: >= 2 inbound edges -> one canonical home, rest references.

        Canonical = the inbound path whose source declares the most datatype
        properties, tie-broken by shallowest BFS depth, then names.

        Same-parent multi-role exception (seller/buyer, insurer/reinsurer):
        when every inbound edge is single-valued and originates from ONE source
        class, the paths are distinct roles of one relationship, not duplicate
        nesting — nothing flips to reference (only ``canonical_home`` is
        recorded). The linter's R10 applies the same shape-keyed exception, so
        flipping here would be unrecoverable data loss.
        """
        by_name = {m["name"]: m for m in models}
        depth_by_name = {
            self._emitted[c]: d for c, d in self._closure.items() if c in self._emitted
        }
        inbound: dict[str, list[tuple[str, str, bool]]] = {}
        for model in models:
            for field_draft in model["fields"]:
                if field_draft.get("role") == "edge":
                    inbound.setdefault(field_draft["type"], []).append(
                        (model["name"], field_draft["name"], bool(field_draft.get("is_list")))
                    )

        def datatype_count(model_name: str) -> int:
            return sum(1 for f in by_name[model_name]["fields"] if f.get("role") != "edge")

        for target_name, paths in sorted(inbound.items()):
            target = by_name.get(target_name)
            if target is None or target["kind"] != "entity" or len(paths) < 2:
                continue
            ordered = sorted(
                paths,
                key=lambda p: (-datatype_count(p[0]), depth_by_name.get(p[0], 0), p[:2]),
            )
            canonical_model, canonical_field = ordered[0][0], ordered[0][1]
            target["canonical_home"] = f"{canonical_model}.{canonical_field}"
            sources = {source for source, _, _ in paths}
            if len(sources) == 1 and all(not is_list for _, _, is_list in paths):
                self._notes.append(
                    f"'{target_name}' has {len(paths)} single-valued edges from "
                    f"'{canonical_model}' (multi-role shape); all paths stay full edges."
                )
                continue
            for source_name, field_name, _ in ordered[1:]:
                for field_draft in by_name[source_name]["fields"]:
                    if field_draft["name"] == field_name:
                        field_draft["reference"] = True
            self._notes.append(
                f"'{target_name}' is reachable from {len(paths)} object properties; "
                f"canonical home '{canonical_model}.{canonical_field}', other paths "
                "marked reference=True."
            )

    def _module_docstring(self, root_class: URIRef) -> str:
        from rdflib import URIRef as _URIRef
        from rdflib.namespace import DCTERMS, OWL, RDF, RDFS

        for ontology in sorted(self._g.subjects(RDF.type, OWL.Ontology), key=str):
            if not isinstance(ontology, _URIRef):
                continue
            for predicate in (RDFS.comment, DCTERMS.description):
                value = self._g.value(ontology, predicate)
                if value is not None:
                    return str(value)
        return (
            f"Template draft compiled from OWL ontology '{self._source.name}' "
            f"(root: {self._emitted[root_class]})."
        )


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #


def _local(node: Node) -> str:
    """Local name of an IRI: the fragment, else the last path segment."""
    text = str(node)
    for separator in ("#", "/", ":"):
        if separator in text:
            candidate = text.rsplit(separator, 1)[1]
            if candidate:
                return candidate
    return text


def _xsd_scalar(range_iri: URIRef | None) -> str:
    if range_iri is None:
        return "str"
    return XSD_SCALAR_MAP.get(_local(range_iri), "str")


def _scalar_of(info: _PropInfo) -> str:
    return info.scalar_override or _xsd_scalar(info.range_iri)


def _literal_int(node: Node | None) -> int | None:
    """Integer value of a cardinality literal, or None when absent/malformed."""
    if node is None:
        return None
    try:
        return int(str(node))
    except ValueError:
        return None


def _join_sentences(base: str, addition: str) -> str:
    if not base:
        return addition
    if not base.rstrip().endswith((".", "!", "?")):
        base = f"{base.rstrip()}."
    return f"{base.rstrip()} {addition}"
