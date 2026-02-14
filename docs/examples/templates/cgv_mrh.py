"""
Pydantic template for MRH insurance terms (CGV) extraction.

Extracts a graph-ready contract structure from French multirisque habitation (MRH) CGV,
with robust parsing helpers for amounts, currencies, and mixed list inputs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


# ----------------------------
# Docling Graph helper
# ----------------------------
def edge(label: str, default: Any = None, **kwargs: Any) -> Any:
    """
    Déclare un champ comme 'edge' pour Docling-Graph via json_schema_extra.
    """
    json_schema_extra = dict(kwargs.pop("json_schema_extra", {}) or {})
    json_schema_extra["edge_label"] = label

    if "default_factory" in kwargs:
        default_factory = kwargs.pop("default_factory")
        return Field(default_factory=default_factory, json_schema_extra=json_schema_extra, **kwargs)

    return Field(default, json_schema_extra=json_schema_extra, **kwargs)


# ----------------------------
# Parsing helpers
# ----------------------------
def parse_nombre_fr(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        clean = re.sub(r"[^\d,.-]", "", v).replace(",", ".")
        try:
            return float(clean)
        except ValueError:
            return None
    return None


def normalise_devise(v: Any) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        return str(v)
    vclean = v.strip().upper()
    for sym, code in {"€": "EUR", "$": "USD", "£": "GBP"}.items():
        if sym in vclean:
            return code
    return vclean if len(vclean) == 3 else "EUR"


def _filtrer_liste(v: Any, nom_champ: str, champs_requis: list[str] | None = None) -> Any:
    """
    Nettoie une liste avant parsing des sous-modèles (évite dict vides / parasites).
    """
    if not isinstance(v, list):
        return v
    champs_requis = champs_requis or []

    out: list[Any] = []
    for item in v:
        if isinstance(item, BaseModel):
            out.append(item)
            continue

        if isinstance(item, dict):
            manquants = [k for k in champs_requis if not item.get(k)]
            if manquants:
                logger.warning(
                    "Suppression dict invalide dans %s (manque %s)", nom_champ, ",".join(manquants)
                )
                continue
            out.append(item)
            continue

        if isinstance(item, str):
            # Très utile pour Bien: on accepte str puis conversion via model_validator
            out.append(item)
            continue

        logger.warning("Suppression élément parasite dans %s: %r", nom_champ, item)

    return out


# ----------------------------
# Components (non-entities)
# ----------------------------
class Montant(BaseModel):
    """
    Montant d'argent (souvent incomplet dans les CGV).
    """

    model_config = ConfigDict(is_entity=False, extra="ignore", populate_by_name=True)

    valeur: float | None = Field(
        None,
        description="Valeur numérique si identifiable (ex. '1 500', '380', '120000').",
        examples=[1500.0, 380.0, 120000.0],
    )
    devise: str | None = Field(
        None,
        description=(
            "Devise si précisée (ISO 4217, ex. EUR, USD). "
            "Si absent du document, omettre (ne pas deviner)."
        ),
        examples=["EUR", "USD"],
    )
    indexe_par: str | None = Field(
        None,
        description="Référence d'indice si exprimé comme 'x fois un indice' (FFB, IRL, etc.).",
        examples=["FFB", "IRL", "Indice du prix de la construction"],
    )

    @model_validator(mode="before")
    @classmethod
    def accepter_scalaires(cls, v: Any) -> Any:
        if isinstance(v, int | float):
            return {"valeur": float(v), "devise": "EUR"}
        if isinstance(v, str):
            return {"valeur": parse_nombre_fr(v), "devise": normalise_devise(v) or "EUR"}
        return v

    @field_validator("valeur", mode="before")
    @classmethod
    def normaliser_valeur(cls, v: Any) -> Any:
        return parse_nombre_fr(v)

    @field_validator("devise", mode="before")
    @classmethod
    def normaliser_devise(cls, v: Any) -> Any:
        return normalise_devise(v)


class Franchise(BaseModel):
    """
    Franchise applicable (souvent 'par sinistre', etc.).
    """

    model_config = ConfigDict(is_entity=False, extra="ignore", populate_by_name=True)

    montant: Montant | None = Field(
        None,
        description="Montant de franchise si indiqué.",
        examples=[{"valeur": 380.0, "devise": "EUR"}, "380 €"],
    )
    type: str | None = Field(
        None,
        description="Type de franchise (fixe, pourcentage, franchise légale, etc.).",
        examples=["Fixe", "Pourcentage", "Franchise légale CAT-NAT"],
    )
    contexte: str | None = Field(
        None,
        description="Contexte d'application (par sinistre, pour le vol, CAT-NAT, etc.).",
        examples=["Par sinistre", "Vol", "Catastrophes naturelles"],
    )


class Condition(BaseModel):
    """
    Condition / obligation de prévention / mesure de sécurité.
    """

    model_config = ConfigDict(is_entity=False, extra="ignore", populate_by_name=True)

    texte: str | None = Field(
        None,
        description="Condition au plus proche du texte du document (éviter de paraphraser).",
        examples=[
            "En cas d'absence de plus de 24h, utiliser tous les moyens de fermeture et de protection.",
            "Faire procéder au ramonage des conduits avant chaque hiver.",
        ],
    )
    jours_inoccupation_max: int | None = Field(
        None,
        description="Si la condition porte sur l'inoccupation, extraire le nombre maximal de jours si présent.",
        examples=[3, 30, 60, 90],
    )


# ----------------------------
# Entities
# ----------------------------
class Bien(BaseModel):
    """
    Bien assuré / bien mentionné.
    Entité (Entity) dédupliquée via nom.
    """

    model_config = ConfigDict(graph_id_fields=["nom"], extra="ignore", populate_by_name=True)

    nom: str = Field(
        ...,
        description=(
            "Nom standardisé du bien (identifiant de déduplication). "
            "Utiliser un libellé cohérent dans tout le document (ex. 'Bâtiment', 'Mobilier')."
        ),
        examples=[
            "Bâtiment",
            "Mobilier",
            "Objets de valeur",
            "Jardin",
            "Piscine",
            "Dépendances",
            "Véranda",
        ],
    )
    description: str | None = Field(
        None,
        description=(
            "Définition/description du bien (périmètre, exemples). "
            "À extraire principalement depuis l'Article 1 'Les biens … assurés'."
        ),
        examples=[
            "Les bâtiments à usage d'habitation (murs, toiture…) selon le contrat.",
            "Biens mobiliers présents dans l'habitation (hors exclusions).",
        ],
    )

    @model_validator(mode="before")
    @classmethod
    def accepter_chaine(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"nom": v}
        return v


class Exclusion(BaseModel):
    """
    Clause d'exclusion (commune ou spécifique).
    Identity uses short exclusion_id for delta-friendly deduplication.
    """

    model_config = ConfigDict(
        graph_id_fields=["exclusion_id"], extra="ignore", populate_by_name=True
    )

    exclusion_id: str = Field(
        ...,
        description=(
            "Short stable identifier for this exclusion (deduplication). "
            "Use a normalized code or short label as in document (e.g. 'vol-sans-effraction', 'defaut-entretien'). "
            "Avoid long free text; keep under ~50 chars."
        ),
        examples=[
            "vol-sans-effraction",
            "defaut-entretien",
            "guerre",
            "risque-nucleaire",
        ],
    )
    resume: str | None = Field(
        None,
        description="Résumé court de la clause (affichage, complément).",
        examples=[
            "Exclusion vol sans effraction",
            "Exclusion défaut d'entretien",
        ],
    )
    texte: str | None = Field(
        None,
        description="Texte de l'exclusion (si possible proche/verbatim).",
        examples=[
            "Sont exclus les vols commis sans effraction ni violence.",
            "Sont exclus les dommages résultant d'un défaut d'entretien notoire.",
        ],
    )
    biens_exclus: list[Bien] = edge(
        label="EXCLUTBIEN",
        default_factory=list,
        validation_alias=AliasChoices("biens_exclus", "biensexclus"),
        description=(
            "Biens explicitement exclus par cette clause (si la clause cite des biens). "
            "Renseigner au minimum le 'nom' (référence vers les biens définis à l'Article 1)."
        ),
        examples=[[{"nom": "Piscine"}, {"nom": "Objets de valeur"}], ["Piscine"]],
    )

    @field_validator("biens_exclus", mode="before")
    @classmethod
    def filtrer_biens_exclus(cls, v: Any) -> Any:
        return _filtrer_liste(v, "biens_exclus")


class Garantie(BaseModel):
    """
    Garantie (ex. 'Dégâts des eaux', 'Vol et Vandalisme', etc.).
    """

    model_config = ConfigDict(graph_id_fields=["nom"], extra="ignore", populate_by_name=True)

    nom: str = Field(
        ...,
        description="Nom de la garantie tel qu'écrit dans le document.",
        examples=[
            "Dégâts des eaux",
            "Incendie et événements assimilés",
            "Vol et Vandalisme",
            "Bris de vitre",
            "Catastrophes naturelles et technologiques",
        ],
    )
    description: str | None = Field(
        None,
        description=(
            "Description (corps du texte de la garantie). "
            "Éviter de recopier le tableau; privilégier les paragraphes 'Nous garantissons…'."
        ),
    )

    biens_couverts: list[Bien] = edge(
        label="COUVREBIEN",
        default_factory=list,
        validation_alias=AliasChoices("biens_couverts", "bienscouverts"),
        description=(
            "Quels biens sont couverts par cette garantie. "
            "Règle pratique (anti-informations dispersées) : "
            "si la garantie parle de 'biens assurés / bien assuré' sans détailler, "
            "renseigner au minimum les biens principaux définis à l'Article 1 (souvent 'Bâtiment' et 'Mobilier'). "
            "Si la garantie/option vise un bien explicite (ex. 'Piscine', 'Jardin', 'Protection du mobilier'), "
            "ajouter ce bien dans la liste."
        ),
        examples=[
            [{"nom": "Bâtiment"}, {"nom": "Mobilier"}],
            [{"nom": "Piscine"}],
            [{"nom": "Jardin"}],
        ],
    )

    plafond: Montant | None = edge(
        label="APLAFOND",
        default=None,
        description="Plafond / limite de garantie si précisé.",
        examples=[{"valeur": 15000.0, "devise": "EUR"}, "15 000 €"],
    )

    franchises: list[Franchise] = edge(
        label="AFRANCHISE",
        default_factory=list,
        description="Franchises applicables à cette garantie.",
        examples=[[{"montant": "380 €", "type": "Fixe", "contexte": "Par sinistre"}]],
    )

    conditions: list[Condition] = edge(
        label="ACONDITION",
        default_factory=list,
        description="Conditions / mesures de sécurité / obligations liées à cette garantie.",
        examples=[[{"texte": "Faire procéder au ramonage avant chaque hiver."}]],
    )

    exclusions_specifiques: list[Exclusion] = edge(
        label="AEXCLUSION",
        default_factory=list,
        validation_alias=AliasChoices(
            "exclusions_specifiques", "exclusions", "exclusionsspecifiques"
        ),
        description=(
            "Exclusions spécifiques à cette garantie (typiquement sous un bloc 'EXCLUSIONS SPÉCIFIQUES' "
            "dans la section de la garantie). "
            "Ne pas y mettre les exclusions 'communes à toutes les garanties' (Article 7), "
            "qui doivent aller dans AssuranceMRH.exclusions_communes."
        ),
        examples=[[{"exclusion_id": "defaut-entretien"}]],
    )

    @field_validator("biens_couverts", mode="before")
    @classmethod
    def filtrer_biens_couverts(cls, v: Any) -> Any:
        return _filtrer_liste(v, "biens_couverts")

    @field_validator("exclusions_specifiques", mode="before")
    @classmethod
    def filtrer_exclusions_specifiques(cls, v: Any) -> Any:
        return _filtrer_liste(v, "exclusions_specifiques", champs_requis=["exclusion_id"])

    @model_validator(mode="after")
    def auto_lier_biens_evidents(self) -> Garantie:
        """
        Garde-fou minimal : si une garantie est *manifestement* un bien (Jardin/Piscine)
        ou une protection ciblée (Protection du mobilier), on crée la référence Bien(nom=...).
        """
        if self.biens_couverts:
            return self
        if not self.nom:
            return self

        mapping = {
            "Jardin": "Jardin",
            "Piscine": "Piscine",
            "Protection du mobilier": "Mobilier",
        }
        bien_nom = mapping.get(self.nom.strip())
        if bien_nom:
            self.biens_couverts = [Bien(nom=bien_nom)]
        return self


class Option(BaseModel):
    """
    Option / pack / extension (ex. 'Dépannage d'urgence', 'Piscine', etc.).
    """

    model_config = ConfigDict(graph_id_fields=["nom"], extra="ignore", populate_by_name=True)

    nom: str = Field(
        ...,
        description="Nom de l'option tel qu'écrit dans le document (identifiant stable).",
        examples=[
            "Dommages électriques",
            "Rééquipement neuf",
            "Dépannage d'urgence",
            "Jardin",
            "Piscine",
        ],
    )
    description: str | None = Field(
        None,
        description="Description de l'option (objectif, périmètre).",
        examples=[
            "Indemnisation des dommages causés par une surtension ou la foudre.",
            "Prise en charge d'un dépannage de serrurerie/électricité/plomberie intérieure.",
        ],
    )

    biens_couverts: list[Bien] = edge(
        label="COUVREBIEN",
        default_factory=list,
        validation_alias=AliasChoices("biens_couverts", "bienscouverts"),
        description=(
            "Biens couverts par l'option. "
            "Si l'option correspond à un bien explicite (ex. 'Piscine', 'Jardin'), "
            "ajouter ce bien au minimum via son 'nom'."
        ),
        examples=[[{"nom": "Piscine"}], [{"nom": "Jardin"}]],
    )

    etend_garanties: list[Garantie] = edge(
        label="ETENDGARANTIE",
        default_factory=list,
        validation_alias=AliasChoices("etend_garanties", "etendgaranties"),
        description="Si indiqué, liste des garanties que l'option étend/active.",
        examples=[[{"nom": "Incendie et événements assimilés"}]],
    )

    @field_validator("biens_couverts", mode="before")
    @classmethod
    def filtrer_biens_couverts(cls, v: Any) -> Any:
        return _filtrer_liste(v, "biens_couverts")

    @field_validator("etend_garanties", mode="before")
    @classmethod
    def filtrer_etend_garanties(cls, v: Any) -> Any:
        return _filtrer_liste(v, "etend_garanties", champs_requis=["nom"])

    @model_validator(mode="after")
    def auto_lier_biens_evidents(self) -> Option:
        if self.biens_couverts:
            return self
        if not self.nom:
            return self
        if self.nom.strip() in {"Jardin", "Piscine"}:
            self.biens_couverts = [Bien(nom=self.nom.strip())]
        return self


class Offre(BaseModel):
    """
    Offre / formule (ESSENTIELLE, CONFORT, CONFORT PLUS, PNO...).
    """

    model_config = ConfigDict(graph_id_fields=["nom"], extra="ignore", populate_by_name=True)

    nom: str = Field(
        ...,
        description="Nom de la formule tel qu'écrit (identifiant stable pour déduplication).",
        examples=["ESSENTIELLE", "CONFORT", "CONFORT PLUS", "PROPRIÉTAIRE NON OCCUPANT"],
    )
    niveau: int | None = Field(
        None,
        description="Niveau si le document l'indique (rare).",
        examples=[1, 2, 3],
    )

    statut_occupation: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("statut_occupation", "statutoccupation"),
        description="Profil d'occupation (ex. 'locataire', 'propriétaire occupant', 'propriétaire non occupant').",
        examples=[["locataire"], ["propriétaire occupant"], ["propriétaire non occupant"]],
    )

    garanties_incluses: list[Garantie] = edge(
        label="INCLUTGARANTIE",
        default_factory=list,
        validation_alias=AliasChoices("garanties_incluses", "garantiesincluses"),
        description="Garanties incluses / non optionnelles dans le tableau des garanties.",
        examples=[[{"nom": "Dégâts des eaux"}, {"nom": "Incendie et événements assimilés"}]],
    )

    garanties_optionnelles: list[Garantie] = edge(
        label="GARANTIEOPTIONNELLE",
        default_factory=list,
        validation_alias=AliasChoices("garanties_optionnelles", "garantiesoptionnelles"),
        description="Garanties marquées 'En option' dans le tableau.",
        examples=[[{"nom": "Dommages électriques"}, {"nom": "Piscine"}]],
    )

    options_disponibles: list[Option] = edge(
        label="PROPOSEOPTION",
        default_factory=list,
        validation_alias=AliasChoices("options_disponibles", "optionsdisponibles"),
        description="Options/packs disponibles pour cette formule (si le document les distingue).",
        examples=[[{"nom": "Dépannage d'urgence"}, {"nom": "Rééquipement neuf"}]],
    )

    notes: str | None = Field(
        None,
        description="Notes libres liées à l'offre (mentions du tableau, remarques).",
        examples=[
            "Certaines garanties dépendent du lieu assuré.",
            "Voir limites et plafonds en conditions spéciales.",
        ],
    )

    @field_validator("garanties_incluses", mode="before")
    @classmethod
    def filtrer_garanties_incluses(cls, v: Any) -> Any:
        return _filtrer_liste(v, "garanties_incluses", champs_requis=["nom"])

    @field_validator("garanties_optionnelles", mode="before")
    @classmethod
    def filtrer_garanties_optionnelles(cls, v: Any) -> Any:
        return _filtrer_liste(v, "garanties_optionnelles", champs_requis=["nom"])

    @field_validator("options_disponibles", mode="before")
    @classmethod
    def filtrer_options_disponibles(cls, v: Any) -> Any:
        return _filtrer_liste(v, "options_disponibles", champs_requis=["nom"])


class AssuranceMRH(BaseModel):
    """
    Racine du document MRH.
    """

    model_config = ConfigDict(
        graph_id_fields=["reference_document"], extra="ignore", populate_by_name=True
    )

    reference_document: str = Field(
        ...,
        validation_alias=AliasChoices("reference_document", "referencedocument"),
        description="Référence/identifiant du document (couverture, pied de page) si présent.",
        examples=["CGV-MRH-2023", "HABITATION 2023-10"],
    )
    assureur: str | None = Field(
        None,
        description="Assureur / marque si présent.",
        examples=["Direct Assurance", "AXA", "MMA"],
    )
    date_version: str | None = Field(
        None,
        validation_alias=AliasChoices("date_version", "dateversion"),
        description="Date/version/édition si présente.",
        examples=["2023-10-01", "Édition Janvier 2024"],
    )
    nom_produit: str | None = Field(
        None,
        validation_alias=AliasChoices("nom_produit", "nomproduit"),
        description="Nom du produit si présent.",
        examples=["Assurance Habitation", "Multirisque Habitation", "MRH"],
    )

    offres: list[Offre] = edge(
        label="AOFFRE",
        default_factory=list,
        description="Liste des offres/formules présentes dans le document (tableau des garanties).",
        examples=[[{"nom": "ESSENTIELLE"}, {"nom": "CONFORT"}]],
    )

    @field_validator("offres", mode="before")
    @classmethod
    def filtrer_offres(cls, v: Any) -> Any:
        return _filtrer_liste(v, "offres", champs_requis=["nom"])
