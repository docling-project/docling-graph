"""
Pydantic templates for Insurance documents.

This file is self-contained and has no external template dependencies.
It includes all necessary sub-models and provides examples for each field.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional, Union, Any
from datetime import date


# --- Edge Helper Function ---

def Edge(label: str, **kwargs: Any) -> Any:
    """Helper function to create a Pydantic Field with edge metadata."""
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)

# --- Reusable Component: MonetaryAmount ---

class MonetaryAmount(BaseModel):
    """
    A component model to represent a monetary value with its currency.
    No graph_id_fields, as it's a value object.
    """
    
    model_config = ConfigDict(is_entity=False)  # Component: deduplicate by content
    
    value: float = Field(
        ...,
        description="The numeric value of the amount",
        examples=[500.00, 150000.00, 75.50, 1200.00, 89.90]
    )

    currency: Optional[str] = Field(
        None,
        description="The ISO 4217 currency code",
        examples=["EUR", "USD", "CAD", "CHF"]
    )
    
    @field_validator('value')
    @classmethod
    def validate_positive(cls, v):
        """Ensure the value is non-negative."""
        if v < 0:
            raise ValueError('Monetary amount must be non-negative')
        return v
    
    def __str__(self):
        return f"{self.value} {self.currency or ''}".strip()

# --- Reusable Component: Address ---

class Address(BaseModel):
    """Represents a physical address entity."""    
    model_config = ConfigDict(is_entity=False)
    
    street_address: Optional[str] = Field(
        None,
        description="Street name and number",
        examples=["1 Rue de Rivoli", "221B Baker Street", "123 Avenue des Champs-Élysées", "90 Boulevard Voltaire"]
    )
    
    city: Optional[str] = Field(
        None,
        description="City",
        examples=["Paris", "London", "Lyon", "Marseille"]
    )
    
    state_or_province: Optional[str] = Field(
        None,
        description="State, province, or region",
        examples=["Île-de-France", "London", "Auvergne-Rhône-Alpes"]
    )
    
    postal_code: Optional[str] = Field(
        None,
        description="Postal or ZIP code",
        examples=["75001", "NW1 6XE", "69002", "13001"]
    )
    
    country: Optional[str] = Field(
        None,
        description="Country",
        examples=["France", "United Kingdom", "FR"]
    )
    
    def __str__(self):
        parts = [self.street_address, self.city, self.state_or_province, self.postal_code, self.country]
        return ", ".join(p for p in parts if p)

# --- Reusable Entity: Organization ---

class Organization(BaseModel):
    """
    A generic model for any organization (insurer, etc.).
    Its name is its unique identifier in the graph.
    """
    
    model_config = ConfigDict(graph_id_fields=['name'])
    
    name: str = Field(
        ...,
        description="The legal name of the organization",
        examples=["AXA Assurance", "Allianz France", "MAIF", "Generali", "MACIF"]
    )
    
    phone: Optional[str] = Field(
        None,
        description="Contact phone number",
        examples=["+33 1 40 75 57 00", "01 55 92 30 00", "09 69 39 30 00"]
    )
    
    email: Optional[str] = Field(
        None,
        description="Contact email address",
        examples=["service.client@axa.fr", "contact@allianz.fr", "info@maif.fr"]
    )
    
    website: Optional[str] = Field(
        None,
        description="Official website",
        examples=["www.axa.fr", "www.allianz.fr", "www.maif.fr"]
    )
    
    tax_id: Optional[str] = Field(
        None,
        description="Tax ID, VAT ID, SIREN, or other official identifier",
        examples=["572 093 920", "775 665 631", "775 689 309"]
    )
    
    # --- Edge Definition ---
    addresses: List[Address] = Edge(
        label="LOCATED_AT",
        default_factory=list,
        description="List of physical addresses for the organization"
    )
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Basic email format validation."""
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v
    
    def __str__(self):
        return self.name

# --- Reusable Entity: Person ---

class Person(BaseModel):
    """
    A generic model for a person (policyholder, etc.).
    A person is uniquely identified by their full name and date of birth.
    """
    
    model_config = ConfigDict(graph_id_fields=['first_name', 'last_name', 'date_of_birth'])
    
    first_name: Optional[str] = Field(
        None,
        description="The person's given name(s)",
        examples=["Jean", "Sophie", "Pierre", "Marie", "Luc"]
    )
    
    last_name: Optional[str] = Field(
        None,
        description="The person's family name (surname)",
        examples=["Dupont", "Martin", "Bernard", "Dubois", "Thomas"]
    )
    
    date_of_birth: Optional[date] = Field(
        None,
        description="Date of birth in YYYY-MM-DD format",
        examples=["1985-03-12", "1990-06-20", "1978-11-05"]
    )
    
    place_of_birth: Optional[str] = Field(
        None,
        description="City and/or country of birth",
        examples=["Paris", "Lyon (France)", "Marseille"]
    )
    
    gender: Optional[str] = Field(
        None,
        description="Gender or sex of the person",
        examples=["M", "F", "Male", "Female"]
    )
    
    nationality: Optional[str] = Field(
        None,
        description="Nationality of the person",
        examples=["Française", "French", "Belge"]
    )
    
    phone: Optional[str] = Field(
        None,
        description="Contact phone number",
        examples=["+33 7 98 76 54 32", "06 12 34 56 78", "01 23 45 67 89"]
    )
    
    email: Optional[str] = Field(
        None,
        description="Contact email address",
        examples=["jean.dupont@email.com", "sophie.martin@gmail.com", "pierre.bernard@orange.fr"]
    )
    
    # --- Edge Definition ---
    addresses: List[Address] = Edge(
        label="LIVES_AT",
        default_factory=list,
        description="List of physical addresses (e.g., home, work)"
    )
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Basic email format validation."""
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v
    
    def __str__(self):
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p)

# --- Document-Specific Entity: Guarantee ---

class Guarantee(BaseModel):
    """
    A single guarantee or coverage item.
    Uniquely identified by its name within the policy.
    """
    
    model_config = ConfigDict(graph_id_fields=['name'])
    
    name: str = Field(
        ...,
        description="Name of the coverage guarantee",
        examples=[
            "Dégâts des eaux",
            "Incendie et événements assimilés",
            "Vol et vandalisme",
            "Bris de vitre",
            "Responsabilité civile",
            "Événements climatiques",
            "Catastrophes naturelles",
            "Assistance",
            "Défense Pénale et Recours",
            "Attentat"
        ]
    )
    
    description: Optional[str] = Field(
        None,
        description="Detailed description of what the guarantee covers",
        examples=[
            "Couvre les dommages à l'intérieur du bien provoqués par la fuite, la rupture ou le débordement des canalisations",
            "Protection contre l'incendie, l'explosion, l'implosion et la chute de la foudre",
            "Couvre le vol, tentative de vol et vandalisme à l'intérieur des locaux privatifs",
            "Indemnise les dommages corporels et matériels causés aux tiers"
        ]
    )
    
    coverage_conditions: Optional[List[str]] = Field(
        default_factory=list,
        description="Specific conditions that must be met for coverage to apply",
        examples=[
            ["Effraction extérieure des bâtiments", "Escalade des bâtiments", "Menaces ou violences"],
            ["Phénomènes d'intensité anormale", "Publication au Journal Officiel"],
            ["Respect des mesures de sécurité", "Maintenance annuelle obligatoire"]
        ]
    )
    
    coverage_limit: Optional[MonetaryAmount] = Field(
        None,
        description="The maximum amount covered for this guarantee"
    )
    
    deductible: Optional[MonetaryAmount] = Field(
        None,
        description="The deductible amount (franchise) that the policyholder must pay"
    )
    
    exclusions: Optional[List[str]] = Field(
        default_factory=list,
        description="List of exclusions that are not covered by this guarantee",
        examples=[
            ["Dommages corporels subis par les personnes assurées", "Frais de réparation des biens à l'origine du sinistre"],
            ["Usure normale", "Défaut d'entretien", "Vice de construction"],
            ["Faute intentionnelle", "Usage non conforme"]
        ]
    )
    
    def __str__(self):
        return self.name

# --- Document-Specific Entity: InsurancePlan ---

class InsurancePlan(BaseModel):
    """
    A specific plan or option (e.g., "Basic", "Premium").
    Uniquely identified by its name.
    """
    
    model_config = ConfigDict(graph_id_fields=['name'])
    
    name: str = Field(
        ...,
        description="The name of the insurance plan or formula",
        examples=[
            "Formule Essentielle",
            "Formule Confort",
            "Formule Confort Plus",
            "Formule Propriétaire Non Occupant",
            "Formule Tous Risques"
        ]
    )
    
    description: Optional[str] = Field(
        None,
        description="Brief description of the plan and its target customers",
        examples=[
            "Formule de base avec garanties essentielles",
            "Couverture étendue pour propriétaires occupants",
            "Protection maximale incluant objets de valeur"
        ]
    )
    
    base_price: Optional[MonetaryAmount] = Field(
        None,
        description="Starting or example price for this plan"
    )
    
    # --- Edge Definition ---
    guarantees: List[Guarantee] = Edge(
        label="INCLUDES_GUARANTEE",
        default_factory=list,
        description="List of guarantees included in this plan"
    )
    
    available_options: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional coverages that can be added to this plan",
        examples=[
            ["Dommages électriques", "Rééquipement à neuf", "Jardin", "Piscine"],
            ["Assurance scolaire", "Protection du mobilier", "Dépannage d'urgence"]
        ]
    )
    
    def __str__(self):
        return self.name

# --- Root Document Model: InsuranceTerms ---

class InsuranceTerms(BaseModel):
    """
    The root model for insurance terms and conditions or options comparison documents.
    Can represent complete CGV or just a comparison of formulas/plans.
    """
    
    model_config = ConfigDict(graph_id_fields=['document_reference'])
    
    # Changed to Optional with default to handle options-only documents
    document_reference: Optional[str] = Field(
        default="UNKNOWN",
        description="Reference identifier or version of the terms document",
        examples=[
            "HABITATION07.25",
            "CGV-AUTO-2024",
            "CONDITIONS-GENERALES-V3.2",
            "OPTIONS-COMPARISON"
        ]
    )
    
    product_name: Optional[str] = Field(
        default=None,
        description="Name of the insurance product these terms apply to",
        examples=[
            "Direct Assurance Habitation",
            "Assurance Auto Tous Risques",
            "Mutuelle Santé Famille",
            "Assurance Multirisque Habitation"
        ]
    )
    
    issuer: Optional[str] = Field(
        default=None,
        description="Name of the insurance company issuing these terms",
        examples=[
            "AXA France IARD",
            "Allianz France",
            "MAIF",
            "Direct Assurance (Avanssur)"
        ]
    )
    
    effective_date: Optional[date] = Field(
        None,
        description="Date when these terms become effective (YYYY-MM-DD)",
        examples=["2024-07-01", "2025-01-01", "2024-09-15"]
    )
    
    document_type: Optional[str] = Field(
        None,
        description="Type of insurance terms document",
        examples=[
            "Conditions Générales",
            "Conditions Générales et Spéciales",
            "Conditions Particulières",
            "Guide des garanties",
            "Comparatif des formules"
        ]
    )
    
    contract_duration: Optional[str] = Field(
        None,
        description="Standard duration and renewal terms",
        examples=[
            "1 an renouvelable tacitement",
            "Annuel avec reconduction automatique",
            "Durée indéterminée"
        ]
    )
    
    # --- Edge Definitions ---
    available_plans: List[InsurancePlan] = Edge(
        label="OFFERS_PLAN",
        default_factory=list,
        description="Insurance plans/formulas offered in these terms"
    )
    
    # Additional coverage details
    territorial_scope: Optional[List[str]] = Field(
        default_factory=list,
        description="Geographic areas where coverage applies",
        examples=[
            ["France métropolitaine"],
            ["France métropolitaine", "DROM", "Union Européenne"],
            ["Monde entier pour séjours < 3 mois"]
        ]
    )
    
    common_exclusions: Optional[List[str]] = Field(
        default_factory=list,
        description="Exclusions that apply to all guarantees",
        examples=[
            ["Faute intentionnelle ou dolosive de l'assuré", "Guerre civile ou étrangère", "Dommages nucléaires"],
            ["Usure normale", "Défaut d'entretien", "Vice de construction"],
            ["Dommages résultant d'une activité professionnelle"]
        ]
    )
    
    claims_procedure: Optional[str] = Field(
        None,
        description="General procedure for filing claims",
        examples=[
            "Déclaration sous 5 jours ouvrés, 2 jours en cas de vol",
            "Déclaration via espace personnel ou application mobile",
            "Contacter l'assistance 24h/24 au numéro indiqué"
        ]
    )
    
    prescription_period: Optional[str] = Field(
        None,
        description="Time limit for legal action on claims",
        examples=[
            "2 ans à compter de l'événement",
            "5 ans pour mouvements de terrain liés à la sécheresse",
            "10 ans pour dommages corporels"
        ]
    )
    
    cancellation_terms: Optional[str] = Field(
        None,
        description="Terms and conditions for contract cancellation",
        examples=[
            "Résiliation annuelle avec préavis de 2 mois",
            "Résiliation à tout moment après 1 an de contrat",
            "Résiliation à l'échéance annuelle par lettre recommandée"
        ]
    )
    
    @field_validator('effective_date')
    @classmethod
    def validate_date_not_future(cls, v):
        """Ensure effective date is not too far in the future."""
        if v and v > date.today().replace(year=date.today().year + 2):
            raise ValueError('Effective date cannot be more than 2 years in the future')
        return v
    
    def __str__(self):
        return f"{self.product_name or 'Insurance'} - {self.document_reference}"
