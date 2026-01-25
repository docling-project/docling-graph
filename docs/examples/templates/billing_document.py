"""
Comprehensive Invoice/Bill extraction template.

Extracts structured data from invoices, bills, credit notes, debit notes, pro forma invoices,
and receipts following EN 16931, Peppol BIS, and UBL standards.

This template supports:
- Multiple document types (Invoice, Credit Note, Debit Note, Pro Forma, Receipt)
- Complex party relationships (issuer, buyer, payee, tax representative)
- Multi-level allowances and charges (document, line, price level)
- Comprehensive tax handling (VAT categories, rates, exemptions)
- Multiple payment methods (bank transfer, card, direct debit, QR codes)
- Delivery information and attachments
- Dual currency support (document currency + tax currency)

The schema is optimized for extraction across all model tiers (SIMPLE/STANDARD/ADVANCED)
with clear field descriptions, extraction hints, and diverse examples.

Version: 1.0.0
Last Updated: 2026-01-25
"""

import re
from datetime import date
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def edge(label: str, **kwargs: Any) -> Any:
    """
    Helper function to create a Pydantic Field with edge metadata.
    The 'edge_label' defines the type of relationship in the knowledge graph.

    Args:
        label: Edge label in ALL_CAPS_WITH_UNDERSCORES format
        **kwargs: Additional Field parameters (default, description, etc.)

    Returns:
        Pydantic Field with edge metadata
    """
    # If 'default' is not provided in kwargs, use ... (required)
    if "default" not in kwargs and "default_factory" not in kwargs:
        kwargs["default"] = ...
    return Field(json_schema_extra={"edge_label": label}, **kwargs)


def _normalize_enum(enum_cls: type[Enum], v: Any) -> Any:
    """
    Normalize enum values to handle various input formats from LLMs.

    Accepts enum instances, value strings, or member names in various formats:
    'VALUE', 'value', 'Value', 'VALUE_NAME', 'value-name', etc.
    Falls back to 'OTHER' member if present.

    Args:
        enum_cls: The enum class to normalize to
        v: Value to normalize (enum instance, string, etc.)

    Returns:
        Normalized enum member

    Raises:
        ValueError: If value cannot be normalized and no OTHER fallback exists
    """
    if isinstance(v, enum_cls):
        return v

    if isinstance(v, str):
        # Normalize to alphanumeric lowercase
        key = re.sub(r"[^A-Za-z0-9]+", "", v).lower()

        # Build mapping of normalized names/values to enum members
        mapping = {}
        for member in enum_cls:
            normalized_name = re.sub(r"[^A-Za-z0-9]+", "", member.name).lower()
            normalized_value = re.sub(r"[^A-Za-z0-9]+", "", member.value).lower()
            mapping[normalized_name] = member
            mapping[normalized_value] = member

        if key in mapping:
            return mapping[key]

        # Last attempt: direct value match
        try:
            return enum_cls(v)
        except Exception:
            # Safe fallback to OTHER if present
            if "OTHER" in enum_cls.__members__:
                return enum_cls.OTHER
            raise

    raise ValueError(f"Cannot normalize {v} to {enum_cls}")


# =============================================================================
# ENUMS
# =============================================================================


class DocumentType(str, Enum):
    """Type of billing document."""

    INVOICE = "Invoice"
    CREDIT_NOTE = "Credit Note"
    DEBIT_NOTE = "Debit Note"
    PRO_FORMA = "Pro Forma"
    RECEIPT = "Receipt"
    OTHER = "Other"


class TaxType(str, Enum):
    """Type of tax applied."""

    VAT = "VAT"
    GST = "GST"
    SALES_TAX = "Sales Tax"
    WITHHOLDING = "Withholding Tax"
    EXCISE = "Excise"
    CUSTOMS = "Customs"
    OTHER = "Other"


class PaymentMeans(str, Enum):
    """Payment method type."""

    BANK_TRANSFER = "Bank Transfer"
    CARD = "Card"
    CASH = "Cash"
    DIRECT_DEBIT = "Direct Debit"
    CHEQUE = "Cheque"
    CREDIT = "Credit"
    QR_CODE = "QR Code"
    OTHER = "Other"


class AllowanceChargeScope(str, Enum):
    """Scope where allowance/charge applies."""

    DOCUMENT = "Document"
    LINE = "Line"
    PRICE = "Price"


class DocumentReferenceType(str, Enum):
    """Type of document reference."""

    PURCHASE_ORDER = "Purchase Order"
    CONTRACT = "Contract"
    SALES_ORDER = "Sales Order"
    DESPATCH_ADVICE = "Despatch Advice"
    RECEIVING_ADVICE = "Receiving Advice"
    PREVIOUS_INVOICE = "Previous Invoice"
    PROJECT = "Project"
    TENDER = "Tender"
    BUYER_REFERENCE = "Buyer Reference"
    ACCOUNTING_REFERENCE = "Accounting Reference"
    OTHER = "Other"


class ItemIdentifierType(str, Enum):
    """Type of item identifier."""

    SELLER_ITEM_ID = "Seller Item ID"
    BUYER_ITEM_ID = "Buyer Item ID"
    GTIN = "GTIN"
    UUID = "UUID"
    OTHER = "Other"


# =============================================================================
# COMPONENTS (Value Objects - is_entity=False)
# =============================================================================


class PostalAddress(BaseModel):
    """
    Physical postal address component.
    Deduplicated by content - identical addresses share the same graph node.
    """

    model_config = ConfigDict(is_entity=False)

    street_lines: List[str] = Field(
        default_factory=list,
        description=(
            "Street address lines including street name, number, building, floor, etc. "
            "Look for address fields, street names, building numbers. "
            "Extract each line separately. Common labels: 'Address', 'Street', 'Adresse', 'Rue'."
        ),
        examples=[
            ["123 Main Street", "Building A, Floor 3"],
            ["45 Avenue des Champs-Élysées"],
            ["Calle Mayor 10", "Edificio Central"],
        ],
    )

    postal_code: str | None = Field(
        None,
        description=(
            "Postal or ZIP code. "
            "Look for numeric codes near city name. "
            "Common labels: 'Postal Code', 'ZIP', 'Code Postal', 'PLZ'. "
            "Keep original format (spaces, dashes)."
        ),
        examples=["75001", "SW1A 1AA", "10001", "28001"],
    )

    city: str | None = Field(
        None,
        description=(
            "City or town name. "
            "Look for city name in address section. "
            "Common labels: 'City', 'Town', 'Ville', 'Ciudad'."
        ),
        examples=["Paris", "London", "Madrid", "New York"],
    )

    region: str | None = Field(
        None,
        description=(
            "State, province, or region. "
            "Look for administrative division above city level. "
            "Common labels: 'State', 'Province', 'Region', 'County'."
        ),
        examples=["Île-de-France", "California", "Madrid", "Greater London"],
    )

    country: str | None = Field(
        None,
        description=(
            "Country name or ISO 3166-1 alpha-2 code. "
            "Look for country name or 2-letter code. "
            "Common labels: 'Country', 'Pays', 'País'. "
            "Accept both full names and codes."
        ),
        examples=["France", "FR", "United Kingdom", "GB", "Spain", "ES"],
    )

    def __str__(self) -> str:
        """Format address for display."""
        parts = [
            ", ".join(self.street_lines) if self.street_lines else None,
            self.postal_code,
            self.city,
            self.region,
            self.country,
        ]
        return ", ".join(p for p in parts if p)


class ContactPoint(BaseModel):
    """
    Contact information component.
    Deduplicated by content - identical contact info shares the same node.
    """

    model_config = ConfigDict(is_entity=False)

    email: str | None = Field(
        None,
        description=(
            "Email address. "
            "Look for text containing '@' symbol. "
            "Common labels: 'Email', 'E-mail', 'Contact', 'Courriel'. "
            "Normalize to lowercase."
        ),
        examples=["contact@company.com", "info@organization.fr", "support@business.co.uk"],
    )

    phone: str | None = Field(
        None,
        description=(
            "Phone number with country code if present. "
            "Look for phone numbers, preserve formatting. "
            "Common labels: 'Phone', 'Tel', 'Telephone', 'Mobile', 'Tél'. "
            "Include country code prefix like +33, +44, +1."
        ),
        examples=["+33 1 23 45 67 89", "+44 20 7123 4567", "+1 (555) 123-4567"],
    )

    website: str | None = Field(
        None,
        description=(
            "Website URL. "
            "Look for web addresses starting with http://, https://, or www. "
            "Common labels: 'Website', 'Web', 'URL', 'Site Web'."
        ),
        examples=["https://www.company.com", "www.organization.fr", "https://business.co.uk"],
    )

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: Any) -> Any:
        """Normalize email to lowercase and strip whitespace."""
        if v and isinstance(v, str):
            return v.lower().strip()
        return v


class ElectronicAddress(BaseModel):
    """
    Electronic invoicing routing endpoint component.
    Used for e-invoicing systems like Peppol.
    """

    model_config = ConfigDict(is_entity=False)

    scheme: str | None = Field(
        None,
        description=(
            "Electronic address scheme identifier. "
            "Look for e-invoicing identifiers, Peppol IDs, EDI codes. "
            "Common schemes: 'Peppol', 'GLN', 'DUNS', 'EDI'."
        ),
        examples=["Peppol", "GLN", "DUNS", "EDI"],
    )

    value: str | None = Field(
        None,
        description=(
            "Electronic address value or identifier. "
            "The actual routing address or participant ID. "
            "Format depends on scheme."
        ),
        examples=["9482:123456789", "1234567890123", "123456789"],
    )


class MonetaryAmount(BaseModel):
    """
    Monetary value with currency component.
    Deduplicated by content - same value and currency share a node.
    """

    model_config = ConfigDict(is_entity=False)

    value: float = Field(
        ...,
        description=(
            "Numeric monetary amount. "
            "Extract numeric value only, remove currency symbols and formatting. "
            "Convert formats like '1,234.56' or '1.234,56' to decimal number. "
            "Look for amounts near currency codes or symbols."
        ),
        examples=[1234.56, 500.00, 89.99, 15000.00],
    )

    currency: str | None = Field(
        None,
        description=(
            "ISO 4217 currency code (3 uppercase letters). "
            "Look for currency symbols (€, $, £) or codes (EUR, USD, GBP). "
            "Convert symbols to codes: € → EUR, $ → USD, £ → GBP. "
            "Common labels: 'Currency', 'Devise', 'Moneda'."
        ),
        examples=["EUR", "USD", "GBP", "CHF"],
    )

    @field_validator("value")
    @classmethod
    def validate_positive(cls, v: Any) -> Any:
        """Ensure amount is non-negative."""
        if v < 0:
            raise ValueError(f"Monetary amount must be non-negative, got {v}")
        return v

    @field_validator("currency")
    @classmethod
    def validate_currency_format(cls, v: Any) -> Any:
        """Ensure currency is 3 uppercase letters (ISO 4217)."""
        if v and not (len(v) == 3 and v.isupper() and v.isalpha()):
            raise ValueError(f"Currency must be 3 uppercase letters (ISO 4217), got {v}")
        return v

    def __str__(self) -> str:
        """Format amount for display."""
        return f"{self.value} {self.currency or ''}".strip()


class Quantity(BaseModel):
    """
    Quantity with unit of measure component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    value: float = Field(
        ...,
        description=(
            "Numeric quantity value. "
            "Extract numeric value from quantity fields. "
            "Look for numbers near unit codes or quantity labels. "
            "Common labels: 'Quantity', 'Qty', 'Quantité', 'Cantidad'."
        ),
        examples=[1.0, 10.5, 100.0, 2.5],
    )

    unit_code: str | None = Field(
        None,
        description=(
            "UN/CEFACT unit of measure code or common unit abbreviation. "
            "Look for unit codes or abbreviations after quantities. "
            "Common units: 'EA' (each), 'KG' (kilogram), 'MTR' (meter), 'LTR' (liter), "
            "'HUR' (hour), 'DAY' (day), 'PC' (piece), 'BOX', 'SET'."
        ),
        examples=["EA", "KG", "MTR", "LTR", "HUR", "PC", "BOX"],
    )

    @field_validator("value")
    @classmethod
    def validate_positive(cls, v: Any) -> Any:
        """Ensure quantity is positive."""
        if v <= 0:
            raise ValueError(f"Quantity must be positive, got {v}")
        return v


class Period(BaseModel):
    """
    Date period component for service periods, invoice periods, etc.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    start_date: date | None = Field(
        None,
        description=(
            "Period start date in YYYY-MM-DD format. "
            "Look for 'From', 'Start', 'Début', 'Desde' labels. "
            "Parse various date formats (DD/MM/YYYY, MM-DD-YYYY) and normalize to YYYY-MM-DD."
        ),
        examples=["2024-01-01", "2024-06-15", "2023-12-01"],
    )

    end_date: date | None = Field(
        None,
        description=(
            "Period end date in YYYY-MM-DD format. "
            "Look for 'To', 'End', 'Fin', 'Hasta' labels. "
            "Parse various date formats and normalize to YYYY-MM-DD."
        ),
        examples=["2024-01-31", "2024-06-30", "2023-12-31"],
    )

    @model_validator(mode="after")
    def validate_date_order(self) -> Self:
        """Ensure end_date is after start_date."""
        if self.start_date and self.end_date:
            if self.end_date < self.start_date:
                raise ValueError("end_date must be after start_date")
        return self


# =============================================================================
# ENTITY COMPONENTS (for Party identifiers)
# =============================================================================


class PartyIdentifier(BaseModel):
    """
    Non-tax party identifier component (GLN, DUNS, local registration numbers).
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    scheme: str | None = Field(
        None,
        description=(
            "Identifier scheme name. "
            "Look for identifier type labels. "
            "Common schemes: 'GLN' (Global Location Number), 'DUNS' (Dun & Bradstreet), "
            "'SIRET' (France), 'CIF' (Spain), 'Company Number', 'Registration Number'."
        ),
        examples=["GLN", "DUNS", "SIRET", "CIF", "Company Number"],
    )

    value: str | None = Field(
        None,
        description=(
            "Identifier value. "
            "The actual identifier number or code. "
            "Keep original format including spaces and dashes."
        ),
        examples=["1234567890123", "123456789", "12345678901234", "A12345678"],
    )


class TaxRegistration(BaseModel):
    """
    Tax registration component (VAT number, tax ID, enterprise number).
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    scheme: str | None = Field(
        None,
        description=(
            "Tax registration scheme. "
            "Look for tax ID type labels. "
            "Common schemes: 'VAT' (Value Added Tax), 'TAX_ID', 'EIN' (Employer ID), "
            "'GST' (Goods and Services Tax), 'Enterprise Number'."
        ),
        examples=["VAT", "TAX_ID", "EIN", "GST", "Enterprise Number"],
    )

    value: str | None = Field(
        None,
        description=(
            "Tax registration number. "
            "Look for VAT numbers, tax IDs near party information. "
            "Common labels: 'VAT Number', 'Tax ID', 'TVA', 'NIF', 'RFC'. "
            "Keep original format including country prefixes."
        ),
        examples=["FR12345678901", "GB123456789", "DE123456789", "123-45-6789"],
    )

    country: str | None = Field(
        None,
        description=(
            "Country of tax registration (ISO 3166-1 alpha-2 code). "
            "Extract from VAT number prefix or separate country field. "
            "2-letter country codes."
        ),
        examples=["FR", "GB", "DE", "ES", "US"],
    )


# =============================================================================
# ENTITIES (Unique Objects - graph_id_fields)
# =============================================================================


class Party(BaseModel):
    """
    Party entity representing any organization or person role.
    Can be issuer, buyer, payee, tax representative, etc.
    Uniquely identified by name and primary tax registration.
    """

    model_config = ConfigDict(graph_id_fields=["name", "primary_tax_id"])

    name: str = Field(
        ...,
        description=(
            "Full legal name of the party (organization or person). "
            "Look for company names, legal names in headers or party sections. "
            "Common labels: 'Company Name', 'Name', 'Legal Name', 'Raison Sociale', "
            "'Seller', 'Buyer', 'Supplier', 'Customer'. "
            "Include legal suffixes like 'Ltd', 'Inc', 'SA', 'GmbH', 'SL'."
        ),
        examples=[
            "Acme Corporation Ltd",
            "Tech Solutions Inc",
            "Global Industries SA",
            "Consulting Services GmbH",
        ],
    )

    legal_name: str | None = Field(
        None,
        description=(
            "Official legal name if different from trading name. "
            "Look for 'Legal Name', 'Registered Name', 'Official Name'. "
            "May include full legal entity type."
        ),
        examples=[
            "Acme Corporation Limited",
            "Tech Solutions Incorporated",
            "Global Industries Société Anonyme",
        ],
    )

    department: str | None = Field(
        None,
        description=(
            "Department or division name within organization. "
            "Look for department, division, or unit names. "
            "Common labels: 'Department', 'Division', 'Unit', 'Service'."
        ),
        examples=["Sales Department", "Accounting Division", "Customer Service"],
    )

    is_public_sector: bool | None = Field(
        None,
        description=(
            "Whether this is a public sector entity. "
            "Look for government, public administration indicators. "
            "True for government agencies, municipalities, public institutions."
        ),
        examples=[True, False],
    )

    primary_tax_id: str | None = Field(
        None,
        description=(
            "Primary tax identifier for uniqueness (part of graph ID). "
            "Extract the main VAT or tax ID number. "
            "Used to distinguish parties with same name."
        ),
        examples=["FR12345678901", "GB123456789", "DE123456789"],
    )

    # Edges to components
    postal_address: PostalAddress | None = edge(
        label="LOCATED_AT", default=None, description="Physical postal address of the party"
    )

    contact: ContactPoint | None = edge(
        label="HAS_CONTACT", default=None, description="Contact information (email, phone, website)"
    )

    electronic_address: ElectronicAddress | None = edge(
        label="HAS_ELECTRONIC_ADDRESS",
        default=None,
        description="Electronic invoicing routing endpoint (Peppol, EDI, etc.)",
    )

    # Non-edge fields for identifiers (embedded data)
    identifiers: List[PartyIdentifier] = Field(
        default_factory=list,
        description=(
            "List of party identifiers (GLN, DUNS, local registration numbers). "
            "Look for various ID numbers and codes. "
            "Extract scheme and value for each identifier."
        ),
    )

    tax_registrations: List[TaxRegistration] = Field(
        default_factory=list,
        description=(
            "List of tax registrations (VAT, tax IDs, enterprise numbers). "
            "Look for VAT numbers, tax IDs, fiscal identifiers. "
            "Extract scheme, value, and country for each registration."
        ),
    )

    def __str__(self) -> str:
        """Format party name for display."""
        return self.name


class Item(BaseModel):
    """
    Item entity representing a product or service.
    Uniquely identified by seller item ID or name.
    """

    model_config = ConfigDict(graph_id_fields=["seller_item_id", "name"])

    seller_item_id: str | None = Field(
        None,
        description=(
            "Seller's item identifier or SKU. "
            "Look for item codes, SKUs, product numbers, article numbers. "
            "Common labels: 'Item Code', 'SKU', 'Product Code', 'Article No', 'Référence'."
        ),
        examples=["SKU-12345", "PROD-ABC-001", "ART-789", "REF-XYZ"],
    )

    buyer_item_id: str | None = Field(
        None,
        description=(
            "Buyer's item identifier. "
            "Look for buyer's product codes or reference numbers. "
            "Common labels: 'Buyer Code', 'Customer Reference', 'Your Reference'."
        ),
        examples=["BUYER-12345", "CUST-REF-001", "YOUR-REF-789"],
    )

    name: str | None = Field(
        None,
        description=(
            "Item name or description. "
            "Look for product names, service descriptions in line items. "
            "Common labels: 'Item', 'Product', 'Description', 'Service', 'Article', 'Produit'."
        ),
        examples=[
            "Professional Consulting Services",
            "Laptop Computer - Model XYZ",
            "Software License - Annual",
            "Office Supplies Bundle",
        ],
    )

    brand: str | None = Field(
        None,
        description=(
            "Brand or manufacturer name. "
            "Look for brand names, manufacturer information. "
            "Common labels: 'Brand', 'Manufacturer', 'Make', 'Marque'."
        ),
        examples=["Dell", "Microsoft", "HP", "Apple"],
    )

    origin_country: str | None = Field(
        None,
        description=(
            "Country of origin (ISO 3166-1 alpha-2 code or name). "
            "Look for 'Made in', 'Origin', 'Country of Origin'. "
            "Accept both codes and full names."
        ),
        examples=["FR", "France", "DE", "Germany", "CN", "China"],
    )

    # Non-edge fields for identifiers and classifications
    identifiers: List[dict] = Field(
        default_factory=list,
        description=(
            "List of item identifiers with type and value. "
            "Look for GTINs, UUIDs, various product codes. "
            "Each identifier has 'id_type' and 'value'. "
            "Types: 'GTIN', 'UUID', 'SELLER_ITEM_ID', 'BUYER_ITEM_ID', 'OTHER'."
        ),
        examples=[
            [{"id_type": "GTIN", "value": "1234567890123"}],
            [{"id_type": "UUID", "value": "550e8400-e29b-41d4-a716-446655440000"}],
        ],
    )

    classifications: List[dict] = Field(
        default_factory=list,
        description=(
            "List of item classifications with scheme and code. "
            "Look for commodity codes, classification codes. "
            "Each classification has 'scheme' and 'code'. "
            "Common schemes: 'UNSPSC', 'CPV', 'HS', 'TARIC'."
        ),
        examples=[
            [{"scheme": "UNSPSC", "code": "43211500"}],
            [{"scheme": "CPV", "code": "30213000"}],
        ],
    )

    def __str__(self) -> str:
        """Format item for display."""
        return self.name or self.seller_item_id or "Unknown Item"


# =============================================================================
# PRICING AND TAX COMPONENTS
# =============================================================================


class Price(BaseModel):
    """
    Price information component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    unit_price: MonetaryAmount | None = edge(
        label="HAS_UNIT_PRICE",
        default=None,
        description="Net unit price (after price-level discounts)",
    )

    gross_price: MonetaryAmount | None = edge(
        label="HAS_GROSS_PRICE",
        default=None,
        description="Gross unit price (before price-level discounts)",
    )

    price_discount: MonetaryAmount | None = edge(
        label="HAS_PRICE_DISCOUNT", default=None, description="Price-level discount amount"
    )

    base_quantity: Quantity | None = edge(
        label="HAS_BASE_QUANTITY",
        default=None,
        description="Base quantity for unit price (e.g., price per 100 units)",
    )


class TaxDetail(BaseModel):
    """
    Tax detail component for line-level or allowance/charge tax.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    tax_type: TaxType = Field(
        default=TaxType.VAT,
        description=(
            "Type of tax. "
            "Look for tax type labels. "
            "Common types: 'VAT', 'GST', 'Sales Tax', 'Withholding Tax'. "
            "Default to VAT if not specified."
        ),
        examples=["VAT", "GST", "Sales Tax"],
    )

    category_code: str | None = Field(
        None,
        description=(
            "Tax category code. "
            "Look for VAT category codes. "
            "Common codes: 'S' (standard), 'Z' (zero-rated), 'E' (exempt), "
            "'AE' (reverse charge), 'K' (intra-community), 'G' (free export). "
            "EN 16931 codes: S, AA, Z, E, AE, K, G, O, L, M."
        ),
        examples=["S", "Z", "E", "AE", "K", "G"],
    )

    rate_percent: float | None = Field(
        None,
        description=(
            "Tax rate as percentage. "
            "Look for tax rate percentages. "
            "Common labels: 'VAT Rate', 'Tax Rate', 'Rate', 'Taux TVA'. "
            "Extract numeric value (e.g., '20%' → 20.0)."
        ),
        examples=[20.0, 10.0, 5.5, 0.0, 19.0],
    )

    exemption_reason: str | None = Field(
        None,
        description=(
            "Tax exemption reason text. "
            "Look for exemption explanations. "
            "Common labels: 'Exemption Reason', 'Tax Exempt', 'Exonération'."
        ),
        examples=[
            "Exempt - Article 151",
            "Reverse charge",
            "Intra-community supply",
            "Export outside EU",
        ],
    )

    exemption_reason_code: str | None = Field(
        None,
        description=(
            "Tax exemption reason code. Look for exemption codes. EN 16931 codes for exemptions."
        ),
        examples=["VATEX-EU-79-C", "VATEX-EU-132", "VATEX-EU-143"],
    )

    taxable_base: MonetaryAmount | None = edge(
        label="HAS_TAXABLE_BASE",
        default=None,
        description="Taxable base amount (amount on which tax is calculated)",
    )

    tax_amount: MonetaryAmount | None = edge(
        label="HAS_TAX_AMOUNT", default=None, description="Calculated tax amount"
    )

    @field_validator("tax_type", mode="before")
    @classmethod
    def normalize_tax_type(cls, v: Any) -> Any:
        """Normalize tax type enum."""
        return _normalize_enum(TaxType, v)


class AllowanceCharge(BaseModel):
    """
    Allowance (discount) or charge (surcharge) component.
    Can apply at document, line, or price level.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    charge_indicator: bool = Field(
        ...,
        description=(
            "True for charge (surcharge), False for allowance (discount). "
            "Look for discount/allowance vs charge/surcharge indicators. "
            "Discounts/allowances reduce amount (False), charges increase amount (True)."
        ),
        examples=[True, False],
    )

    scope: AllowanceChargeScope = Field(
        default=AllowanceChargeScope.DOCUMENT,
        description=(
            "Scope where this applies: Document, Line, or Price level. "
            "Document-level: applies to entire document. "
            "Line-level: applies to specific line item. "
            "Price-level: applies to unit price calculation."
        ),
        examples=["Document", "Line", "Price"],
    )

    reason: str | None = Field(
        None,
        description=(
            "Reason for allowance or charge. "
            "Look for discount reasons, charge descriptions. "
            "Common labels: 'Reason', 'Description', 'Motif', 'Razón'."
        ),
        examples=[
            "Early payment discount",
            "Volume discount",
            "Shipping charge",
            "Handling fee",
            "Special promotion",
        ],
    )

    reason_code: str | None = Field(
        None,
        description=(
            "Coded reason for allowance or charge. "
            "Look for reason codes. "
            "Common codes: '95' (discount), '64' (special agreement), 'FC' (freight charge)."
        ),
        examples=["95", "64", "FC", "AA", "ABL"],
    )

    percent: float | None = Field(
        None,
        description=(
            "Percentage rate for allowance or charge. "
            "Look for percentage values. "
            "Extract numeric value (e.g., '10%' → 10.0)."
        ),
        examples=[10.0, 5.0, 2.5, 15.0],
    )

    amount: MonetaryAmount | None = edge(
        label="HAS_AMOUNT", default=None, description="Allowance or charge amount"
    )

    base_amount: MonetaryAmount | None = edge(
        label="HAS_BASE_AMOUNT",
        default=None,
        description="Base amount on which percentage is calculated",
    )

    tax: TaxDetail | None = edge(
        label="HAS_TAX", default=None, description="Tax applied to this allowance or charge"
    )

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_scope(cls, v: Any) -> Any:
        """Normalize scope enum."""
        return _normalize_enum(AllowanceChargeScope, v)


# =============================================================================
# DOCUMENT LINE ENTITY
# =============================================================================


class DocumentLine(BaseModel):
    """
    Document line entity representing a billed line item.
    Can be goods, services, fees, or returnable items.
    Uniquely identified by line number and item reference.
    """

    model_config = ConfigDict(graph_id_fields=["line_no", "item_seller_id"])

    line_no: str = Field(
        ...,
        description=(
            "Line number or identifier. "
            "Look for line numbers, position numbers in line item tables. "
            "Common labels: 'Line', 'No', 'Pos', 'Item No', 'Ligne', 'Línea'. "
            "Can be numeric or alphanumeric."
        ),
        examples=["1", "2", "10", "A1", "LINE-001"],
    )

    item_seller_id: str | None = Field(
        None,
        description=(
            "Seller item ID for graph uniqueness (part of graph ID). "
            "Extract from item reference or SKU. "
            "Used to distinguish lines with same line number."
        ),
        examples=["SKU-12345", "PROD-001", "ART-789"],
    )

    description: str | None = Field(
        None,
        description=(
            "Line item description or note. "
            "Look for item descriptions, notes, additional details. "
            "Common labels: 'Description', 'Details', 'Note', 'Remarque'."
        ),
        examples=[
            "Professional consulting services for Q1 2024",
            "Laptop computer with extended warranty",
            "Monthly software subscription",
        ],
    )

    accounting_account: str | None = Field(
        None,
        description=(
            "Accounting account code or reference. "
            "Look for account codes, GL codes, cost centers. "
            "Common labels: 'Account', 'GL Code', 'Cost Center', 'Compte'."
        ),
        examples=["4000", "6100", "ACC-001", "CC-SALES"],
    )

    # Edges to entities and components
    item: Item | None = edge(
        label="REFERENCES_ITEM",
        default=None,
        description="The item (product or service) being billed",
    )

    quantity: Quantity | None = edge(
        label="HAS_QUANTITY", default=None, description="Quantity ordered or delivered"
    )

    price: Price | None = edge(
        label="HAS_PRICE",
        default=None,
        description="Price information (unit price, gross price, discounts)",
    )

    line_total: MonetaryAmount | None = edge(
        label="HAS_LINE_TOTAL",
        default=None,
        description="Total line amount (quantity x unit price +/- line allowances/charges)",
    )

    tax: TaxDetail | None = edge(
        label="HAS_TAX", default=None, description="Tax applied to this line"
    )

    period: Period | None = edge(
        label="HAS_PERIOD", default=None, description="Service or delivery period for this line"
    )

    # List edges
    allowances_charges: List[AllowanceCharge] = edge(
        label="HAS_ALLOWANCE_CHARGE",
        default_factory=list,
        description="Line-level allowances (discounts) and charges (surcharges)",
    )

    def __str__(self) -> str:
        """Format line for display."""
        return f"Line {self.line_no}"


# =============================================================================
# TAX SUMMARY COMPONENTS
# =============================================================================


class TaxTotal(BaseModel):
    """
    Tax total component for one tax category/rate combination.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    tax_type: TaxType = Field(
        default=TaxType.VAT,
        description=("Type of tax. Common types: 'VAT', 'GST', 'Sales Tax'."),
        examples=["VAT", "GST", "Sales Tax"],
    )

    category_code: str | None = Field(
        None,
        description=("Tax category code. EN 16931 codes: S, AA, Z, E, AE, K, G, O, L, M."),
        examples=["S", "Z", "E", "AE"],
    )

    rate_percent: float | None = Field(
        None,
        description=("Tax rate as percentage. Extract numeric value."),
        examples=[20.0, 10.0, 5.5, 0.0],
    )

    taxable_base: MonetaryAmount = edge(
        label="HAS_TAXABLE_BASE", description="Total taxable base for this tax category/rate"
    )

    tax_amount: MonetaryAmount = edge(
        label="HAS_TAX_AMOUNT", description="Total tax amount for this tax category/rate"
    )

    @field_validator("tax_type", mode="before")
    @classmethod
    def normalize_tax_type(cls, v: Any) -> Any:
        """Normalize tax type enum."""
        return _normalize_enum(TaxType, v)


class TaxSummary(BaseModel):
    """
    Tax summary component containing totals by category/rate.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    total_tax_amount: MonetaryAmount | None = edge(
        label="HAS_TOTAL_TAX_AMOUNT",
        default=None,
        description="Total tax amount across all categories",
    )

    tax_totals: List[TaxTotal] = edge(
        label="HAS_TAX_TOTAL",
        default_factory=list,
        description="Tax totals broken down by category and rate",
    )


class DocumentTotals(BaseModel):
    """
    Document totals component containing all computed financial totals.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    net_total: MonetaryAmount | None = edge(
        label="HAS_NET_TOTAL",
        default=None,
        description="Sum of line net amounts (before document-level allowances/charges)",
    )

    allowance_total: MonetaryAmount | None = edge(
        label="HAS_ALLOWANCE_TOTAL",
        default=None,
        description="Sum of document-level allowances (discounts)",
    )

    charge_total: MonetaryAmount | None = edge(
        label="HAS_CHARGE_TOTAL",
        default=None,
        description="Sum of document-level charges (surcharges)",
    )

    tax_exclusive_total: MonetaryAmount | None = edge(
        label="HAS_TAX_EXCLUSIVE_TOTAL",
        default=None,
        description="Total amount excluding tax (net ± allowances/charges)",
    )

    tax_total: MonetaryAmount | None = edge(
        label="HAS_TAX_TOTAL", default=None, description="Total tax amount"
    )

    tax_inclusive_total: MonetaryAmount | None = edge(
        label="HAS_TAX_INCLUSIVE_TOTAL", default=None, description="Total amount including tax"
    )

    prepaid_total: MonetaryAmount | None = edge(
        label="HAS_PREPAID_TOTAL",
        default=None,
        description="Total prepaid or advance payment amount",
    )

    rounding: MonetaryAmount | None = edge(
        label="HAS_ROUNDING", default=None, description="Rounding amount applied to final total"
    )

    amount_due: MonetaryAmount | None = edge(
        label="HAS_AMOUNT_DUE",
        default=None,
        description="Final amount due for payment (tax inclusive - prepaid + rounding)",
    )


# =============================================================================
# PAYMENT AND SETTLEMENT MODELS
# =============================================================================


class BankAccount(BaseModel):
    """
    Bank account component for payment instructions.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    iban: str | None = Field(
        None,
        description=(
            "International Bank Account Number. "
            "Look for IBAN codes (typically starting with country code). "
            "Common labels: 'IBAN', 'Account Number', 'Compte Bancaire'. "
            "Keep original format with spaces."
        ),
        examples=["FR76 1234 5678 9012 3456 7890 123", "GB29 NWBK 6016 1331 9268 19"],
    )

    bic: str | None = Field(
        None,
        description=(
            "Bank Identifier Code (SWIFT code). "
            "Look for BIC/SWIFT codes (8 or 11 characters). "
            "Common labels: 'BIC', 'SWIFT', 'Bank Code'."
        ),
        examples=["BNPAFRPP", "NWBKGB2L", "DEUTDEFF"],
    )

    account_no: str | None = Field(
        None,
        description=(
            "Local bank account number (if not IBAN). "
            "Look for account numbers. "
            "Common labels: 'Account No', 'Account Number', 'Numéro de Compte'."
        ),
        examples=["12345678", "1234567890", "ACC-123456"],
    )

    bank_name: str | None = Field(
        None,
        description=(
            "Name of the bank. Look for bank names. Common labels: 'Bank', 'Bank Name', 'Banque'."
        ),
        examples=["BNP Paribas", "HSBC", "Deutsche Bank"],
    )

    account_holder: str | None = Field(
        None,
        description=(
            "Name of account holder. "
            "Look for account holder names. "
            "Common labels: 'Account Holder', 'Beneficiary', 'Titulaire'."
        ),
        examples=["Acme Corporation Ltd", "Tech Solutions Inc"],
    )


class RemittanceInfo(BaseModel):
    """
    Remittance information component for payment reference.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    structured_reference: str | None = Field(
        None,
        description=(
            "Structured payment reference (e.g., ISO 11649 creditor reference). "
            "Look for structured references, creditor references. "
            "Common labels: 'Reference', 'Payment Reference', 'Structured Reference'."
        ),
        examples=["RF18 5390 0754 7034", "+++123/4567/89012+++"],
    )

    unstructured_message: str | None = Field(
        None,
        description=(
            "Unstructured payment message or note. "
            "Look for payment messages, remittance notes. "
            "Common labels: 'Payment Note', 'Message', 'Communication'."
        ),
        examples=["Payment for Invoice INV-2024-001", "Monthly subscription fee"],
    )


class CardPayment(BaseModel):
    """
    Card payment information component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    masked_pan: str | None = Field(
        None,
        description=(
            "Masked card number (PAN). "
            "Look for masked card numbers (e.g., **** **** **** 1234). "
            "Common labels: 'Card Number', 'PAN'."
        ),
        examples=["**** **** **** 1234", "XXXX-XXXX-XXXX-5678"],
    )

    authorization_id: str | None = Field(
        None,
        description=(
            "Card payment authorization ID. "
            "Look for authorization codes, transaction IDs. "
            "Common labels: 'Authorization', 'Auth Code', 'Transaction ID'."
        ),
        examples=["AUTH-123456", "TXN-789012"],
    )


class DirectDebit(BaseModel):
    """
    Direct debit mandate information component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    mandate_id: str | None = Field(
        None,
        description=(
            "Direct debit mandate identifier. "
            "Look for mandate IDs, SEPA mandate references. "
            "Common labels: 'Mandate ID', 'Mandate Reference', 'SEPA Mandate'."
        ),
        examples=["MANDATE-12345", "SEPA-REF-67890"],
    )

    creditor_id: str | None = Field(
        None,
        description=(
            "Creditor identifier for direct debit. "
            "Look for creditor IDs, SEPA creditor identifiers. "
            "Common labels: 'Creditor ID', 'SEPA Creditor ID'."
        ),
        examples=["FR12ZZZ123456", "DE98ZZZ09999999999"],
    )


class QrPaymentData(BaseModel):
    """
    QR code payment data component (e.g., QR-bill, EPC QR).
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    payload: str | None = Field(
        None,
        description=(
            "QR code payload data. "
            "Look for QR code data, encoded payment information. "
            "May be base64 encoded or structured format."
        ),
        examples=["SPC\n0200\n1\nCH...", "BCD\n002\n1\nSCT..."],
    )

    reference_type: str | None = Field(
        None,
        description=(
            "QR payment reference type. "
            "Look for reference type indicators. "
            "Common types: 'QRR' (QR reference), 'SCOR' (creditor reference), 'NON' (no reference)."
        ),
        examples=["QRR", "SCOR", "NON"],
    )

    reference: str | None = Field(
        None,
        description=(
            "QR payment reference number. "
            "Look for QR reference numbers. "
            "Format depends on reference type."
        ),
        examples=["210000000003139471430009017", "RF18 5390 0754 7034"],
    )


class PaymentTerms(BaseModel):
    """
    Payment terms component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    terms_text: str | None = Field(
        None,
        description=(
            "Payment terms description. "
            "Look for payment terms text. "
            "Common labels: 'Payment Terms', 'Terms', 'Conditions de Paiement'. "
            "Examples: 'Net 30', 'Due on receipt', '2/10 Net 30'."
        ),
        examples=["Net 30", "Due on receipt", "2/10 Net 30", "Payment within 14 days"],
    )

    due_date: date | None = Field(
        None,
        description=(
            "Payment due date in YYYY-MM-DD format. "
            "Look for due dates, payment deadlines. "
            "Common labels: 'Due Date', 'Payment Due', 'Date d'Échéance'. "
            "Parse various date formats and normalize to YYYY-MM-DD."
        ),
        examples=["2024-02-15", "2024-03-01", "2024-01-31"],
    )

    late_payment_interest: float | None = Field(
        None,
        description=(
            "Late payment interest rate as percentage. "
            "Look for late payment penalties, interest rates. "
            "Common labels: 'Late Payment Interest', 'Penalty Rate'. "
            "Extract numeric value."
        ),
        examples=[1.5, 2.0, 5.0],
    )

    penalty_text: str | None = Field(
        None,
        description=(
            "Late payment penalty description. "
            "Look for penalty clauses, late payment terms. "
            "Common labels: 'Late Payment Penalty', 'Penalty Terms'."
        ),
        examples=[
            "1.5% interest per month on overdue amounts",
            "€50 fixed penalty for late payment",
        ],
    )


class PaymentInstruction(BaseModel):
    """
    Payment instruction component for one payment method.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    instruction_id: str | None = Field(
        None,
        description=(
            "Payment instruction identifier. "
            "Look for instruction IDs, payment references. "
            "Common labels: 'Instruction ID', 'Payment ID'."
        ),
        examples=["PAY-001", "INST-12345"],
    )

    method: PaymentMeans = Field(
        default=PaymentMeans.BANK_TRANSFER,
        description=(
            "Payment method type. "
            "Look for payment method indicators. "
            "Common methods: 'Bank Transfer', 'Card', 'Cash', 'Direct Debit', 'Cheque', 'QR Code'."
        ),
        examples=["Bank Transfer", "Card", "Direct Debit", "QR Code"],
    )

    payee_account: BankAccount | None = edge(
        label="HAS_PAYEE_ACCOUNT",
        default=None,
        description="Bank account for payment (if bank transfer)",
    )

    payment_amount: MonetaryAmount | None = edge(
        label="HAS_PAYMENT_AMOUNT",
        default=None,
        description="Specific amount for this payment instruction",
    )

    qr: QrPaymentData | None = edge(
        label="HAS_QR_DATA", default=None, description="QR code payment data (if QR payment)"
    )

    card: CardPayment | None = edge(
        label="HAS_CARD_DATA",
        default=None,
        description="Card payment information (if card payment)",
    )

    direct_debit: DirectDebit | None = edge(
        label="HAS_DIRECT_DEBIT_DATA",
        default=None,
        description="Direct debit mandate information (if direct debit)",
    )

    @field_validator("method", mode="before")
    @classmethod
    def normalize_method(cls, v: Any) -> Any:
        """Normalize payment method enum."""
        return _normalize_enum(PaymentMeans, v)


class Settlement(BaseModel):
    """
    Settlement entity containing payment information.
    Uniquely identified by payment means and due date.
    """

    model_config = ConfigDict(graph_id_fields=["payment_means", "due_date_str"])

    payment_means: str = Field(
        ...,
        description=(
            "Primary payment method. "
            "Look for payment method labels. "
            "Common labels: 'Payment Method', 'Payment Means', 'Mode de Paiement'."
        ),
        examples=["Bank Transfer", "Card", "Cash", "Direct Debit"],
    )

    due_date_str: str | None = Field(
        None,
        description=(
            "Due date as string for graph ID uniqueness. "
            "Extract from payment terms or due date fields."
        ),
        examples=["2024-02-15", "2024-03-01"],
    )

    requested_execution_date: date | None = Field(
        None,
        description=(
            "Requested payment execution date in YYYY-MM-DD format. "
            "Look for requested payment dates. "
            "Common labels: 'Execution Date', 'Payment Date'. "
            "Parse various date formats and normalize to YYYY-MM-DD."
        ),
        examples=["2024-02-01", "2024-03-15"],
    )

    payment_terms: PaymentTerms | None = edge(
        label="HAS_PAYMENT_TERMS",
        default=None,
        description="Payment terms (due date, late payment penalties)",
    )

    remittance: RemittanceInfo | None = edge(
        label="HAS_REMITTANCE_INFO",
        default=None,
        description="Remittance information (payment reference, message)",
    )

    payment_instructions: List[PaymentInstruction] = edge(
        label="HAS_PAYMENT_INSTRUCTION",
        default_factory=list,
        description="Payment instructions for different methods",
    )


# =============================================================================
# DELIVERY AND ATTACHMENT MODELS
# =============================================================================


class Delivery(BaseModel):
    """
    Delivery information component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    delivery_date: date | None = Field(
        None,
        description=(
            "Actual or expected delivery date in YYYY-MM-DD format. "
            "Look for delivery dates, shipment dates. "
            "Common labels: 'Delivery Date', 'Ship Date', 'Date de Livraison', "
            "'Fecha de Entrega'. "
            "Parse various date formats and normalize to YYYY-MM-DD."
        ),
        examples=["2024-01-20", "2024-02-15", "2024-03-01"],
    )

    delivery_location_name: str | None = Field(
        None,
        description=(
            "Name of delivery location or site. "
            "Look for delivery location names, site names. "
            "Common labels: 'Delivery Location', 'Ship To', 'Site', 'Lieu de Livraison'."
        ),
        examples=["Main Warehouse", "Customer Site A", "Distribution Center"],
    )

    delivery_address: PostalAddress | None = edge(
        label="HAS_DELIVERY_ADDRESS",
        default=None,
        description="Delivery address (may differ from buyer address)",
    )

    delivery_party: Party | None = edge(
        label="DELIVERED_TO",
        default=None,
        description="Party receiving the delivery (may differ from buyer)",
    )

    period: Period | None = edge(
        label="HAS_DELIVERY_PERIOD",
        default=None,
        description="Delivery period (start and end dates)",
    )


class Attachment(BaseModel):
    """
    Attached document component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    filename: str | None = Field(
        None,
        description=(
            "Filename of attached document. "
            "Look for attachment filenames, document names. "
            "Common labels: 'Attachment', 'File', 'Document', 'Fichier Joint'."
        ),
        examples=["contract.pdf", "specifications.docx", "drawing.png", "terms.pdf"],
    )

    description: str | None = Field(
        None,
        description=(
            "Description of attached document. "
            "Look for attachment descriptions. "
            "Common labels: 'Description', 'Document Type', 'Details'."
        ),
        examples=[
            "Contract agreement",
            "Technical specifications",
            "Product drawing",
            "Terms and conditions",
        ],
    )

    mime_type: str | None = Field(
        None,
        description=(
            "MIME type of attachment. "
            "Look for file type indicators. "
            "Common types: 'application/pdf', 'image/png', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'."
        ),
        examples=["application/pdf", "image/png", "image/jpeg", "application/xml"],
    )

    embedded_data: str | None = Field(
        None,
        description=(
            "Base64 encoded attachment data (if embedded). "
            "Look for embedded document data. "
            "Typically base64 encoded binary data."
        ),
        examples=["JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlL..."],
    )

    external_reference: str | None = Field(
        None,
        description=(
            "External reference or URL to attachment. "
            "Look for URLs, file paths, external references. "
            "Common labels: 'URL', 'Link', 'External Reference'."
        ),
        examples=["https://example.com/documents/contract.pdf", "file://server/share/document.pdf"],
    )


class DocumentReference(BaseModel):
    """
    Document reference component for linking to other business documents.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    ref_type: DocumentReferenceType = Field(
        ...,
        description=(
            "Type of referenced document. "
            "Look for document type labels. "
            "Common types: 'Purchase Order', 'Contract', 'Previous Invoice', "
            "'Despatch Advice', 'Project', 'Tender'."
        ),
        examples=["Purchase Order", "Contract", "Previous Invoice", "Project"],
    )

    ref_no: str = Field(
        ...,
        description=(
            "Reference document number or identifier. "
            "Look for PO numbers, contract numbers, invoice references. "
            "Common labels: 'PO Number', 'Contract No', 'Reference', 'Order No', "
            "'Commande', 'Pedido'."
        ),
        examples=["PO-2024-001", "CONTRACT-12345", "INV-2023-999", "PROJ-ABC"],
    )

    ref_date: date | None = Field(
        None,
        description=(
            "Date of referenced document in YYYY-MM-DD format. "
            "Look for document dates. "
            "Parse various date formats and normalize to YYYY-MM-DD."
        ),
        examples=["2024-01-15", "2023-12-20", "2024-02-01"],
    )

    @field_validator("ref_type", mode="before")
    @classmethod
    def normalize_ref_type(cls, v: Any) -> Any:
        """Normalize reference type enum."""
        return _normalize_enum(DocumentReferenceType, v)


# =============================================================================
# ROOT BILLING DOCUMENT
# =============================================================================


class BillingDocument(BaseModel):
    """
    Root billing document entity (Invoice, Credit Note, Debit Note, Pro Forma, Receipt).

    Uniquely identified by document number.
    Supports EN 16931, Peppol BIS, and UBL standards.
    """

    model_config = ConfigDict(graph_id_fields=["document_no"])

    # --- Core Document Fields ---

    document_id: str | None = Field(
        None,
        description=(
            "Stable document identifier (UUID or internal ID). "
            "Look for document IDs, UUIDs, internal references. "
            "Common labels: 'Document ID', 'ID', 'UUID'."
        ),
        examples=["550e8400-e29b-41d4-a716-446655440000", "DOC-2024-001"],
    )

    document_no: str = Field(
        ...,
        description=(
            "Human-readable document number (invoice number, receipt number, etc.). "
            "Look for document numbers in headers. "
            "Common labels: 'Invoice No', 'Invoice Number', 'Receipt No', 'Document Number', "
            "'Facture No', 'Número de Factura'. "
            "This is the primary identifier visible to users."
        ),
        examples=["INV-2024-001", "2024-INV-12345", "REC-001", "CN-2024-050"],
    )

    document_type: DocumentType = Field(
        default=DocumentType.INVOICE,
        description=(
            "Type of billing document. "
            "Look for document type indicators in headers or titles. "
            "Common types: 'Invoice', 'Credit Note', 'Debit Note', 'Pro Forma', 'Receipt'. "
            "Common labels: 'Document Type', 'Type', 'Type de Document'."
        ),
        examples=["Invoice", "Credit Note", "Debit Note", "Pro Forma", "Receipt"],
    )

    issue_date: date | None = Field(
        None,
        description=(
            "Date the document was issued in YYYY-MM-DD format. "
            "Look for issue dates, document dates in headers. "
            "Common labels: 'Date', 'Issue Date', 'Invoice Date', 'Date d'Émission', "
            "'Fecha de Emisión'. "
            "Parse various date formats (DD/MM/YYYY, MM-DD-YYYY, DD.MM.YYYY) and normalize to YYYY-MM-DD."
        ),
        examples=["2024-01-15", "2024-02-20", "2024-03-01"],
    )

    due_date: date | None = Field(
        None,
        description=(
            "Payment due date in YYYY-MM-DD format. "
            "Look for due dates, payment deadlines. "
            "Common labels: 'Due Date', 'Payment Due', 'Date d'Échéance', 'Fecha de Vencimiento'. "
            "Parse various date formats and normalize to YYYY-MM-DD."
        ),
        examples=["2024-02-15", "2024-03-20", "2024-04-01"],
    )

    currency: str | None = Field(
        None,
        description=(
            "Document currency (ISO 4217 code). "
            "Look for currency codes or symbols in headers or totals. "
            "Convert symbols to codes: € → EUR, $ → USD, £ → GBP. "
            "Common labels: 'Currency', 'Devise', 'Moneda'. "
            "This is the main document currency for all amounts."
        ),
        examples=["EUR", "USD", "GBP", "CHF"],
    )

    tax_currency: str | None = Field(
        None,
        description=(
            "Tax currency if different from document currency (ISO 4217 code). "
            "Look for separate tax currency indicators. "
            "Common labels: 'Tax Currency', 'VAT Currency'. "
            "Only present if tax is calculated in a different currency."
        ),
        examples=["EUR", "USD", "GBP"],
    )

    language: str | None = Field(
        None,
        description=(
            "Document language (BCP 47 code). "
            "Look for language indicators. "
            "Common codes: 'en' (English), 'fr' (French), 'de' (German), 'es' (Spanish). "
            "Can be 2-letter (en) or extended (en-US, fr-FR)."
        ),
        examples=["en", "fr", "de", "es", "en-US", "fr-FR"],
    )

    profile: str | None = Field(
        None,
        description=(
            "Business process profile identifier. "
            "Look for profile IDs, process identifiers. "
            "Common profiles: 'urn:fdc:peppol.eu:2017:poacc:billing:01:1.0' (Peppol BIS). "
            "Indicates which standard/profile the document follows."
        ),
        examples=[
            "urn:fdc:peppol.eu:2017:poacc:billing:01:1.0",
            "urn:cen.eu:en16931:2017",
            "peppol-bis",
        ],
    )

    notes: List[str] = Field(
        default_factory=list,
        description=(
            "General notes or remarks about the document. "
            "Look for notes, remarks, comments sections. "
            "Common labels: 'Notes', 'Remarks', 'Comments', 'Remarques', 'Notas'. "
            "Extract each note as a separate string."
        ),
        examples=[
            ["Payment terms: Net 30 days", "Thank you for your business"],
            ["Special handling required", "Urgent delivery"],
            ["This is a pro forma invoice for customs purposes only"],
        ],
    )

    # --- Party Relationships (Edges) ---

    issuer: Party = edge(
        label="ISSUED_BY",
        description=(
            "Party that issued this document (seller, supplier). "
            "Look for seller information, supplier details in headers. "
            "Common labels: 'Seller', 'Supplier', 'From', 'Vendeur', 'Proveedor'."
        ),
    )

    buyer: Party | None = edge(
        label="SENT_TO",
        default=None,
        description=(
            "Party receiving this document (buyer, customer). "
            "Look for buyer information, customer details. "
            "Common labels: 'Buyer', 'Customer', 'Bill To', 'Client', 'Acheteur', 'Comprador'."
        ),
    )

    payee: Party | None = edge(
        label="PAYABLE_TO",
        default=None,
        description=(
            "Party to receive payment (if different from issuer). "
            "Look for payee information, payment recipient. "
            "Common labels: 'Payee', 'Payment To', 'Bénéficiaire'. "
            "Only present if payment goes to a different party than the issuer."
        ),
    )

    tax_representative: Party | None = edge(
        label="HAS_TAX_REPRESENTATIVE",
        default=None,
        description=(
            "Tax representative party (for VAT representation cases). "
            "Look for tax representative information. "
            "Common labels: 'Tax Representative', 'VAT Representative', 'Représentant Fiscal'. "
            "Required in some cross-border VAT scenarios."
        ),
    )

    # --- Document Structure (Edges) ---

    lines: List[DocumentLine] = edge(
        label="CONTAINS_LINE",
        default_factory=list,
        description=(
            "Document line items (goods, services, fees). "
            "Look for line item tables, itemized lists. "
            "Extract each line with quantity, price, description, etc."
        ),
    )

    allowances_charges: List[AllowanceCharge] = edge(
        label="HAS_ALLOWANCE_CHARGE",
        default_factory=list,
        description=(
            "Document-level allowances (discounts) and charges (surcharges). "
            "Look for document-level discounts, shipping charges, handling fees. "
            "Common labels: 'Discount', 'Shipping', 'Handling Fee', 'Remise', 'Frais'."
        ),
    )

    # --- Financial Information (Edges) ---

    tax_summary: TaxSummary | None = edge(
        label="HAS_TAX_SUMMARY",
        default=None,
        description=(
            "Tax summary with totals by category and rate. "
            "Look for VAT breakdown, tax summary sections. "
            "Common labels: 'VAT Summary', 'Tax Breakdown', 'Récapitulatif TVA'."
        ),
    )

    totals: DocumentTotals = edge(
        label="HAS_TOTALS",
        description=(
            "Document totals (net, tax, gross, amount due). "
            "Look for totals section, summary amounts. "
            "Common labels: 'Total', 'Amount Due', 'Grand Total', 'Montant Total'."
        ),
    )

    # --- Payment Information (Edge) ---

    payment: Settlement | None = edge(
        label="HAS_SETTLEMENT",
        default=None,
        description=(
            "Payment and settlement information. "
            "Look for payment instructions, bank details, payment terms. "
            "Common labels: 'Payment Information', 'Bank Details', 'Payment Terms'."
        ),
    )

    # --- Delivery Information (List Edge) ---

    delivery: List[Delivery] = edge(
        label="HAS_DELIVERY",
        default_factory=list,
        description=(
            "Delivery information (dates, locations, parties). "
            "Look for delivery details, shipping information. "
            "Common labels: 'Delivery', 'Shipping', 'Livraison', 'Entrega'."
        ),
    )

    # --- References and Attachments (List Edges) ---

    references: List[DocumentReference] = edge(
        label="REFERENCES_DOCUMENT",
        default_factory=list,
        description=(
            "References to other business documents (PO, contracts, previous invoices). "
            "Look for reference sections, related documents. "
            "Common labels: 'References', 'Related Documents', 'PO Number', 'Contract No'."
        ),
    )

    attachments: List[Attachment] = edge(
        label="HAS_ATTACHMENT",
        default_factory=list,
        description=(
            "Attached documents (contracts, specifications, drawings). "
            "Look for attachments, enclosed documents. "
            "Common labels: 'Attachments', 'Enclosed Documents', 'Pièces Jointes'."
        ),
    )

    # --- Validators ---

    @field_validator("document_type", mode="before")
    @classmethod
    def normalize_document_type(cls, v: Any) -> Any:
        """Normalize document type enum."""
        return _normalize_enum(DocumentType, v)

    @field_validator("currency", "tax_currency")
    @classmethod
    def validate_currency_format(cls, v: Any) -> Any:
        """Ensure currency is 3 uppercase letters (ISO 4217)."""
        if v and not (len(v) == 3 and v.isupper() and v.isalpha()):
            raise ValueError(f"Currency must be 3 uppercase letters (ISO 4217), got {v}")
        return v

    def __str__(self) -> str:
        """Format document for display."""
        return f"{self.document_type.value} {self.document_no}"
