"""
Simplified Invoice/Bill extraction template.

Extracts structured data from invoices, bills, credit notes, and receipts.
Streamlined for clarity and ease of use while maintaining essential business logic.

Version: 2.0.0
Last Updated: 2026-01-26
"""

from datetime import date
from enum import Enum
from typing import Any, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def edge(label: str, **kwargs: Any) -> Any:
    """
    Create a Pydantic Field with edge metadata for graph relationships.

    Args:
        label: Edge label in ALL_CAPS format (e.g., "ISSUED_BY", "CONTAINS_LINE")
        **kwargs: Additional Field parameters

    Returns:
        Pydantic Field with edge metadata
    """
    if "default" not in kwargs and "default_factory" not in kwargs:
        kwargs["default"] = ...
    return Field(json_schema_extra={"edge_label": label}, **kwargs)


# =============================================================================
# ENUMS
# =============================================================================


class DocumentType(str, Enum):
    """Type of billing document."""

    INVOICE = "Invoice"
    CREDIT_NOTE = "Credit Note"
    DEBIT_NOTE = "Debit Note"
    RECEIPT = "Receipt"
    OTHER = "Other"


class TaxType(str, Enum):
    """Type of tax applied."""

    VAT = "VAT"
    GST = "GST"
    SALES_TAX = "Sales Tax"
    OTHER = "Other"


class PaymentMethod(str, Enum):
    """Payment method type."""

    BANK_TRANSFER = "Bank Transfer"
    CARD = "Card"
    CASH = "Cash"
    DIRECT_DEBIT = "Direct Debit"
    CHEQUE = "Cheque"
    OTHER = "Other"


# =============================================================================
# ENTITIES
# =============================================================================


class Party(BaseModel):
    """
    Party entity representing an organization or person (seller, buyer, etc.).
    Identified primarily by name with tax_id as secondary disambiguator.
    """

    model_config = ConfigDict(graph_id_fields=["name"], extra="ignore", populate_by_name=True)

    name: str = Field(
        ...,
        description=(
            "Full legal name of the party. "
            "LOOK FOR: Company name in header, 'Seller', 'Buyer', 'From', 'To', 'Bill To' sections. "
            "EXTRACT: Complete name including legal suffixes (Ltd, Inc, SA, GmbH, SL). "
            "EXAMPLES: 'Acme Corporation Ltd', 'Tech Solutions Inc', 'Global Industries SA'"
        ),
        examples=["Acme Corporation Ltd", "Tech Solutions Inc", "Global Industries SA"],
    )

    tax_id: str | None = Field(
        None,
        description=(
            "Primary tax identifier (VAT number, tax ID, EIN). "
            "LOOK FOR: 'VAT', 'Tax ID', 'TVA', 'NIF', 'RFC', 'EIN' labels near party name. "
            "EXTRACT: Full number including country prefix if present. "
            "EXAMPLES: 'FR12345678901', 'GB123456789', 'DE123456789'"
        ),
        examples=["FR12345678901", "GB123456789", "DE123456789"],
    )

    # Contact Information (embedded as simple fields)
    email: str | None = Field(
        None,
        description=(
            "Email address. "
            "LOOK FOR: Text containing '@' symbol, 'Email', 'E-mail', 'Contact' labels. "
            "EXAMPLES: 'contact@company.com', 'info@business.fr'"
        ),
        examples=["contact@company.com", "info@business.fr"],
    )

    phone: str | None = Field(
        None,
        description=(
            "Phone number with country code if present. "
            "LOOK FOR: 'Phone', 'Tel', 'Telephone', 'Mobile' labels, numbers with +/parentheses. "
            "EXAMPLES: '+33 1 23 45 67 89', '+44 20 7123 4567', '(555) 123-4567'"
        ),
        examples=["+33 1 23 45 67 89", "+44 20 7123 4567"],
    )

    website: str | None = Field(
        None,
        description=(
            "Website URL. "
            "LOOK FOR: 'Website', 'Web', 'URL' labels, text starting with http://, https://, www. "
            "EXAMPLES: 'https://www.company.com', 'www.business.fr'"
        ),
        examples=["https://www.company.com", "www.business.fr"],
    )

    # Address (embedded as simple fields)
    street: str | None = Field(
        None,
        description=(
            "Street address with number. "
            "LOOK FOR: 'Address', 'Street', 'Adresse', 'Rue' labels in party section. "
            "PRESERVE: Exact spacing between street name and number. "
            "EXAMPLES: 'Rue du Lac 1268', '123 Main Street', 'Marktgasse 28'"
        ),
        examples=["Rue du Lac 1268", "123 Main Street", "Marktgasse 28"],
    )

    city: str | None = Field(
        None,
        description=(
            "City or town name. "
            "LOOK FOR: 'City', 'Town', 'Ville' labels, typically after postal code. "
            "EXAMPLES: 'Paris', 'London', 'New York'"
        ),
        examples=["Paris", "London", "New York"],
    )

    postal_code: str | None = Field(
        None,
        description=(
            "Postal or ZIP code. "
            "LOOK FOR: Numeric code near city, 'Postal Code', 'ZIP', 'Code Postal', 'PLZ' labels. "
            "EXAMPLES: '75001', 'SW1A 1AA', '10001'"
        ),
        examples=["75001", "SW1A 1AA", "10001"],
    )

    country: str | None = Field(
        None,
        description=(
            "Country name or ISO code. "
            "LOOK FOR: 'Country', 'Pays', 'País' labels, 2-letter codes or full names. "
            "EXAMPLES: 'France', 'FR', 'United Kingdom', 'GB'"
        ),
        examples=["France", "FR", "United Kingdom", "GB"],
    )


class Item(BaseModel):
    """
    Item entity representing a product or service.
    Uniquely identified by item_code (required) and name (optional fallback).
    At least one of item_code or name must be provided.
    """

    model_config = ConfigDict(graph_id_fields=["item_code"], extra="ignore", populate_by_name=True)

    item_code: str = Field(
        ...,
        validation_alias=AliasChoices("item_code", "itemCode", "sku", "product_code"),
        description=(
            "Item identifier, SKU, or product code (required for uniqueness). "
            "LOOK FOR: 'Item Code', 'SKU', 'Product Code', 'Article No', 'Référence' in line items. "
            "EXTRACT: If not present, use item name or generate from line number (e.g., 'ITEM-1'). "
            "EXAMPLES: 'SKU-12345', 'PROD-ABC-001', 'ART-789', 'ITEM-1'"
        ),
        examples=["SKU-12345", "PROD-ABC-001", "ART-789", "ITEM-1"],
    )

    name: str | None = Field(
        None,
        description=(
            "Item name or description. "
            "LOOK FOR: 'Item', 'Product', 'Description', 'Service' columns in line item table. "
            "EXTRACT: Full product/service description. "
            "EXAMPLES: 'Professional Consulting Services', 'Laptop Computer - Model XYZ'"
        ),
        examples=["Professional Consulting Services", "Laptop Computer - Model XYZ"],
    )

    category: str | None = Field(
        None,
        description=(
            "Item category or classification. "
            "LOOK FOR: 'Category', 'Type', 'Class' labels. "
            "EXAMPLES: 'Electronics', 'Services', 'Office Supplies'"
        ),
        examples=["Electronics", "Services", "Office Supplies"],
    )


class Tax(BaseModel):
    """
    Tax information component (can be line-level or document-level).
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    tax_type: TaxType | None = Field(
        None,
        description=(
            "WHAT: Type of tax applied. "
            "EXTRACT: One of the allowed values (VAT, GST, Sales Tax, Other). "
            "If not explicitly stated, omit this field (do not guess). "
            "LOOK FOR: 'VAT', 'GST', 'Sales Tax', 'Tax' labels in tax sections. "
            "EXAMPLES: VAT, GST, Sales Tax"
        ),
        examples=["VAT", "GST", "Sales Tax"],
    )

    rate_percent: float | None = Field(
        None,
        description=(
            "Tax rate as percentage. "
            "LOOK FOR: 'VAT Rate', 'Tax Rate', 'Rate', '%' symbol in tax sections. "
            "EXTRACT: Numeric value only (e.g., '20%' → 20.0). "
            "EXAMPLES: 20.0, 10.0, 5.5, 0.0"
        ),
        examples=[20.0, 10.0, 5.5, 0.0],
    )

    taxable_amount: float | None = Field(
        None,
        description=(
            "WHAT: Amount on which tax is calculated (tax base). "
            "EXTRACT: Numeric value only (e.g., 1000.00, 500.50). "
            "NEVER extract names, text, or dates into this field. "
            "LOOK FOR: 'Taxable Amount', 'Base', 'Net Amount' in tax breakdown. "
            "EXAMPLES: 1000.00, 500.50. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[1000.00, 500.50],
    )

    tax_amount: float | None = Field(
        None,
        description=(
            "WHAT: Calculated tax amount. "
            "EXTRACT: Numeric value only (e.g., 200.00, 100.10). "
            "NEVER extract names, text, or dates into this field. "
            "LOOK FOR: 'Tax Amount', 'VAT Amount', 'Tax' column in totals or tax breakdown. "
            "EXAMPLES: 200.00, 100.10. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[200.00, 100.10],
    )

    exemption_reason: str | None = Field(
        None,
        description=(
            "Tax exemption reason if applicable. "
            "LOOK FOR: 'Exempt', 'Exemption', 'Reverse Charge', 'Zero-rated' text. "
            "EXAMPLES: 'Exempt - Article 151', 'Reverse charge', 'Export outside EU'"
        ),
        examples=["Exempt - Article 151", "Reverse charge", "Export outside EU"],
    )


class LineItem(BaseModel):
    """
    Document line item entity representing a billed item.
    Identified by line_number with item_code as secondary discriminator.
    """

    model_config = ConfigDict(graph_id_fields=["line_number"])

    line_number: str = Field(
        ...,
        description=(
            "Line number or position (required for uniqueness). "
            "LOOK FOR: 'Line', 'No', 'Pos', '#' column in line item table (usually first column). "
            "EXTRACT: Line identifier (can be numeric or alphanumeric). "
            "EXAMPLES: '1', '2', '10', 'A1'"
        ),
        examples=["1", "2", "10", "A1"],
    )

    item_code: str | None = Field(
        None,
        description=(
            "Item code reference (optional). "
            "EXTRACT: From item reference if present. "
            "EXAMPLES: 'SKU-12345', 'PROD-001'"
        ),
        examples=["SKU-12345", "PROD-001"],
    )

    description: str | None = Field(
        None,
        description=(
            "Line item description or additional details. "
            "LOOK FOR: 'Description', 'Details', 'Note' column or text below item name. "
            "EXAMPLES: 'Professional consulting services for Q1 2024', 'Extended warranty included'"
        ),
        examples=["Professional consulting services for Q1 2024"],
    )

    # Quantity and Unit (embedded)
    quantity: float | None = Field(
        None,
        description=(
            "WHAT: Quantity ordered or delivered. "
            "EXTRACT: Numeric value only (e.g., 1.0, 10.5, 28.0). "
            "NEVER extract names, text, units, or dates into this field. "
            "LOOK FOR: 'Quantity', 'Qty', 'Quantité', 'Menge' column in line items. "
            "EXAMPLES: 1.0, 10.5, 28.0, 100.0. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[1.0, 10.5, 28.0, 100.0],
    )

    unit: str | None = Field(
        None,
        description=(
            "Unit of measure. "
            "LOOK FOR: Unit abbreviation near quantity ('EA', 'KG', 'HUR', 'Std.'). "
            "EXTRACT: Unit code separately from quantity. "
            "EXAMPLES: 'EA' (each), 'KG', 'HUR' (hours), 'PC' (pieces)"
        ),
        examples=["EA", "KG", "HUR", "PC"],
    )

    # Price (embedded)
    unit_price: float | None = Field(
        None,
        description=(
            "WHAT: Price per unit (excluding tax). "
            "EXTRACT: Numeric value only (e.g., 100.00, 25.50). "
            "NEVER extract names, text, or dates into this field. "
            "LOOK FOR: 'Unit Price', 'Price', 'Rate', 'Prix Unitaire' column in line items. "
            "EXAMPLES: 100.00, 25.50, 1500.00. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[100.00, 25.50, 1500.00],
    )

    discount_percent: float | None = Field(
        None,
        description=(
            "Discount percentage applied to this line. "
            "LOOK FOR: 'Discount', 'Disc', '%' in line item. "
            "EXTRACT: Numeric value (e.g., '10%' → 10.0). "
            "EXAMPLES: 10.0, 5.0, 15.0"
        ),
        examples=[10.0, 5.0, 15.0],
    )

    line_total: float | None = Field(
        None,
        description=(
            "WHAT: Total amount for this line (quantity x unit_price - discount). "
            "EXTRACT: Numeric value only (e.g., 1000.00, 500.50). "
            "NEVER extract names, text, or dates into this field. "
            "LOOK FOR: 'Total', 'Amount', 'Line Total' column (usually last column in line items). "
            "EXAMPLES: 1000.00, 500.50, 2500.00. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[1000.00, 500.50, 2500.00],
    )

    # Relationships
    item: Item | None = edge(
        label="REFERENCES_ITEM",
        default=None,
        description=(
            "WHAT: The product or service being billed. "
            "EXTRACT: As an object with identity fields (item_code, name, category). "
            "NEVER return as an array. "
            "LOOK FOR: Item details in line item description or separate item columns. "
            "EXAMPLES: {'item_code': 'SKU-12345', 'name': 'Laptop Computer'}. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
    )

    tax: Tax | None = edge(
        label="HAS_TAX",
        default=None,
        description="Tax applied to this line item",
    )


class Payment(BaseModel):
    """
    Payment information component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    method: PaymentMethod | None = Field(
        None,
        description=(
            "WHAT: Payment method used or accepted. "
            "EXTRACT: One of the allowed values (Bank Transfer, Card, Cash, Direct Debit, Cheque, Other). "
            "If not explicitly stated, omit this field (do not guess). "
            "LOOK FOR: 'Payment Method', 'Payment Means', 'Mode de Paiement' labels in payment section. "
            "EXAMPLES: Bank Transfer, Card, Cash, Direct Debit"
        ),
        examples=["Bank Transfer", "Card", "Cash"],
    )

    due_date: date | None = Field(
        None,
        description=(
            "Payment due date. "
            "LOOK FOR: 'Due Date', 'Payment Due', 'Date d'Échéance' labels. "
            "EXTRACT: Parse date and convert to YYYY-MM-DD format. "
            "EXAMPLES: '2024-02-15', '2024-03-01'"
        ),
        examples=["2024-02-15", "2024-03-01"],
    )

    terms: str | None = Field(
        None,
        description=(
            "Payment terms description. "
            "LOOK FOR: 'Payment Terms', 'Terms', 'Conditions' text. "
            "EXAMPLES: 'Net 30', 'Due on receipt', '2/10 Net 30'"
        ),
        examples=["Net 30", "Due on receipt", "2/10 Net 30"],
    )

    # Bank details (embedded)
    bank_name: str | None = Field(
        None,
        description=(
            "Bank name. "
            "LOOK FOR: 'Bank', 'Bank Name', 'Banque' in payment section. "
            "EXAMPLES: 'BNP Paribas', 'HSBC', 'Deutsche Bank'"
        ),
        examples=["BNP Paribas", "HSBC"],
    )

    iban: str | None = Field(
        None,
        description=(
            "International Bank Account Number. "
            "LOOK FOR: 'IBAN', 'Account Number' starting with country code. "
            "EXAMPLES: 'FR76 1234 5678 9012 3456 7890 123', 'GB29 NWBK 6016 1331 9268 19'"
        ),
        examples=["FR76 1234 5678 9012 3456 7890 123"],
    )

    bic: str | None = Field(
        None,
        description=(
            "Bank Identifier Code (SWIFT). "
            "LOOK FOR: 'BIC', 'SWIFT', 'Bank Code' (8 or 11 characters). "
            "EXAMPLES: 'BNPAFRPP', 'NWBKGB2L'"
        ),
        examples=["BNPAFRPP", "NWBKGB2L"],
    )

    reference: str | None = Field(
        None,
        description=(
            "Payment reference or message. "
            "LOOK FOR: 'Reference', 'Payment Reference', 'Communication' in payment section. "
            "EXAMPLES: 'Invoice INV-2024-001', 'RF18 5390 0754 7034'"
        ),
        examples=["Invoice INV-2024-001"],
    )


class Delivery(BaseModel):
    """
    Delivery information component.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    delivery_date: date | None = Field(
        None,
        description=(
            "WHAT: The delivery or shipment date. "
            "EXTRACT: In ISO format (YYYY-MM-DD). Parse natural language dates. "
            "LOOK FOR: 'Delivery Date', 'Ship Date', 'Date de Livraison' labels in delivery section. "
            "EXAMPLES: 2024-01-20, 2024-02-15. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=["2024-01-20", "2024-02-15"],
    )

    location: str | None = Field(
        None,
        description=(
            "Delivery location or address. "
            "LOOK FOR: 'Delivery Address', 'Ship To', 'Delivery Location' section. "
            "EXTRACT: Full address or location name. "
            "EXAMPLES: 'Main Warehouse, 123 Industrial Rd, London'"
        ),
        examples=["Main Warehouse, 123 Industrial Rd, London"],
    )


class DocumentReference(BaseModel):
    """
    Reference to another business document.
    Deduplicated by content.
    """

    model_config = ConfigDict(is_entity=False)

    ref_type: str | None = Field(
        None,
        description=(
            "Type of referenced document. "
            "LOOK FOR: 'PO', 'Purchase Order', 'Contract', 'Order No', 'Reference' labels. "
            "EXTRACT: Omit when not in document. "
            "EXAMPLES: 'Purchase Order', 'Contract', 'Previous Invoice', 'Project'"
        ),
        examples=["Purchase Order", "Contract", "Previous Invoice"],
    )

    ref_number: str | None = Field(
        None,
        description=(
            "Reference document number. "
            "LOOK FOR: Document numbers after reference type labels. "
            "EXTRACT: Omit when not in document. "
            "EXAMPLES: 'PO-2024-001', 'CONTRACT-12345', 'INV-2023-999'"
        ),
        examples=["PO-2024-001", "CONTRACT-12345"],
    )

    ref_date: date | None = Field(
        None,
        description=(
            "Date of referenced document. "
            "LOOK FOR: Date near reference number. "
            "EXTRACT: Parse and convert to YYYY-MM-DD format. "
            "EXAMPLES: '2024-01-15', '2023-12-20'"
        ),
        examples=["2024-01-15", "2023-12-20"],
    )


# =============================================================================
# ROOT BILLING DOCUMENT
# =============================================================================


class BillingDocument(BaseModel):
    """
    Root billing document entity (Invoice, Credit Note, Receipt, etc.).

    Uniquely identified by document_number.
    Represents the complete billing document with all related information.
    """

    model_config = ConfigDict(
        graph_id_fields=["document_number"], extra="ignore", populate_by_name=True
    )

    # --- Core Document Fields ---

    document_number: str = Field(
        ...,
        validation_alias=AliasChoices("document_number", "documentNumber", "invoice_number"),
        description=(
            "Invoice/document number (primary identifier). "
            "LOOK FOR: Large, bold text in header, 'Invoice No', 'Invoice Number', 'Receipt No', "
            "'Facture No', 'Número de Factura' labels (usually top right). "
            "EXTRACT: Complete number including prefixes/suffixes. "
            "EXAMPLES: 'INV-2024-001', '2024-INV-12345', 'REC-001', 'CN-2024-050'"
        ),
        examples=["INV-2024-001", "2024-INV-12345", "REC-001"],
    )

    document_type: DocumentType | None = Field(
        None,
        description=(
            "WHAT: The type of billing document. "
            "EXTRACT: One of the allowed values (Invoice, Credit Note, Debit Note, Receipt, Other). "
            "If not explicitly stated, omit this field (do not guess). "
            "LOOK FOR: Document title/header text ('INVOICE', 'CREDIT NOTE', 'RECEIPT', 'FACTURE'). "
            "EXAMPLES: Invoice, Credit Note, Receipt"
        ),
        examples=["Invoice", "Credit Note", "Receipt"],
    )

    issue_date: date | None = Field(
        None,
        description=(
            "WHAT: The date the document was issued. "
            "EXTRACT: In ISO format (YYYY-MM-DD). Parse natural language dates. "
            "LOOK FOR: 'Date', 'Issue Date', 'Invoice Date', 'Date d'Émission' labels in document header. "
            "EXAMPLES: 2024-01-15, 2024-02-20. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=["2024-01-15", "2024-02-20"],
    )

    due_date: date | None = Field(
        None,
        description=(
            "WHAT: The payment due date. "
            "EXTRACT: In ISO format (YYYY-MM-DD). Parse natural language dates. "
            "LOOK FOR: 'Due Date', 'Payment Due', 'Date d'Échéance' labels in header or payment section. "
            "EXAMPLES: 2024-02-15, 2024-03-20. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=["2024-02-15", "2024-03-20"],
    )

    currency: str | None = Field(
        None,
        description=(
            "Document currency (ISO 4217 code). "
            "LOOK FOR: Currency symbols (€, $, £) or codes (EUR, USD, GBP) near amounts. "
            "CONVERT: € → EUR, $ → USD, £ → GBP. "
            "EXAMPLES: 'EUR', 'USD', 'GBP', 'CHF'"
        ),
        examples=["EUR", "USD", "GBP"],
    )

    # --- Financial Totals (embedded) ---

    subtotal: float | None = Field(
        None,
        description=(
            "WHAT: Subtotal before tax and discounts. "
            "EXTRACT: Numeric value only (e.g., 1000.00, 5000.50). "
            "NEVER extract names, text, or dates into this field. "
            "LOOK FOR: 'Subtotal', 'Net Total', 'Total HT' labels in totals section. "
            "EXAMPLES: 1000.00, 5000.50. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[1000.00, 5000.50],
    )

    discount_total: float | None = Field(
        None,
        description=(
            "WHAT: Total discount amount applied. "
            "EXTRACT: Numeric value only (e.g., 100.00, 50.00). "
            "NEVER extract names, text, or dates into this field. "
            "LOOK FOR: 'Discount', 'Remise', 'Descuento' labels in totals section. "
            "EXAMPLES: 100.00, 50.00. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[100.00, 50.00],
    )

    tax_total: float | None = Field(
        None,
        description=(
            "Total tax amount. "
            "LOOK FOR: 'Tax', 'VAT', 'GST', 'Total Tax', 'TVA' in totals section. "
            "EXTRACT: Numeric value without currency symbols. "
            "EXAMPLES: 200.00, 100.10"
        ),
        examples=[200.00, 100.10],
    )

    total_amount: float | None = Field(
        None,
        description=(
            "WHAT: Final total amount due (including tax). "
            "EXTRACT: Numeric value only (e.g., 1200.00, 6000.60). "
            "NEVER extract names, text, or dates into this field. "
            "LOOK FOR: 'Total', 'Grand Total', 'Amount Due', 'Total TTC' labels (usually largest/bold number in totals). "
            "EXAMPLES: 1200.00, 6000.60. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[1200.00, 6000.60],
    )

    amount_paid: float | None = Field(
        None,
        description=(
            "WHAT: Amount already paid (if any). "
            "EXTRACT: Numeric value only (e.g., 500.00, 1000.00). "
            "NEVER extract names, text, or dates into this field. "
            "LOOK FOR: 'Paid', 'Amount Paid', 'Prepaid', 'Acompte' labels in totals section. "
            "EXAMPLES: 500.00, 1000.00. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[500.00, 1000.00],
    )

    balance_due: float | None = Field(
        None,
        description=(
            "WHAT: Remaining balance due after payments. "
            "EXTRACT: Numeric value only (e.g., 700.00, 5000.60). "
            "NEVER extract names, text, postal codes, or dates into this field. "
            "LOOK FOR: 'Balance Due', 'Amount Due', 'Solde', 'Restant à Payer' labels in totals section. "
            "EXAMPLES: 700.00, 5000.60. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
        examples=[700.00, 5000.60],
    )

    # --- Additional Information ---

    notes: str | None = Field(
        None,
        description=(
            "General notes or remarks. "
            "LOOK FOR: 'Notes', 'Remarks', 'Comments', 'Terms', 'Remarques' sections (usually at bottom). "
            "EXTRACT: Full text of notes/terms. "
            "EXAMPLES: 'Payment terms: Net 30 days. Thank you for your business.'"
        ),
        examples=["Payment terms: Net 30 days. Thank you for your business."],
    )

    # --- Relationships (Edges) ---

    seller: Party | None = edge(
        label="ISSUED_BY",
        default=None,
        description=(
            "WHAT: The party that issued this document (seller/supplier). "
            "EXTRACT: As an object with identity fields (name, tax_id, address, etc.). "
            "Omit when not present in extracted batch. "
            "LOOK FOR: Company information in header/top section, 'From', 'Seller', 'Supplier' labels. "
            "EXAMPLES: {'name': 'Acme Corp', 'tax_id': 'FR12345678901', 'city': 'Paris'}"
        ),
    )

    buyer: Party | None = edge(
        label="BILLED_TO",
        default=None,
        description=(
            "WHAT: The party receiving this document (buyer/customer). "
            "EXTRACT: As an object with identity fields (name, tax_id, address, etc.). "
            "NEVER return as an array. "
            "PRESERVE ROLE: 'To:', 'Bill To:', 'Customer:' indicates buyer. "
            "LOOK FOR: 'Bill To', 'Customer', 'Buyer', 'Client' section (often left side or below header). "
            "EXAMPLES: {'name': 'Tech Solutions Inc', 'city': 'London'}. "
            "If not found in document, OMIT this field entirely. NEVER use placeholder values like 'N/A' or 'Not specified'."
        ),
    )

    line_items: List[LineItem] = edge(
        label="CONTAINS_LINE",
        default_factory=list,
        description=(
            "Line items (products/services billed). "
            "LOOK FOR: Table with columns like 'Item', 'Description', 'Qty', 'Price', 'Total'. "
            "EXTRACT: Each row as a separate line item with all details."
        ),
    )

    taxes: List[Tax] = edge(
        label="HAS_TAX",
        default_factory=list,
        description=(
            "Tax breakdown by rate/category. "
            "LOOK FOR: 'Tax Breakdown', 'VAT Summary', tax table with rates and amounts. "
            "EXTRACT: Each tax rate as a separate tax entry."
        ),
    )

    payment: Payment | None = edge(
        label="HAS_PAYMENT_INFO",
        default=None,
        description=(
            "Payment information and instructions. "
            "LOOK FOR: 'Payment Information', 'Bank Details', 'Payment Terms' section (usually at bottom). "
            "EXTRACT: Payment method, bank details, due date, terms."
        ),
    )

    delivery: Delivery | None = edge(
        label="HAS_DELIVERY_INFO",
        default=None,
        description=(
            "Delivery information. "
            "LOOK FOR: 'Delivery', 'Ship To', 'Shipping' section. "
            "EXTRACT: Delivery date and location."
        ),
    )

    references: List[DocumentReference] = edge(
        label="REFERENCES_DOCUMENT",
        default_factory=list,
        description=(
            "References to other documents (PO, contracts, etc.). "
            "LOOK FOR: 'PO Number', 'Order No', 'Contract', 'Reference' fields. "
            "EXTRACT: Each reference as a separate entry with type and number."
        ),
    )

    # --- Validators ---

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, v: Any) -> Any:
        """Normalize currency symbols to ISO codes."""
        if not v:
            return v

        v_str = str(v).strip()

        # Currency symbol mappings
        symbol_map = {
            "€": "EUR",
            "$": "USD",
            "£": "GBP",
            "¥": "JPY",
            "₹": "INR",
            "₽": "RUB",
            "₩": "KRW",
            "₪": "ILS",
            "₺": "TRY",
            "₴": "UAH",
            "₱": "PHP",
            "฿": "THB",
        }

        if v_str in symbol_map:
            return symbol_map[v_str]

        # Normalize to uppercase
        v_upper = v_str.upper()
        if len(v_upper) == 3 and v_upper.isalpha():
            return v_upper

        return v_str
