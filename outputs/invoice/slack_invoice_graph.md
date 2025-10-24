# Knowledge Graph Report

---

## Nodes (with properties):

### Node: Invoice (`Invoice_c84b53e96b`)
- **id**: Invoice_c84b53e96b
- **label**: Invoice
- **bill_no**: 1223113
- **date**: 2023-12-01
- **currency**: CHF
- **subtotal**: 1725.0
- **vat_amount**: 0.0
- **total**: 1725.0

### Node: Issuer (`Issuer_8d96021948`)
- **id**: Issuer_8d96021948
- **label**: Issuer
- **name**: Slack

### Node: Address (`Address_85a65de79b`)
- **id**: Address_85a65de79b
- **label**: Address
- **street**: 500 Howard Street
- **postal_code**: 94105
- **city**: San Francisco
- **country**: CA

### Node: Client (`Client_c873814ac1`)
- **id**: Client_c873814ac1
- **label**: Client
- **name**: MineralTree

### Node: Address (`Address_aa5185e65d`)
- **id**: Address_aa5185e65d
- **label**: Address
- **street**: 101 Arch Street
- **postal_code**: 02110
- **city**: Boston
- **country**: MA

### Node: LineItem (`LineItem_6a33c9a6a6`)
- **id**: LineItem_6a33c9a6a6
- **label**: LineItem
- **description**: Business+ Monthly User License
- **quantity**: 1.0
- **unit**: user
- **unit_price**: 307.35
- **total**: 307.35

## Edges (with labels):

- **Invoice → Issuer**  `issued_by`
- **Invoice → Client**  `sent_to`
- **Invoice → LineItem**  `contains_items`
- **Issuer → Address**  `located_at`
- **Client → Address**  `lives_at`

---

_End of Graph Summary_
