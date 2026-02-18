# CHANGELOG

<!-- version list -->

## v1.4.3 (2026-02-18)

### Bug Fixes

- **ci**: Preserve release tag across rebase in semantic-release
  ([`84b4e51`](https://github.com/docling-project/docling-graph/commit/84b4e512feddfe2f0af85a6d1ceb10b90ab94aee))


## v1.4.2 (2026-02-18)

### Bug Fixes

- **ci**: Reset working tree before rebase in semantic-release push step
  ([`e31c29b`](https://github.com/docling-project/docling-graph/commit/e31c29b3d120d0ae25c4c427bf9e8d733edc4783))


## v1.4.1 (2026-02-18)

### Bug Fixes

- **ci**: Ensure GitHub release gets notes and assets in Release workflow
  ([`485b875`](https://github.com/docling-project/docling-graph/commit/485b87571214d2f284b23c1513e3a17dc2013c91))

### Chores

- **docs**: Refresh install steps and GitHub Pages content
  ([`07e1136`](https://github.com/docling-project/docling-graph/commit/07e11362cc02a8031b8755c10c5ed385b6382d77))


## v1.4.0 (2026-02-18)

### Bug Fixes

- **ci**: Allow semantic-release on protected main
  ([`9291bf5`](https://github.com/docling-project/docling-graph/commit/9291bf5df13575189d285aae5b03f16f658ebbfd))

- **ci**: Refactor semantic release workflow
  ([`7480a3e`](https://github.com/docling-project/docling-graph/commit/7480a3e735c63135ea65932f2af813f9f550dac8))

- **ci**: Simplify semantic release to single job
  ([`b10041f`](https://github.com/docling-project/docling-graph/commit/b10041ff904fb5d0e99f121789ea77001963a412))

- **ci**: Use inline DCO check to satisfy org action allowlist
  ([`3bfd298`](https://github.com/docling-project/docling-graph/commit/3bfd29816ee572b9aac2031e317c00ed7967417e))

- **delta**: Improve parent lookup quality gate handling with -1 disable and adaptive adjustment
  ([`d25567d`](https://github.com/docling-project/docling-graph/commit/d25567d393583d3fa67c69800ca6b374177720c0))

- **logs**: Add extraction-phase progress and move raw extracted payloads to trace only
  ([`dd32f66`](https://github.com/docling-project/docling-graph/commit/dd32f663d98fc9267769abb61b303a24c43a8da1))

- **schema**: Strengthen schema validation, guidance and deduplication; add string coercion fallback
  in backend
  ([`61e7520`](https://github.com/docling-project/docling-graph/commit/61e7520e99d13670a93e3f4f83100c01f3624905))

### Chores

- **ci**: Run ruff-format to fix pre-commit failure in delta runtime tests
  ([`ab1469c`](https://github.com/docling-project/docling-graph/commit/ab1469c705b5b90820961c55819ddd1012fa42f9))

- **deps**: Update types-setuptools requirement
  ([`15e7fa8`](https://github.com/docling-project/docling-graph/commit/15e7fa8c1915c7f6054ecab81ec426a79d5ad4f8))

- **refs**: Update documentation to new repository URL
  ([`f14fcd9`](https://github.com/docling-project/docling-graph/commit/f14fcd9b2a87ae3f76e797e24fbb930183b0eb79))

### Documentation

- **schema**: Update guides on descriptive IDs, enum synonyms, and deduplication patterns
  ([`7db205d`](https://github.com/docling-project/docling-graph/commit/7db205d6188e22166ecb79c67eb9a1e320ffb95a))

### Features

- **core**: Add entity name normalizer, description merge helper, and gleaning retry logic
  ([`8e5858a`](https://github.com/docling-project/docling-graph/commit/8e5858a425daf1bc20ee0ec3b647d555fe3fb7de))

- **delta**: Persist orphan parent_ids and reattach orphans by id match when multiple parent
  candidates exist
  ([`ead9096`](https://github.com/docling-project/docling-graph/commit/ead90968ac1e900ba4c98e0ab3f709ed73c93daf))

- **llm**: Support LM Studio as a local inference provider
  ([`d317459`](https://github.com/docling-project/docling-graph/commit/d31745914291d917760cb75f2e2b4a8ace62491d))

### Testing

- **coverage**: Add targeted unit tests to raise Codecov patch coverage
  ([`c39e02c`](https://github.com/docling-project/docling-graph/commit/c39e02cedc12e5e9cf2650e984699e2837d31c1a))

- **coverage**: Cover remaining patch lines in llm_backend, catalog, and convert
  ([`e09998d`](https://github.com/docling-project/docling-graph/commit/e09998da851f6880758b542ac420e1b76570d053))

- **coverage**: Further extend test coverage
  ([`5ec1562`](https://github.com/docling-project/docling-graph/commit/5ec156214c49b99599af284d8d9fa8acfd6d1b2f))


## v1.3.1 (2026-02-16)

### Bug Fixes

- **delta**: Backfill root ids, normalize paths, and repair scalar id fields before validation
  ([`6877b38`](https://github.com/docling-project/docling-graph/commit/6877b380673f553a8c10cd598d143ea9a3132992))

### Chores

- **deps**: Bump pillow from 11.3.0 to 12.1.1
  ([`9faa614`](https://github.com/docling-project/docling-graph/commit/9faa614b681559f234531640f596a949cbc374ea))


## v1.3.0 (2026-02-15)

### Bug Fixes

- **ci**: Remove unused mypy ignores for rapidfuzz and spacy imports
  ([`9fa8f75`](https://github.com/docling-project/docling-graph/commit/9fa8f75a453ec8f42c5aaa9f4174e42cbcb7344d))

- **delta**: Improve entity ID quality, limiting index-based ID inference, and enabling
  content-based dedup
  ([`4767e26`](https://github.com/docling-project/docling-graph/commit/4767e26e9bf5ffba10a094edc97a767d1ebc852d))

- **delta**: Prevent spurious list-entity nodes by adding identity allowlists and post-merge
  filtering
  ([`f45f790`](https://github.com/docling-project/docling-graph/commit/f45f790b670437d74d6bebb9ce22eaf402b39683))

### Chores

- **docs**: Update staged extraction docs, schema definition and performance tuning guides
  ([`bfabcbb`](https://github.com/docling-project/docling-graph/commit/bfabcbbf9d8a76cff5ff075407bffb9a004f13ad))

- **tests**: Update staged extraction tests and remove obsolete structure-only coverage
  ([`2483a4b`](https://github.com/docling-project/docling-graph/commit/2483a4b4027082ad85ca8c1b5e8064f356304771))

### Documentation

- **delta**: Document delta extraction contract (flat graph IR), config/CLI flags, and migration
  notes
  ([`66aa6be`](https://github.com/docling-project/docling-graph/commit/66aa6bee837a1211fa6879d76b16c8157cffab7b))

- **traces**: Refresh pages with updated output handling and debug artifacts
  ([`07e0cbc`](https://github.com/docling-project/docling-graph/commit/07e0cbc985727e7c9502493d09c52aba3532b855))

### Features

- **contracts**: Harden llm pipeline w/ contract dispatch, staged extraction, deterministic merge &
  observability
  ([`92a5089`](https://github.com/docling-project/docling-graph/commit/92a50895a6f3ec94c1036bc56149afc3c78ee949))

- **contracts**: Improve catalog definition, flatten ID discovery & add validation retries
  ([`a1aba89`](https://github.com/docling-project/docling-graph/commit/a1aba893f4fd0527f1164b047c228628d9bddc00))

- **delta**: Add opt-in delta contract with flat graph IR batching, global merge/dedup, and template
  projection controls
  ([`0b19e08`](https://github.com/docling-project/docling-graph/commit/0b19e089bda1ce1c47b4e0b6022c479a82bab155))

- **llm**: Enable default schema-enforced structured output via LiteLLM with prompt-schema fallback
  ([`6e96f54`](https://github.com/docling-project/docling-graph/commit/6e96f543db9c0f24b8ff8db38e4055fdcd9ec004))

- **llm_clients**: Support custom OpenAI-compatible endpoints via env-based auth and init
  scaffolding
  ([`0bebc44`](https://github.com/docling-project/docling-graph/commit/0bebc44beb2030e05eebcacb7ea366b7dcfe427f))

### Refactoring

- **input**: Unify ingestion via Docling conversion with DoclingDocument passthrough
  ([`689426b`](https://github.com/docling-project/docling-graph/commit/689426b085c78cfcefa65f969745f3cc463926fa))

- **trace**: Improve stage naming and split serializer into helpers
  ([`0378f65`](https://github.com/docling-project/docling-graph/commit/0378f651a46ebd4fd59d693a03018ad8043e0e0c))

- **trace**: Revamp debug trace_data into a chronological event log
  ([`4ba4b5b`](https://github.com/docling-project/docling-graph/commit/4ba4b5bd3df54a5052e523d8456784c97b922267))

### Testing

- **delta**: Update unit + integration coverage for delta batching/merge/projection and contract
  routing
  ([`8150def`](https://github.com/docling-project/docling-graph/commit/8150defb14bbb56f158f090164836c534b2e8254))


## v1.2.4 (2026-02-10)

### Bug Fixes

- **ci**: Use typing_extensions.Self in Pydantic templates for Python 3.10 compatibility
  ([`912e16c`](https://github.com/docling-project/docling-graph/commit/912e16c62d4267c405f0b7a6126967a530e17c84))

### Chores

- **deps**: Bump nbconvert from 7.16.6 to 7.17.0
  ([`1485e5d`](https://github.com/docling-project/docling-graph/commit/1485e5d0828f3f538d5ecf52ef20e854110fd8a2))

- **deps**: Update types-setuptools requirement
  ([`6e4fc37`](https://github.com/docling-project/docling-graph/commit/6e4fc37086bf7a923e9cf8eb013211ad2ac8cd9b))

- **docs**: Document and exemplify custom LLM client (BYO URL/auth)
  ([`87e730a`](https://github.com/docling-project/docling-graph/commit/87e730a5652a363efadfa82e8d9c13b709f2ce6b))

### Refactoring

- **llm_clients**: Use LiteLLM as single gateway for local and remote providers
  ([`94cad99`](https://github.com/docling-project/docling-graph/commit/94cad99ef4bd86f8832180a784c84fc20a323bdf))

- **trace**: Unify TraceData and debug flow
  ([`d25ef06`](https://github.com/docling-project/docling-graph/commit/d25ef06a46d216e4a86a6ad7f2e03d6ebea4485f))


## v1.2.3 (2026-01-26)

### Documentation

- **pages**: Update references and examples related to rheology template
  ([`35a9243`](https://github.com/docling-project/docling-graph/commit/35a9243e59bad9f0ed084269c27670e8aee19bd3))

- **templates**: Fix broken references to ScholarlyRheologyPaper template
  ([`a6d994b`](https://github.com/docling-project/docling-graph/commit/a6d994b1dec497d39f1a61de8c824500fd5c2a0f))

### Refactoring

- **templates**: Improve slurry-battery rheology Pydantic schema
  ([`b737343`](https://github.com/docling-project/docling-graph/commit/b737343db31433f6be64fb437750a6e044aa3d92))


## v1.2.2 (2026-01-26)

### Bug Fixes

- **converters**: Preserve component data during graph pruning
  ([`8552ea5`](https://github.com/docling-project/docling-graph/commit/8552ea55aa5ce76d9b284f4582217ba15ae39dcc))

- **converters**: Tighten error logging, and improve node-id collision detection
  ([`bafab02`](https://github.com/docling-project/docling-graph/commit/bafab024238fd59adfa5153611d8643abdf8794f))

- **input**: Add User-Agent header for URL downloads to avoid 403 and add tests
  ([`77dbd02`](https://github.com/docling-project/docling-graph/commit/77dbd029351741354c4d6c2c21cbb19002b98df2))

- **pipeline**: Auto-clean empty output directories on failure when dump_to_disk is enabled
  ([`9e4c031`](https://github.com/docling-project/docling-graph/commit/9e4c0312528fc1352fa403dc34a17403e952447c))

- **release**: Restore default semantic-release templates and regenerate changelog
  ([`59b2f43`](https://github.com/docling-project/docling-graph/commit/59b2f43caa4541140579eba5bedf4da73e9fd91f))

- **validation**: Make billing template validators lenient with coercion logging
  ([`fb1bb37`](https://github.com/docling-project/docling-graph/commit/fb1bb3707e16b5282a17f308d249718efbd92c33))

- **visualization**: Render nested node/edge details as formatted JSON
  ([`014778a`](https://github.com/docling-project/docling-graph/commit/014778ac9fc6cc8609d476d6f6e2ff7a098f1ceb))

### Chores

- **deps**: Update aiofiles requirement
  ([`afc6d52`](https://github.com/docling-project/docling-graph/commit/afc6d52a1371c45fc879e914215a68a60691723a))

- **docs**: Refine wording and improve styling
  ([`e7b7f0a`](https://github.com/docling-project/docling-graph/commit/e7b7f0af509958b8bb48e27ac21553eff2671e32))

- **docs**: Update examples and navigation to align with BillingDocument schema references
  ([`c254825`](https://github.com/docling-project/docling-graph/commit/c254825ffd17cab7f88f9c9ca35f2f823ca2d6e7))

### Documentation

- **problem-statement**: Update content to reflect recent docling-graph improvements
  ([`872a3a3`](https://github.com/docling-project/docling-graph/commit/872a3a37d11a6c89fd11df91bec604a34c6b60b9))

### Refactoring

- **templates**: Add comprehensive billing document Pydantic extraction template
  ([`21a2200`](https://github.com/docling-project/docling-graph/commit/21a22002520f113a3ba2c8f17a67cdcec05a1232))

- **templates**: Simplify BillingDocument schema and prompts for better extraction
  ([`81fdbc9`](https://github.com/docling-project/docling-graph/commit/81fdbc912ce3164f979b37e06aaada73aec906d9))


## v1.2.1 (2026-01-25)

### Bug Fixes

- **ci**: Apply ruff formatter to codebase
  ([`a224192`](https://github.com/docling-project/docling-graph/commit/a2241925b9cc0410916c34d954fe651e03ef5406))

- **ci**: Remove invalid dependabot commit-message include property
  ([`f4c9cff`](https://github.com/docling-project/docling-graph/commit/f4c9cff47787934a7fd8d840a752effc7f062aad))

### Chores

- **mkdocs**: Streamline structure, align API refs, and convert notes to admonitions
  ([`3b03b63`](https://github.com/docling-project/docling-graph/commit/3b03b63c740971ea87dccc9746fc38e2d6e9d403))

### Documentation

- **examples**: Replace legacy content with updated scripts and CLI recipes guide
  ([`005597d`](https://github.com/docling-project/docling-graph/commit/005597d79b40678380d533d097c535c56a835d5f))

- **examples**: Replace legacy content with updated scripts and CLI recipes guide
  ([`84bf728`](https://github.com/docling-project/docling-graph/commit/84bf728763ca12aec449818688bf407b6b7df242))


## v1.2.0 (2026-01-25)

### Bug Fixes

- **llm**: Restore accurate model capability detection and provider resolution
  ([`5efd6bc`](https://github.com/docling-project/docling-graph/commit/5efd6bccefd76e46f874a5c324b240d102e1d966))

- **llm_clients**: Handle Mistral SDK timeout incompatibility and model attribute access
  ([`6691be4`](https://github.com/docling-project/docling-graph/commit/6691be43c8bc591e4eba0f1c888fc98d5b43f50d))

- **pipeline**: Correct trace data table detection and visualization directory handling
  ([`38b1e0b`](https://github.com/docling-project/docling-graph/commit/38b1e0b0a51eb9974d1ee150f3423012b8574ed4))

- **pipeline**: Prevent unwanted disk exports in API mode with dump_to_disk control
  ([`ffb4264`](https://github.com/docling-project/docling-graph/commit/ffb4264ca9bbe02fdc164e78c9a013b7de10fa1c))

- **pipeline**: Resolve trace collection & output dir structure
  ([`35d1f0f`](https://github.com/docling-project/docling-graph/commit/35d1f0f536c57d1eaa31ec2082f09088c822fa6b))

### Chores

- **build**: Stop tracking generated mkdocs dir
  ([`e0b8f7e`](https://github.com/docling-project/docling-graph/commit/e0b8f7ea27a217fa09fda7f665241a554fc9a77b))

### Features

- **export**: Add trace data system, unified exports, and optimized io ops
  ([`b52e342`](https://github.com/docling-project/docling-graph/commit/b52e342759a2e51773969a3dc260b0849dd19fd3))

### Refactoring

- **metadata**: Add source for intermediate graphs info to trace_summary
  ([`3a90796`](https://github.com/docling-project/docling-graph/commit/3a907962d5f69b97b6126123777239d2e822aa57))

- **metadata**: Use dynamic version and improve json layout
  ([`6d431d8`](https://github.com/docling-project/docling-graph/commit/6d431d817b4413a89a3debc8f22ecb9f9e1b1a28))

- **pipeline**: Streamline output structure and content
  ([`2ee6d42`](https://github.com/docling-project/docling-graph/commit/2ee6d4261c468991d7184e3e30143f29850799a8))


## v1.1.0 (2026-01-24)

### Bug Fixes

- **llm**: Handle max_tokens truncation with partial recovery and clear warnings
  ([`f0b2f53`](https://github.com/docling-project/docling-graph/commit/f0b2f530692f42d7a8af02411cf38fc2f8adde94))

- **llm**: Prevent vLLM hanging with max token limits and request timeouts
  ([`605bc99`](https://github.com/docling-project/docling-graph/commit/605bc9977872b4dde89daa1bbe8c80e5d4e3d0cb))

### Chores

- **docs**: Improve GitHub Pages styling and layout
  ([`78d6a02`](https://github.com/docling-project/docling-graph/commit/78d6a022ceadd47e83dc55be0eaaadfdae96dc2b))

- **docs**: Update docs with latest features and changes
  ([`8994f72`](https://github.com/docling-project/docling-graph/commit/8994f72323dacd94959e33b0051582d9c13b486f))

- **docs**: Update Python badge to remove pre-release notation
  ([`4588eb0`](https://github.com/docling-project/docling-graph/commit/4588eb0cc1837737c18ee0a642841c0fc6c58a9a))

### Documentation

- **structure**: Reorganize documentation into minimalist layout
  ([`d705039`](https://github.com/docling-project/docling-graph/commit/d7050394219e3302813624471271af5b1b9fd4c7))

### Features

- **docs**: Enable dynamic Mermaid flowchart loading with custom styling
  ([`2c16d25`](https://github.com/docling-project/docling-graph/commit/2c16d25899f2581d3bbb3dcf7766aed84d051a72))

- **extraction**: Overhaul extraction layer with adaptive prompting, batching optimization, and
  robust recovery
  ([`1507dec`](https://github.com/docling-project/docling-graph/commit/1507dec68363df4b84e71cd2ed8c8bebefd910a4))


## v1.0.0 (2026-01-23)

### Chores

- **docling**: Bump docling dependency to version 2.70
  ([#34](https://github.com/docling-project/docling-graph/pull/34),
  [`023841f`](https://github.com/docling-project/docling-graph/commit/023841fec00cdcdc92689048e0416e94a8ca66c8))

### Documentation

- Review and polish documentation for GitHub Pages
  ([#34](https://github.com/docling-project/docling-graph/pull/34),
  [`023841f`](https://github.com/docling-project/docling-graph/commit/023841fec00cdcdc92689048e0416e94a8ca66c8))

- **flowcharts**: Update diagrams and add multi-input logic
  ([#34](https://github.com/docling-project/docling-graph/pull/34),
  [`023841f`](https://github.com/docling-project/docling-graph/commit/023841fec00cdcdc92689048e0416e94a8ca66c8))

### Features

- **input**: Add multi-format input support and normalization layer
  ([#34](https://github.com/docling-project/docling-graph/pull/34),
  [`023841f`](https://github.com/docling-project/docling-graph/commit/023841fec00cdcdc92689048e0416e94a8ca66c8))

- **input**: Complete multi-format input extension for Docling Graph pipeline
  ([#34](https://github.com/docling-project/docling-graph/pull/34),
  [`023841f`](https://github.com/docling-project/docling-graph/commit/023841fec00cdcdc92689048e0416e94a8ca66c8))


## v0.4.1 (2026-01-22)

### Bug Fixes

- **ci**: Fix -exec syntax in GitHub release workflow
  ([`f40d189`](https://github.com/docling-project/docling-graph/commit/f40d18931055df5286013ecc61b32a909c7a574b))


## v0.4.0 (2026-01-22)

### Chores

- **ci**: Correct PyPI deployment in release workflow
  ([`c985d25`](https://github.com/docling-project/docling-graph/commit/c985d25059629fa5b63e36f9b283678158c392b7))

### Documentation

- **module**: Refactor complete documentation suite for docling-graph
  ([`7b8cd6c`](https://github.com/docling-project/docling-graph/commit/7b8cd6c8bb92e3c10b6bc105d4888ad9a90f6a67))

### Features

- **cli**: Align CLI with refactored core and improve performance
  ([`d87f8c8`](https://github.com/docling-project/docling-graph/commit/d87f8c835d8314a2a2661e21064e55b5b8513bd0))

### Refactoring

- **core**: Major LLM client and pipeline simplification
  ([`20b904c`](https://github.com/docling-project/docling-graph/commit/20b904c8695146a5e1803480d720ee674c41816f))

### Testing

- **core**: Update unit and integration tests for refactored architecture
  ([`7d61dd7`](https://github.com/docling-project/docling-graph/commit/7d61dd734af22243738329f5d146fc71a61408f7))


## v0.3.0 (2026-01-22)

### Bug Fixes

- **release**: Resolve semantic-release changelog config deprecation
  ([`aa64dab`](https://github.com/docling-project/docling-graph/commit/aa64dab707daf5225569edce98286a1702781c37))

### Chores

- **ci**: Exclude attestation files from GitHub releases
  ([`1ae2685`](https://github.com/docling-project/docling-graph/commit/1ae2685ccba1234ed99d6eb95e5f8f39525237bf))

- **deps**: Bump the all-actions group with 5 updates
  ([#30](https://github.com/docling-project/docling-graph/pull/30),
  [`e7eace5`](https://github.com/docling-project/docling-graph/commit/e7eace5eb73ef3cb2a4ddb155e1698710986b16b))

### Documentation

- **deps**: Add MkDocs and documentation dependencies
  ([`c4378d3`](https://github.com/docling-project/docling-graph/commit/c4378d39b347aee266642c270f41fb9756e19d05))

- **readme**: Fix PyPI badge to include prereleases
  ([`ca3b14f`](https://github.com/docling-project/docling-graph/commit/ca3b14fa7f960348f2d3a257173843a7dc9f81ec))

- **readme**: Update documentation section with MkDocs links
  ([`87e0990`](https://github.com/docling-project/docling-graph/commit/87e09906718ac14b77dfd1a48c4161919efd6ab7))

- **repo**: Move community health files to .github and update links
  ([`2ab0471`](https://github.com/docling-project/docling-graph/commit/2ab04714f85bbe24f26c0b9ef931674ce22c90ee))

### Refactoring

- **core**: Remove document caching for stateless operation
  ([#32](https://github.com/docling-project/docling-graph/pull/32),
  [`3cf8bd0`](https://github.com/docling-project/docling-graph/commit/3cf8bd0cbe855b022ecb9a7b056468c8fcc98b17))


## v0.2.5 (2026-01-21)

### Bug Fixes

- **ci**: Disable attestations for TestPyPI to prevent conflict
  ([`2f2d79b`](https://github.com/docling-project/docling-graph/commit/2f2d79bc44e9419453332be0b7bf02adf8be6aa4))


## v0.2.4 (2026-01-21)

### Bug Fixes

- **ci**: Disable attestations for TestPyPI to prevent conflict
  ([`0cb6022`](https://github.com/docling-project/docling-graph/commit/0cb6022a7df5a113dc1399c8d78f293653443206))


## v0.2.3 (2026-01-21)

### Bug Fixes

- **ci**: Skip TestPyPI installation test due to dependency issues
  ([`da001fd`](https://github.com/docling-project/docling-graph/commit/da001fd8de88c0cc82f07ed1888f8e4945ad3498))

### Chores

- **whitesource**: Update policies and scanning rules
  ([`211d55e`](https://github.com/docling-project/docling-graph/commit/211d55e6ead84c5b7a77750a3275802609a6154b))


## v0.2.2 (2026-01-21)

### Bug Fixes

- **ci**: Add retry logic for TestPyPI package availability
  ([`d9b9bd7`](https://github.com/docling-project/docling-graph/commit/d9b9bd736bb9e81ed4ce91b5c380bf4b9d4549df))

- **ci**: Improve release workflow and commit message
  ([`2ce1ac6`](https://github.com/docling-project/docling-graph/commit/2ce1ac6f5c6082e29fb52636b0056dd7cec2c4dc))

### Chores

- **deps**: Clean up Dependabot commit message configuration
  ([`1a4b46d`](https://github.com/docling-project/docling-graph/commit/1a4b46df97d72fd83b108f7d7de30b1fc4c41306))

- **deps)(deps**: Bump the all-actions group across 1 directory with 7 updates
  ([#29](https://github.com/docling-project/docling-graph/pull/29),
  [`0a26352`](https://github.com/docling-project/docling-graph/commit/0a26352e656bacee444be753bb9786a24bfe49c7))

- **deps)(deps**: Bump the dev-dependencies group with 6 updates
  ([#27](https://github.com/docling-project/docling-graph/pull/27),
  [`1bcdd58`](https://github.com/docling-project/docling-graph/commit/1bcdd5850a1bed61ffb249cd1ac0761b5504e19c))

- **lint**: Update Ruff linter configuration
  ([`c452d5d`](https://github.com/docling-project/docling-graph/commit/c452d5df17e658bd8b28014bd38661329a200dad))


## v0.2.1 (2026-01-21)

### Bug Fixes

- **ci**: Strip 'v' prefix from tag for TestPyPI installation
  ([`2ddfb42`](https://github.com/docling-project/docling-graph/commit/2ddfb420f23678d781d9e869c3afe8374cdccda5))

### Chores

- **docs**: Update code of conduct
  ([`ff06127`](https://github.com/docling-project/docling-graph/commit/ff0612709687c77c8f9d35b29c8ba243f8dd14a5))

- **docs**: Update contributing guidelines
  ([`0d9061d`](https://github.com/docling-project/docling-graph/commit/0d9061d8cc3c41bf04a4ae261d3b8d803510e762))

- **docs**: Update governance
  ([`797261a`](https://github.com/docling-project/docling-graph/commit/797261ad1f254d78eaad35b8b8e8a11bf6739d78))

- **docs**: Update maintainers
  ([`ddb40a7`](https://github.com/docling-project/docling-graph/commit/ddb40a7003fc680a3df0eb853df1ea0bea942df9))

- **docs**: Update maintainers
  ([`02ff051`](https://github.com/docling-project/docling-graph/commit/02ff05162e0996ad9a53d0bc944f736102301801))


## v0.2.0 (2026-01-21)

### Features

- Configure automated semantic versioning
  ([`bdace4c`](https://github.com/docling-project/docling-graph/commit/bdace4c318b5f8362ce04d9a6bb199c74f846815))


## v0.1.0 (2026-01-21)

- Initial Release
