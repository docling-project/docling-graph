# CHANGELOG


## v0.1.0 (2026-01-21)

### Bug Fixes

- Correct semantic-release configuration
  ([`a8736a4`](https://github.com/IBM/docling-graph/commit/a8736a4fdda26fb9015a598aa351193e5fcff18f))

Signed-off-by: Ayoub EL BOUCHTILI <ayoub.elbouchtili@fr.ibm.com>

- **converter**: Add debuf for local one to one pipeline
  ([`dae7588`](https://github.com/IBM/docling-graph/commit/dae758812342a41b6a591aa8394de2e2a24642ef))

- **core**: Add init scripts to fix imports issues
  ([`c5d77cc`](https://github.com/IBM/docling-graph/commit/c5d77cc5ec27437b115a8374593a8a65d2e4160f))

- **dependencies**: Reverted uv syntax to support old versions
  ([`c0f40ac`](https://github.com/IBM/docling-graph/commit/c0f40ac6afa4d44f60822e26f9b7d7dd723798a2))

- **doc-proc**: Misleading log message, pipeline supports english and french
  ([`01d6746`](https://github.com/IBM/docling-graph/commit/01d674664cb0bda102eff5547c4ccb0fb948355a))

- **docs**: Add pytest badge for test status indication
  ([`be807ba`](https://github.com/IBM/docling-graph/commit/be807ba267e5252652246fa226c6d1706f4525fe))

- **docs**: Add pytest badge for test status indication
  ([`71addff`](https://github.com/IBM/docling-graph/commit/71addffd333ed27ffdf42d2ae58a1fc676c8bf49))

- **docs**: Correct flowchart node configuration in README
  ([`c5ae224`](https://github.com/IBM/docling-graph/commit/c5ae22422bd16c5f4cc91ecf039e15d73179f760))

- **docs**: Update links following repository migration
  ([`fbb89c3`](https://github.com/IBM/docling-graph/commit/fbb89c395ead8ba2442c771d192a8fbecfd2472f))

- **examples**: Resolve Ruff linting errors in Pydantic templates
  ([`f1578f4`](https://github.com/IBM/docling-graph/commit/f1578f4c0d02d5a9ddf901313200de04ccd4186c))

- **graph-converter**: Taking into account is_entity=false
  ([`e8024f1`](https://github.com/IBM/docling-graph/commit/e8024f154f292a6e68e7b573160164cda6f90b25))

- **id_registry**: Refine fingerprint generation to prevent hash collisions
  ([`b78d779`](https://github.com/IBM/docling-graph/commit/b78d779e44059cf4231c95f39389c286c06d23b1))

- **mock**: Update extract methods to return proper lists instead of Mock objects
  ([`a62c087`](https://github.com/IBM/docling-graph/commit/a62c0870cd52659d1353e71aa83270f13eb3ddc6))

- **notebook**: Improve markdown report formatting and display
  ([`2fa1d39`](https://github.com/IBM/docling-graph/commit/2fa1d399edf773c4315fba8e9998f024cc7655f6))

- **notebook**: Update execution counts and adjust PNG visualization width
  ([`d702aa8`](https://github.com/IBM/docling-graph/commit/d702aa880448d2be240d117f51f9ad1c0db86b31))

- **pipeline**: Add required pipeline fixes following graph api refactoring
  ([`0b9ecca`](https://github.com/IBM/docling-graph/commit/0b9eccaac6e7debed668af3ff4e56573eb84c29a))

- **pipeline**: Avoid reconverting when document already converted
  ([`69b365e`](https://github.com/IBM/docling-graph/commit/69b365e121dd89e5afd1c69ce52ee8db794b5208))

- **pipeline**: Better output management
  ([`b698709`](https://github.com/IBM/docling-graph/commit/b698709d2af7491d751224a8a42a56a2285cc115))

- **setup**: Correct optional dependencies and lazy import handling based on inference site
  ([`8d59ceb`](https://github.com/IBM/docling-graph/commit/8d59ceb7cf4fc2c87b287c336eb6b02bf6f6d746))

- **templates**: Use converter id generation instead of relying on llm extraction
  ([`6a95e48`](https://github.com/IBM/docling-graph/commit/6a95e48594f7495f669732bfc6ecf28db67dd5da))

- **tests**: Adjust test assertion for merging empty list to expect None
  ([`4e6b51f`](https://github.com/IBM/docling-graph/commit/4e6b51f8f53f2534554628963ca28d8bc27205c5))

- **tests**: Update import paths for extractors to core module
  ([`cf7660c`](https://github.com/IBM/docling-graph/commit/cf7660c3a76a4816bb4c1525bd98aff4249a0aa2))

- **tests**: Update incorrect import paths in test files
  ([`7bab57a`](https://github.com/IBM/docling-graph/commit/7bab57ab2ab6054cf69ae9614c59d79e6d81d9ae))

- **types**: Resolve MyPy errors for imports, hints, and base class init
  ([`a19ca27`](https://github.com/IBM/docling-graph/commit/a19ca27d17f7770f00741c16d9096ba4bfb8af37))

- **visualizer**: Udpate essential columns for tooltip display using cosmograph
  ([`eade954`](https://github.com/IBM/docling-graph/commit/eade954dd09844f657341c0e59e5e683fb146767))

- **visualizers**: Update default output path for CosmoGraph visualization
  ([`69e9c0e`](https://github.com/IBM/docling-graph/commit/69e9c0e1cc2f16b29940f4c667dc79cb4682deea))

- **watsonx**: Fix json parsing post watsonx api call
  ([`b036044`](https://github.com/IBM/docling-graph/commit/b036044836e133168c7dd93415a101ef6a0fed33))

### Chores

- **attributes**: Workaround GitHub Linguist misclassification
  ([`03749a1`](https://github.com/IBM/docling-graph/commit/03749a1b5d3ef4ec0de43e0440f7bdad2905d4f8))

- **cli**: Support llm-consolidation and use-chunking flags
  ([`6ac5dae`](https://github.com/IBM/docling-graph/commit/6ac5dae1eb3d08af74449a274ded552599f43778))

- **config**: Stop tracking personal config from root
  ([`e8b265c`](https://github.com/IBM/docling-graph/commit/e8b265c4e5839a5607244674ebf0f4531ae3f550))

- **core**: Fix ruff linter errors
  ([`43acc4e`](https://github.com/IBM/docling-graph/commit/43acc4ec921fdb998d78e437faaa37fc59a9e70a))

- **core**: Fix type incompatibility based on mypy validation
  ([`63a3713`](https://github.com/IBM/docling-graph/commit/63a371380ae00cb0932717c0015f98dd9a8c7698))

- **dependencies**: Clean up and update dependencies
  ([`d0eef47`](https://github.com/IBM/docling-graph/commit/d0eef478cc5e81e9da1c3d9c6498cdb3e6fb03ff))

- **dependencies**: Clean up and update dependencies
  ([`adf4252`](https://github.com/IBM/docling-graph/commit/adf4252d579ecb58898e35054554527752aff941))

- **deps**: Bump aiohttp from 3.13.2 to 3.13.3 ([#21](https://github.com/IBM/docling-graph/pull/21),
  [`7e7e678`](https://github.com/IBM/docling-graph/commit/7e7e678829e5fb59cba27eb13a6123e4f7ea7932))

--- updated-dependencies: - dependency-name: aiohttp dependency-version: 3.13.3

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

Co-authored-by: Ayoub El Bouchtili <Ayoub.elbouchtili@fr.ibm.com>

- **deps**: Bump cbor2 from 5.7.1 to 5.8.0 ([#22](https://github.com/IBM/docling-graph/pull/22),
  [`4f74dc9`](https://github.com/IBM/docling-graph/commit/4f74dc9f7cdb02b4abd719f8a0e9b038f4dfbbae))

Bumps [cbor2](https://github.com/agronholm/cbor2) from 5.7.1 to 5.8.0. - [Release
  notes](https://github.com/agronholm/cbor2/releases) -
  [Commits](https://github.com/agronholm/cbor2/compare/5.7.1...5.8.0)

--- updated-dependencies: - dependency-name: cbor2 dependency-version: 5.8.0

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

Co-authored-by: Ayoub El Bouchtili <Ayoub.elbouchtili@fr.ibm.com>

- **deps**: Bump filelock from 3.20.0 to 3.20.3
  ([#19](https://github.com/IBM/docling-graph/pull/19),
  [`3435653`](https://github.com/IBM/docling-graph/commit/3435653f66b4b297f1f55192df704928fff52ae8))

Bumps [filelock](https://github.com/tox-dev/py-filelock) from 3.20.0 to 3.20.3. - [Release
  notes](https://github.com/tox-dev/py-filelock/releases) -
  [Changelog](https://github.com/tox-dev/filelock/blob/main/docs/changelog.rst) -
  [Commits](https://github.com/tox-dev/py-filelock/compare/3.20.0...3.20.3)

--- updated-dependencies: - dependency-name: filelock dependency-version: 3.20.3

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

Co-authored-by: Ayoub El Bouchtili <Ayoub.elbouchtili@fr.ibm.com>

- **deps**: Bump mlx from 0.29.3 to 0.29.4 ([#24](https://github.com/IBM/docling-graph/pull/24),
  [`de5b8db`](https://github.com/IBM/docling-graph/commit/de5b8db35b9cd0506ec93182ef80b7f0ba36151a))

Bumps [mlx](https://github.com/ml-explore/mlx) from 0.29.3 to 0.29.4. - [Release
  notes](https://github.com/ml-explore/mlx/releases) -
  [Commits](https://github.com/ml-explore/mlx/compare/v0.29.3...v0.29.4)

--- updated-dependencies: - dependency-name: mlx dependency-version: 0.29.4

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

Co-authored-by: Ayoub El Bouchtili <Ayoub.elbouchtili@fr.ibm.com>

- **deps**: Bump pyasn1 from 0.6.1 to 0.6.2 ([#16](https://github.com/IBM/docling-graph/pull/16),
  [`005f736`](https://github.com/IBM/docling-graph/commit/005f736cd98a8fa537d2e195d5ba51fde9526aee))

Bumps [pyasn1](https://github.com/pyasn1/pyasn1) from 0.6.1 to 0.6.2. - [Release
  notes](https://github.com/pyasn1/pyasn1/releases) -
  [Changelog](https://github.com/pyasn1/pyasn1/blob/main/CHANGES.rst) -
  [Commits](https://github.com/pyasn1/pyasn1/compare/v0.6.1...v0.6.2)

--- updated-dependencies: - dependency-name: pyasn1 dependency-version: 0.6.2

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

- **deps**: Bump ray from 2.51.1 to 2.52.1 ([#23](https://github.com/IBM/docling-graph/pull/23),
  [`3da0ece`](https://github.com/IBM/docling-graph/commit/3da0ecea2176585a51b2a4276a7ef6edcfbb266f))

Bumps [ray](https://github.com/ray-project/ray) from 2.51.1 to 2.52.1. - [Release
  notes](https://github.com/ray-project/ray/releases) -
  [Commits](https://github.com/ray-project/ray/compare/ray-2.51.1...ray-2.52.1)

--- updated-dependencies: - dependency-name: ray dependency-version: 2.52.1

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

Co-authored-by: Ayoub El Bouchtili <Ayoub.elbouchtili@fr.ibm.com>

- **deps**: Bump urllib3 from 2.5.0 to 2.6.3 ([#17](https://github.com/IBM/docling-graph/pull/17),
  [`6454c29`](https://github.com/IBM/docling-graph/commit/6454c29940db618aa61998e11802f20a102b66cf))

Bumps [urllib3](https://github.com/urllib3/urllib3) from 2.5.0 to 2.6.3. - [Release
  notes](https://github.com/urllib3/urllib3/releases) -
  [Changelog](https://github.com/urllib3/urllib3/blob/main/CHANGES.rst) -
  [Commits](https://github.com/urllib3/urllib3/compare/2.5.0...2.6.3)

--- updated-dependencies: - dependency-name: urllib3 dependency-version: 2.6.3

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

- **deps**: Bump virtualenv from 20.35.4 to 20.36.1
  ([#20](https://github.com/IBM/docling-graph/pull/20),
  [`61678e9`](https://github.com/IBM/docling-graph/commit/61678e9fde19d364745f757a44e5ee2a1cc0cdaa))

Bumps [virtualenv](https://github.com/pypa/virtualenv) from 20.35.4 to 20.36.1. - [Release
  notes](https://github.com/pypa/virtualenv/releases) -
  [Changelog](https://github.com/pypa/virtualenv/blob/main/docs/changelog.rst) -
  [Commits](https://github.com/pypa/virtualenv/compare/20.35.4...20.36.1)

--- updated-dependencies: - dependency-name: virtualenv dependency-version: 20.36.1

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

Co-authored-by: Ayoub El Bouchtili <Ayoub.elbouchtili@fr.ibm.com>

- **deps**: Bump vllm from 0.11.0 to 0.12.0 ([#18](https://github.com/IBM/docling-graph/pull/18),
  [`9eda941`](https://github.com/IBM/docling-graph/commit/9eda941f83a74c9c746be2b8641911ac5c9c834d))

Bumps [vllm](https://github.com/vllm-project/vllm) from 0.11.0 to 0.12.0. - [Release
  notes](https://github.com/vllm-project/vllm/releases) -
  [Changelog](https://github.com/vllm-project/vllm/blob/main/RELEASE.md) -
  [Commits](https://github.com/vllm-project/vllm/compare/v0.11.0...v0.12.0)

--- updated-dependencies: - dependency-name: vllm dependency-version: 0.12.0

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

Co-authored-by: Ayoub El Bouchtili <Ayoub.elbouchtili@fr.ibm.com>

- **deps**: Bump vllm from 0.12.0 to 0.14.0 ([#25](https://github.com/IBM/docling-graph/pull/25),
  [`69e6484`](https://github.com/IBM/docling-graph/commit/69e64843b5c2c438a24ada096aaf43e780acd78b))

Bumps [vllm](https://github.com/vllm-project/vllm) from 0.12.0 to 0.14.0. - [Release
  notes](https://github.com/vllm-project/vllm/releases) -
  [Changelog](https://github.com/vllm-project/vllm/blob/main/RELEASE.md) -
  [Commits](https://github.com/vllm-project/vllm/compare/v0.12.0...v0.14.0)

--- updated-dependencies: - dependency-name: vllm dependency-version: 0.14.0

dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>

Co-authored-by: Ayoub El Bouchtili <Ayoub.elbouchtili@fr.ibm.com>

- **deps**: Optimize dependabot groups
  ([`d707009`](https://github.com/IBM/docling-graph/commit/d707009302799d80830cd28f9472276c034b0150))

Updates the Dependabot configuration to reduce PR noise and consolidate dependency bumps.

Changes: - created a catch-all group for 'github-actions' to bundle all CI/CD updates into a single
  weekly PR. - added 'ipykernel', 'python-semantic-release', and 'rich' to the 'dev-dependencies'
  pip group to prevent them from spawning individual PRs.

Signed-off-by: Ayoub EL BOUCHTILI <ayoub.elbouchtili@fr.ibm.com>

- **doc**: Add project logo to README
  ([`a71f4da`](https://github.com/IBM/docling-graph/commit/a71f4dad6471446fc29037c571dade8d9394e7e9))

- **doc**: Add project logo to README
  ([`23f092b`](https://github.com/IBM/docling-graph/commit/23f092bb3ac51c6c3c765e8c7a8d28ac9ffb9493))

- **doc**: Add project logo to README
  ([`683c18a`](https://github.com/IBM/docling-graph/commit/683c18a77b1c86dcc3de62d45fdb096c00ef4bd8))

- **doc**: Update backend and pipeline settings in README
  ([`4441668`](https://github.com/IBM/docling-graph/commit/4441668c60f95037ad109819164f05bc41a22096))

- **doc**: Update README badges
  ([`bb06a64`](https://github.com/IBM/docling-graph/commit/bb06a64ef6560121bacc78108a335de723660b0a))

- **doc**: Update README badges
  ([`2cefd81`](https://github.com/IBM/docling-graph/commit/2cefd810d031e1494543fa409ae38b049a75d064))

- **doc**: Update README badges
  ([`2c0fc3b`](https://github.com/IBM/docling-graph/commit/2c0fc3b57913998e676181e193cd8331274021ef))

- **doc**: Update todo list
  ([`34eb703`](https://github.com/IBM/docling-graph/commit/34eb7031be569740f6d4368589333eed7da6d23a))

- **doc**: Update todo list
  ([`287a606`](https://github.com/IBM/docling-graph/commit/287a606e27f75c803f2216fbfecbbbad450482c7))

- **docs**: Add guide for installing torch with GPU support
  ([`9b7186c`](https://github.com/IBM/docling-graph/commit/9b7186c3967a5f581b611834444204edea2d5311))

- **docs**: Add license
  ([`f3f2b14`](https://github.com/IBM/docling-graph/commit/f3f2b140b48c35df40cfffcc62b2d9206693c76f))

- **docs**: Add Maintainers
  ([`f91b064`](https://github.com/IBM/docling-graph/commit/f91b0643d2eb9adda67fe6cff8c79744a114cfe2))

- **docs**: Add Maintainers
  ([`92d54ec`](https://github.com/IBM/docling-graph/commit/92d54ec00b823e293d97bffd7e3135ab34ccc377))

- **docs**: Add test suite setup and usage documentation
  ([`a0145ec`](https://github.com/IBM/docling-graph/commit/a0145ec4cb2b7f353b25a3da52e69a209e6528aa))

- **docs**: Add Windows CMD and PowerShell alternatives for setting API keys
  ([`e0f355a`](https://github.com/IBM/docling-graph/commit/e0f355aba9901cdfe37530652fe4449c92767571))

- **docs**: Add workflow diagram for Docling Graph
  ([`f2e0ffa`](https://github.com/IBM/docling-graph/commit/f2e0ffa3d6fe36daee21652b8b231f5bba0b11f8))

- **docs**: Add workflow diagram for Docling Graph
  ([`222c3d0`](https://github.com/IBM/docling-graph/commit/222c3d0feb9e8452d5e10fddc4978ebf5a97a3b3))

- **docs**: Add workflow diagram for Docling Graph
  ([`57fa68a`](https://github.com/IBM/docling-graph/commit/57fa68a0e4d9f1e302276a38dc9f446b79b06d03))

- **docs**: Remove deprecated content
  ([`46ff1d8`](https://github.com/IBM/docling-graph/commit/46ff1d8a78f047fc8a8881627a0fe4c139d7cce6))

- **docs**: Reorganize dir structure for clarity
  ([`d49898e`](https://github.com/IBM/docling-graph/commit/d49898e9c57ec2178c21b66aa4747dce0efb7d6e))

- **docs**: Samples
  ([`f81ff58`](https://github.com/IBM/docling-graph/commit/f81ff58c8e49a582ccd21493eee55c84f75bd4ae))

- **docs**: Simplify and shorten README content
  ([`971988a`](https://github.com/IBM/docling-graph/commit/971988a4f26faaf14abccb03500aedf3afd5b3c3))

- **docs**: Update badges and acknowledgments
  ([`b30a492`](https://github.com/IBM/docling-graph/commit/b30a492bbf4232f3ae09dabca314ca0a4e625b6e))

- **docs**: Update key capabilities
  ([`9b4eaf4`](https://github.com/IBM/docling-graph/commit/9b4eaf4a9562f611cc18da3bd3c95230214ef627))

- **docs**: Update key capabilities
  ([`16fc8ae`](https://github.com/IBM/docling-graph/commit/16fc8ae45d69fb9fc10ae539cb3aa48dae11dd15))

- **docs**: Update list of upcomming features
  ([`598e880`](https://github.com/IBM/docling-graph/commit/598e88090c3b27936ac6e5f0b70e69286bbf3067))

- **docs**: Update next steps in README
  ([`5d0763e`](https://github.com/IBM/docling-graph/commit/5d0763ece0f4984f1b6f0bf0062de27d905415e4))

- **docs**: Update README with CosmoGraph visualization details and CLI commands
  ([`44baf72`](https://github.com/IBM/docling-graph/commit/44baf724416865cac48a434f85233ecff4540c01))

- **docs**: Update roadmap and R&D directions
  ([`54be027`](https://github.com/IBM/docling-graph/commit/54be02732e653490f339d884c38f16c1e0631353))

- **docs**: Update sections with numbered headings for better organization
  ([`a56a73a`](https://github.com/IBM/docling-graph/commit/a56a73ac11828ee976eb2d72054c5238ce1857e5))

- **docs**: Update setup section
  ([`7a708f4`](https://github.com/IBM/docling-graph/commit/7a708f410876643f0a72a5d909c5ded038e5735d))

- **docs**: Update upcoming features in README
  ([`3ab7ce2`](https://github.com/IBM/docling-graph/commit/3ab7ce23c44cb942a45c14c555db14b2157295fd))

- **docs**: Updated docling graph workflow
  ([`583fce2`](https://github.com/IBM/docling-graph/commit/583fce23ac054306e75c0839668c307106166cf7))

- **examples**: Add cosmograph for battery research
  ([`2998007`](https://github.com/IBM/docling-graph/commit/29980070458e9fa31babe78464b3f678363964d1))

- **examples**: Provide sample scripts showcasing core features
  ([`b05283b`](https://github.com/IBM/docling-graph/commit/b05283b6606c91ecac7072f1d876a1f4e20f1efc))

- **examples**: Update interactive graphs examples
  ([`c2dcf31`](https://github.com/IBM/docling-graph/commit/c2dcf3166cf34f8aeb2ce889a122bec9ede49450))

- **examples**: Update Pydantic templates creation guide
  ([`625f246`](https://github.com/IBM/docling-graph/commit/625f246eee7d6301247815f6fe2f692b0233f412))

- **gitignore**: Update to include data and outputs directories while preserving .gitkeep files
  ([`37feca1`](https://github.com/IBM/docling-graph/commit/37feca19977cf24d6ef4fd1918479919f505f73b))

- **mirror**: Migrated repo to internal IBM organization
  ([`e61feae`](https://github.com/IBM/docling-graph/commit/e61feae4d5151c13f2332133376fb4aef574927c))

- **notebooks**: Sync execution logic with refactored module structure
  ([`4c9aa7c`](https://github.com/IBM/docling-graph/commit/4c9aa7ca981f60af29651777280f721e86cfeae9))

- **pre-commit**: Fix linter and typing errors
  ([`aabc35c`](https://github.com/IBM/docling-graph/commit/aabc35c85d9b76e4540d0542230251d13d7ee45e))

- **project**: Add initial boilerplate for Docling-Graph
  ([`8d88b73`](https://github.com/IBM/docling-graph/commit/8d88b73520d4b2ff23aae9533103856c5c06f567))

- Set up project structure with templates, notebooks, converters, and visualizations - Provide
  initial boilerplate for code files and examples - Add README with project overview, core features,
  workflow, and setup instructions

- **tests**: Update tests for core modules
  ([`baccb8f`](https://github.com/IBM/docling-graph/commit/baccb8fa9f2055ee9faa1a002ac452cc0ef5fcdc))

- **tests**: Update tests for llm clients
  ([`bdfada7`](https://github.com/IBM/docling-graph/commit/bdfada723886ee3474abab46cc4947b896e82a6d))

### Continuous Integration

- Bootstrap GitHub Actions (tests, lint, DCO)
  ([`203bc5b`](https://github.com/IBM/docling-graph/commit/203bc5bec79a835b6a98dcc79297f0e988d0a798))

- GitHub Actions for tests and pre-commit checks - DCO sign-off verificationDependabot + PR
  auto-labeling - Security scanning and release workflow docs

Signed-off-by: Ayoub EL BOUCHTILI <ayoub.elbouchtili@fr.ibm.com>

### Features

- Automation setup ([#28](https://github.com/IBM/docling-graph/pull/28),
  [`b404401`](https://github.com/IBM/docling-graph/commit/b4044013e07ddbb36f7f5e81a52483af1200efe0))

* feat: add complete automation setup

Configure semantic versioning with python-semantic-release

Add automated changelog generation

Set up GitHub Pages documentation with MkDocs

Add TestPyPI staging to release workflow

Create automated release notes generation

Add comprehensive documentation

Signed-off-by: Ayoub EL BOUCHTILI <ayoub.elbouchtili@fr.ibm.com>

* docs: minor updates to documentation

* chore: resolve Ruff linter warnings and errors

- **chunker**: New strategy for chunking - dependant of max_new_tokens and context_size
  ([`4f25dc4`](https://github.com/IBM/docling-graph/commit/4f25dc4790dbc96d0bc4e76e66e9b067861443b4))

- **cli**: Add argument for docling pipeline configuration
  ([`42e3016`](https://github.com/IBM/docling-graph/commit/42e3016068c25c650960fc0bbb9a0fd1df65a4f3))

- **cli**: Add inspect command for visualizing graph data in browser
  ([`93e2b6d`](https://github.com/IBM/docling-graph/commit/93e2b6d71af19f6bbd3f489d8e6025fde17160b0))

- **cli**: Enhance init command, streamline extraction process and improve template merging
  ([`c6f8a5b`](https://github.com/IBM/docling-graph/commit/c6f8a5bb404f67cad590985b816dd41d3056b827))

- **core**: Refactor module structure and add base configuration and models for graph conversion
  ([`401846b`](https://github.com/IBM/docling-graph/commit/401846bc803f0c32d456d5c529172bbbd51bef71))

- **docs**: Update README to include Docling Graph workflow and mermaid flowchart
  ([`3d2bc79`](https://github.com/IBM/docling-graph/commit/3d2bc79a26470c47554017817232a9bba4af6107))

- **examples**: Add Pydantic template for Battery Slurry Ontology
  ([`1a7678f`](https://github.com/IBM/docling-graph/commit/1a7678fa06c3bf59e1fa60e611d0dabe9e52bfd0))

- **examples**: Update examples for pydantic templates, data and outputs
  ([`827a096`](https://github.com/IBM/docling-graph/commit/827a096ac1fb1bab335ee1e2de0f754b4486a8ea))

- **exporters**: Add export for docling artifacts
  ([`ad8e559`](https://github.com/IBM/docling-graph/commit/ad8e559740a2954c2cc1f98d42c45c6994a0edd9))

- **exporters**: Enable json format export by default
  ([`bfa9f5b`](https://github.com/IBM/docling-graph/commit/bfa9f5b04daf397b20218dfeb37c45901db94bc8))

- **exporters**: Enable json format export by default
  ([`8ec6030`](https://github.com/IBM/docling-graph/commit/8ec6030e121be012f2cc2962f1a63037ecac2614))

- **graph**: Add advanced and robust graph generation logic
  ([`ed881e5`](https://github.com/IBM/docling-graph/commit/ed881e570bd93c23581f23834f3a2b26c156a208))

- **graph**: Add export to csv and cypher formats
  ([`6eebdd6`](https://github.com/IBM/docling-graph/commit/6eebdd63737eb06225f757ae5205023e94fb327b))

- **graph**: Automatic node deduplication
  ([`ac65140`](https://github.com/IBM/docling-graph/commit/ac651406ef6a8e24ceb0952368e9de66640f553f))

- **graph**: Enhance graph validation and serialization features
  ([`8919aff`](https://github.com/IBM/docling-graph/commit/8919affe32427c06087dc3bf114c353c4cb6429a))

- **init**: Add interactive wizard for config setup
  ([`b884e73`](https://github.com/IBM/docling-graph/commit/b884e7389155a54788bf9cbfc525fc916d5482e8))

- **llm_backend**: Add pydantic model consolidation via llm
  ([`3f1bab2`](https://github.com/IBM/docling-graph/commit/3f1bab2f7138a74dca1376a8c4db51ebb948d838))

- **llm_clients**: Add support for local inference with vLLM
  ([`8ccd6ba`](https://github.com/IBM/docling-graph/commit/8ccd6baadc23918a2485a7ba8c69df961269ed53))

- **notebooks**: Update notebooks following graph moule refactor
  ([`a1ea119`](https://github.com/IBM/docling-graph/commit/a1ea11999f3d51150cee1a4a489b842655c48bb1))

- **notebooks**: Update notebooks following graph moule refactor
  ([`56c8175`](https://github.com/IBM/docling-graph/commit/56c817517b7ac5960ccd0451781b5b68eb4e62a8))

- **pipeline**: Introduce type-safe configuration class and support config injection from Python
  ([`2e72221`](https://github.com/IBM/docling-graph/commit/2e72221158a14509d562f3c4ad9e685f6623636d))

- **templates**: Add insurance models
  ([`b2e738f`](https://github.com/IBM/docling-graph/commit/b2e738fe6b2fa0d6db10cabe8e2a9a1f0be54e52))

- **templates**: Add templates for llm extraction
  ([`80506a2`](https://github.com/IBM/docling-graph/commit/80506a26813df0a77bb7e9011bbfaff8d9c41ac7))

- **templates**: Improve model design and validation rules
  ([`04f3c64`](https://github.com/IBM/docling-graph/commit/04f3c64c14f172903b57035519590b300b73661a))

- **templates**: Update ID card Pydantic template
  ([`49e0bff`](https://github.com/IBM/docling-graph/commit/49e0bff472ecf259d900f2a5b030fa9adcc7f5bc))

- Modify the ID card Pydantic model to include new fields and validations - Ensure compatibility
  with extraction and graph conversion workflow

- **tests**: Add tests suite for docling graph module
  ([`05c8ea3`](https://github.com/IBM/docling-graph/commit/05c8ea3895918a369bcbc10beddad395202e7377))

- **tests**: Extend test suite to cover entire code base
  ([`583eba2`](https://github.com/IBM/docling-graph/commit/583eba23a70bd0a922ab4459b2d1f769bd4298e4))

- **visualizer**: Add fit_view_delay parameter for improved visualization
  ([`2cf81a6`](https://github.com/IBM/docling-graph/commit/2cf81a6c2093058574cd436997d53155494240a0))

- **visualizers**: Add cosmograph alternative for html graph generation
  ([`30fcabe`](https://github.com/IBM/docling-graph/commit/30fcabeb5de347cd2b5badaa2a3d6d0580a94559))

- **visualizers**: Implement cosmograph-based graph generation
  ([`d226293`](https://github.com/IBM/docling-graph/commit/d22629303d62b3c513f4b07cfce9c62a2ec33cbc))

- **visualizers**: Replace cosmograph graph generation with cytoscape
  ([`60fd642`](https://github.com/IBM/docling-graph/commit/60fd6423ab0b51aa841b2fcfe8b4547c9b6a1c89))

- **visualizers**: Replace cosmograph graph generation with cytoscape
  ([`34d125e`](https://github.com/IBM/docling-graph/commit/34d125e0d3be4ddcb5fb76bda1346a211d609291))

- **watsonx**: Implement watsonx llms api
  ([`2dcfe4d`](https://github.com/IBM/docling-graph/commit/2dcfe4d70d05f7eadeb001479bf1487fed17098a))

- **watsonx**: Move example to right path
  ([`37a03e4`](https://github.com/IBM/docling-graph/commit/37a03e4129f83e3141768e890f7bb4d4187ca690))

- **watsonx**: Update readme
  ([`6d97d76`](https://github.com/IBM/docling-graph/commit/6d97d76dc6df419a5275c373097dd56f31557be5))

### Refactoring

- Add cli and update project design
  ([`742e43a`](https://github.com/IBM/docling-graph/commit/742e43ad8b724a5bbd00275be5ded594f99c2454))

- **cli**: Major structure update for cli and configuration handling
  ([`a4836df`](https://github.com/IBM/docling-graph/commit/a4836df82f6423c852b311dadfbb6cb8944287bc))

- **cli**: Major structure update for cli and configuration handling
  ([`44b086b`](https://github.com/IBM/docling-graph/commit/44b086bf71ad64275601d59ae3e4a8bd9088e837))

- **cli**: Reduce redundancy and streamline initialization process
  ([`2f86b4f`](https://github.com/IBM/docling-graph/commit/2f86b4f63fc50343b0c18cc6fa2ef01d7b355578))

- **converter**: Enhance GraphConverter for stateless and thread-safe operations
  ([`97b2bb4`](https://github.com/IBM/docling-graph/commit/97b2bb4ed6eb2b1b31af739f7e7d28f91fb3130b))

- **core**: Relocate files to improve module structure
  ([`4666c57`](https://github.com/IBM/docling-graph/commit/4666c57fef30f7523459ac548a4316e1bf2984f8))

- **core**: Unify date serialization logic with shared utility
  ([`96dee50`](https://github.com/IBM/docling-graph/commit/96dee50456fcf17a232b1f562cef78a782ddaa1e))

- **graph**: Add modular structure with clear separation of concerns
  ([`b66f61f`](https://github.com/IBM/docling-graph/commit/b66f61f3178c82cd77db2dd887d12410e0054ffb))

- **graph**: Add protocol-based interfaces
  ([`e70a80c`](https://github.com/IBM/docling-graph/commit/e70a80c7f75948f621b05e49d129962f06010376))

- **graph**: Major stability and logic upgrade for graph generation and pydantic models
  ([`daf74ab`](https://github.com/IBM/docling-graph/commit/daf74ab4b23884dbfe3889a6844d8ba3631354f5))

- **llm_clients**: Enhance client implementations with structured prompts and error handling
  ([`b25713d`](https://github.com/IBM/docling-graph/commit/b25713df453912a6e9accc8c0945a6486d33a46c))

- **pipeline**: Extraction backends and strategies
  ([`b555edd`](https://github.com/IBM/docling-graph/commit/b555eddac1d36726923d1bc4484257a8e4b5cb27))

- **tests**: Overhaul test suite structure and organization
  ([`64c4f59`](https://github.com/IBM/docling-graph/commit/64c4f59578b0f4e14eb22b816226d227047a98c6))
