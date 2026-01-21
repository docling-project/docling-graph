# Governance

## Overview

This project aims to be governed in a transparent, accessible way for the benefit of the community. All participation in this project is open and not bound to corporate affilation. Participants are bound to the project's [Code of Conduct](./CODE_OF_CONDUCT.md).

## Project roles

### Contributor

The *contributor* role is the starting role for anyone participating in the project and wishing to contribute code.

#### Process for becoming a contributor

* Review the [Contribution Guidelines](./CONTRIBUTING.md) to ensure your contribution is inline with the project's coding and styling guidelines.
* Submit your code as a PR with the appropriate DCO signoff
* Have your submission approved by the committer(s) and merged into the codebase.

### Committer

A *committer* is a contributor who has the additional privilege to commit code directly to the repository, but also the duty of being a responsible leader in the community.

Current committers (in alphabetical order):

- Michele Dolfi - [@dolfim-ibm](https://github.com/dolfim-ibm)
- Ayoub El Bouchtili - [@ayoub-ibm](https://github.com/ayoub-ibm)
- Peter Staar - [@PeterStaar-IBM](https://github.com/PeterStaar-IBM)

#### Process for becoming a committer

A contributor can be nominated for the committer role by a committer. There will be a vote by the TSC members. While it is expected that most votes will be unanimous, a two-thirds majority of the cast votes is enough.

### Maintainer

A repository *maintainer* is a committer who has the additional priviledge to actually merge pull requests into the main branch of the particular repository.

Each Docling repository has a list of maintainers in its `MAINTAINERS.md` page.

#### Process for becoming a maintainer

A committer can be nominated for becoming maintainer of a given repository by another committer. There will be a vote by the TSC members. While it is expected that most votes will be unanimous, a two-thirds majority of the cast votes is enough.

### TSC member

The *Techincal Steering Committee (TSC) members* are committers who have additional responsibilities to ensure the smooth running of the project. TSC members are expected to participate in strategic planning, and approve changes to the governance model. The purpose of the TSC is to ensure a smooth progress from the big-picture perspective.

Current TSC members (in alphabetical order):

- [Christoph Auer](https://github.com/cau-git)
- [Michele Dolfi](https://github.com/dolfim-ibm)
- [Peter Staar](https://github.com/PeterStaar-IBM)
- [Panos Vagenas](https://github.com/vagenas)

One of the TSC members is the chairperson of the TSC and should ensure the smooth running of the TSC. They do not have more voting power than other TSC members.

Currently [Peter Staar](https://github.com/PeterStaar-IBM) is the chairperson of the TSC.

#### Process for becoming a TSC member

At the moment, the TSC is not open for new members. This is expected to change after the first 24 months.

## Release process

Project releases will occur on a scheduled basis as agreed by the committers.

## Communication

This project, just like all open source, is a global community. In addition to the [Code of Conduct](./CODE_OF_CONDUCT.md), this project will:

* Keep all communication on open channels (mailing list, forums, chat).
* Be respectful of time and language differences between community members (such as scheduling meetings, email/issue responsiveness, etc).
* Ensure tools are able to be used by community members regardless of their region.

If you have concerns about communication challenges for this project, please contact the committers.

## Satellite Projects

Satellite projects are endorsed projects by the Docling team and provide additional capabilities and/or features to Docling.

### Procedure to become a satellite project

To become a satellite project of Docling, you need to follow these steps.

1. Your project has to be open-source repository with a permissive license (eg MIT or Apache-v2)
2. Reach out to the TSC members of Docling to register and review the code. The review will include (but is not limited to)
    - Ensure you have proper Linting
    - Ensure you have proper CI/CD
    - Ensure all dependencies have proper permissive licenses
    - Follow the [OpenSSF](https://www.bestpractices.dev/en) best practices badge: minimum requirement is to have silver badge
3. Hand over the ownership to the Linux Foundation for AI & Data with the help the TSC members of Docling. The minimum information we need for that is,
    - Name, license and repository URL
    - A brief mission statement of that project    
    - The logo (if there is one)
    - The website (if there is one)
4. Move open source repository into Docling project and all related artifacts (eg AI models) are moved to the Docling HuggingFace page

### Governance of Satellite projects

The advantage of satellite projects is that they can benefit from the Docling community and have greater autonomy, i.e. have their own dedicated Maintainer and Committer lists. Nevertheless, satellite projects must fullfill the following requirements,

1. Docling project MAINTAINERS and COMMITTERS keep their role also in Satellite projects
2. Election of Satellite MAINTAINERS: by the TSC of Docling
3. Election of Satellite COMMITTERS: Satellite MAINTAINERS + 1 (at least) member of TSC
