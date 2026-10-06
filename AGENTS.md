# Agent Instructions

## Commits MUST Follow Conventional Commits

Every commit in this repository MUST follow
[Conventional Commits](https://www.conventionalcommits.org/). Write the subject
as `<type>(<optional scope>): <short imperative description>`.

- Allowed types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `perf`,
  `build`, `ci`, `style`, `revert`.
- Write the description in the imperative, in lower case, without a trailing
  full stop.
- Put ticket keys in the body or footer as `Jira: <TICKET-KEY>`, never in the
  subject.
- Mark breaking changes with `!` before the colon or with a
  `BREAKING CHANGE:` footer.
- Use the same pattern for squash-merged pull request titles.

## Start Here

Read these first:

- `README.md`
- `docs/understanding_model_generation.md`
- `docs/understanding_cleaning.md`
- `docs/keeping_amis_updated.md`

This repository is the FPO Parcel Item Categorisation API. It trains, benchmarks, and runs inference for a model that classifies parcel item descriptions to commodity codes.

## Working Rules

- Keep scope tight and prefer simple changes.
- Verify generated or AI-suggested claims against source code and tests.
- Use `rg` for code search.
- Follow existing patterns.

## Pull request writing

Write all pull request titles and descriptions in ASD-STE100 Simplified Technical English.

- Write short sentences. Put one fact or one instruction in each sentence.
- Use the same word for the same thing. Do not use synonyms.
- Use active voice.
- Use simple present tense for facts. Use simple past tense for completed work.
- Use words that a general technical reader can understand. Keep product names and legal terms when they are necessary.
- Do not write long noun clusters.
- Do not write filler, marketing language, or vague claims.
- Follow the repository pull request template. Fill each required section with concrete facts.
- Write for the reviewer. Do not write a work diary.

## Pull requests and commits

- Follow `.github/pull_request_template.md`.
- Use conventional commits.
- Apply exactly one risk label.
