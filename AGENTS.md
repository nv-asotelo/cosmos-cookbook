# Cosmos Cookbook Agent Guide

This repository ships project-scoped Agent Skills in `.agents/skills/`. Codex
loads those skills from the repository root, and Kimi Code CLI also discovers the
same generic skill directory. Claude Code users can use the mirrored wrappers in
`.claude/skills/` or the existing slash commands in `.claude/commands/`.

When running or authoring recipes, prefer recipe-local `AGENTS.md` guidance when
present, then `CLAUDE.md`, then the human recipe docs such as `inference.md`,
`post_training.md`, or `README.md`. New recipe scaffolds should keep `AGENTS.md`
and `CLAUDE.md` behaviorally identical so Codex, Claude, and Kimi receive the
same operational instructions.

For helper scripts referenced by the BYO-video skills, prefer
`.agents/skills/byo-video/scripts/` from this repository. The legacy
`.claude/scripts/` copies are kept for existing Claude slash-command users.
