/**
 * Every relative module path in the website's tests, and in `capture/`, `integration/`
 * and `playwright/`, names a file.
 *
 * `tsconfig.app.json` type-checks the TypeScript in `src/` except the tests, and no
 * tsconfig covers `capture/`, `integration/` or `playwright/`. A path in those files
 * that names no file fails nothing on its own: a type-only import is erased before
 * the file runs, and a `vi.mock` of a path nothing imports mocks nothing, so the
 * spec quietly runs the real module.
 *
 * Every quoted relative path in a file is resolved first, and only a file holding one
 * that names no file is parsed, to confirm the path is a module path rather than a
 * string fixture or a comment. Parsing every file can take more than a minute on a
 * loaded CI runner.
 */
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'

import ts from 'typescript'
import { describe, expect, it } from 'vitest'

const SRC = resolve(__dirname, '..')
const WEBSITE = resolve(SRC, '..')
const UNCHECKED_FOLDERS = ['capture', 'integration', 'playwright']
const SOURCE_FILE = /\.(ts|tsx|js|jsx|mjs)$/
const SPEC_FILE = /\.(test|spec)\.[jt]sx?$/
const RESOLVABLE_EXTENSIONS = ['', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.json', '.d.ts']
const MODULE_PATH_CALLS = new Set(['import', 'require', 'vi.mock', 'vi.doMock', 'vi.unmock', 'vi.doUnmock', 'vi.importActual'])
// shortcut: a path spelled with an escape sequence is not matched; widen this if a module path ever needs one.
const QUOTED_RELATIVE_PATH = /(['"])(\.\.?(?:\/[^'"\n]*)?)\1/g

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (entry === 'node_modules') continue
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) walk(full, out)
    else if (SOURCE_FILE.test(entry)) out.push(full)
  }
  return out
}

function uncheckedFiles(): string[] {
  const specs = walk(SRC).filter(file => relative(SRC, file).split(sep)[0] === 'test' || SPEC_FILE.test(file))
  const folders = UNCHECKED_FOLDERS.map(folder => join(WEBSITE, folder)).filter(existsSync)
  return [...specs, ...folders.flatMap(folder => walk(folder))]
}

function calleeName(callee: ts.Expression): string {
  if (callee.kind === ts.SyntaxKind.ImportKeyword) return 'import'
  if (ts.isIdentifier(callee)) return callee.text
  if (ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression)) return `${callee.expression.text}.${callee.name.text}`
  return ''
}

/** The module paths a file names: imports, re-exports, `import()`, `typeof import()`, `require()` and the `vi.mock()` family. */
function modulePathsIn(file: string, text: string): { modulePath: string; line: number }[] {
  const kind = /\.[jt]sx$/.test(file) ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, false, kind)
  const found: { modulePath: string; line: number }[] = []
  const record = (literal: ts.StringLiteral) => {
    found.push({ modulePath: literal.text, line: source.getLineAndCharacterOfPosition(literal.getStart(source)).line + 1 })
  }
  const visit = (node: ts.Node) => {
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
      record(node.moduleSpecifier)
    } else if (ts.isCallExpression(node) && node.arguments[0] && ts.isStringLiteral(node.arguments[0]) && MODULE_PATH_CALLS.has(calleeName(node.expression))) {
      record(node.arguments[0])
    } else if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument) && ts.isStringLiteral(node.argument.literal)) {
      record(node.argument.literal)
    }
    ts.forEachChild(node, visit)
  }
  visit(source)
  return found
}

const isFile = (candidate: string) => existsSync(candidate) && statSync(candidate).isFile()

function namesAFile(from: string, modulePath: string): boolean {
  const target = resolve(dirname(from), modulePath.split('?')[0])
  const stems = new Set([target, target.replace(/\.(js|jsx|mjs)$/, '')])
  return [...stems].some(stem =>
    RESOLVABLE_EXTENSIONS.some(extension => isFile(stem + extension) || (extension !== '' && isFile(join(stem, `index${extension}`)))),
  )
}

/** The module paths in *file* that name no file, as `file:line: path`. */
function missingModulePathsIn(file: string): string[] {
  const text = readFileSync(file, 'utf8')
  const unresolved = new Set(
    [...text.matchAll(QUOTED_RELATIVE_PATH)].map(match => match[2]).filter(quoted => !namesAFile(file, quoted)),
  )
  if (unresolved.size === 0) return []
  return modulePathsIn(file, text)
    .filter(({ modulePath }) => unresolved.has(modulePath))
    .map(({ modulePath, line }) => `${relative(WEBSITE, file).split(sep).join('/')}:${line}: ${modulePath}`)
}

// One read per scanned file, a parse only where a quoted path names no file, and nothing
// awaited: CPU time that grows with the file count and can exceed vitest's 15 s default
// on a loaded host.
const SCAN_TIMEOUT_MS = 60_000

describe('relative module paths in the tests and in capture/, integration/ and playwright/', () => {
  it('each one names a file that exists', { timeout: SCAN_TIMEOUT_MS }, () => {
    expect(uncheckedFiles().flatMap(missingModulePathsIn)).toEqual([])
  })
})
