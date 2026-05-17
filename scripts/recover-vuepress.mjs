import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, extname, join, relative } from 'node:path'

const root = process.cwd()
const docsDir = join(root, 'docs')
const publicDir = join(docsDir, '.vuepress', 'public')

const htmlFiles = [
  'index.html',
  'get-started.html',
  'change-log.html',
  'license.html',
  'intro/index.html',
  'intro/intro.html',
  'intro/download.html',
  'intro/overview.html',
  'intro/FAQ.html',
  'intro/14.html',
  'intro/28.html',
  'feature/index.html',
  'feature/properties.html',
  'feature/properties-14.html',
  'feature/routability features.html',
  'feature/ir drop features.html',
  'feature/graph.html',
  'feature/timing features.html',
  'feature/changelog.html',
  'tutorial/experiment_tutorial.html',
]

const titleOverrides = {
  'intro/index.html': 'Introduction',
  'feature/index.html': 'Introduction',
}

function ensureDir(path) {
  mkdirSync(path, { recursive: true })
}

function copyDir(src, dest) {
  ensureDir(dest)
  for (const entry of readdirSync(src)) {
    const srcPath = join(src, entry)
    const destPath = join(dest, entry)
    if (statSync(srcPath).isDirectory()) {
      copyDir(srcPath, destPath)
    } else if (isPublicAsset(srcPath)) {
      copyFileSync(srcPath, destPath)
    }
  }
}

function isPublicAsset(path) {
  return ['.gif', '.jpeg', '.jpg', '.png', '.svg', '.webp'].includes(extname(path).toLowerCase())
}

function targetMarkdownPath(htmlFile) {
  if (htmlFile === 'index.html') return join(docsDir, 'README.md')
  if (htmlFile.endsWith('/index.html')) return join(docsDir, dirname(htmlFile), 'README.md')
  return join(docsDir, htmlFile.replace(/\.html$/, '.md'))
}

function extractTitle(html, file) {
  if (titleOverrides[file]) return titleOverrides[file]
  const match = html.match(/<title>(.*?)<\/title>/i)
  return match ? decodeEntities(match[1]).trim() : ''
}

function decodeEntities(text) {
  return text
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
}

function stripVueArtifacts(content) {
  return content
    .replace(/<!--\[--><!--\]-->/g, '')
    .replace(/<!--\[-->/g, '')
    .replace(/<!--\]-->/g, '')
    .replace(/<!---->/g, '')
    .trim()
}

function extractContent(html, file) {
  const startMarker = '<div vp-content>'
  const start = html.indexOf(startMarker)
  if (start === -1) return ''

  const rest = html.slice(start + startMarker.length)
  const pageEnd = rest.indexOf('<!--[--><!--]--></div><footer class="vp-page-meta"')
  const homeEnd = rest.indexOf('</div><!--[--><div class="vp-footer"')
  const endCandidates = [pageEnd, homeEnd].filter((index) => index >= 0)

  if (endCandidates.length === 0) {
    throw new Error(`Could not find content end marker for ${file}`)
  }

  const end = Math.min(...endCandidates)
  return stripVueArtifacts(rest.slice(0, end))
}

function frontmatterFor(file, title) {
  if (file === 'index.html') {
    return `---\nhome: true\ntitle: Home\nheroImage: /circuitnet.png\nheroImageDark: /circuitnet-dark.png\nheroText: null\ntagline: A Large-Scale Open-Source AI4EDA Dataset.\nsidebar: true\nactions:\n  - text: Get Started\n    link: /get-started.html\n    type: primary\n  - text: GitHub\n    link: https://github.com/circuitnet/CircuitNet\n    type: secondary\nfeatures:\n  - title: Physical Design\n    details: Data is extracted from commercial physical design flow, including floorplan, powerplan, placement, clock tree synthesis, routing, etc.\n  - title: Samples\n    details: CircuitNet project contains 20K+ samples (including RISC-V CPU, GPU, and AI chip).\n  - title: Technology\n    details: The realistic chip data is based on the commercial 28nm and 14nm PDKs.\n  - title: Data Format\n    details: The data in CircuitNet is packed in the easy-to-use .npz format.\n  - title: Supported Tasks\n    details: CircuitNet supports ML tasks on routability, IR-drop, timing, etc.\n  - title: Tutorial\n    details: CircuitNet is equipped with tutorials for four prediction tasks.\nfooter: BSD 3-Clause License | Copyright © 2022-present CircuitNet Team\n---`
  }

  if (!title) return '---\n---'
  return `---\ntitle: ${JSON.stringify(title)}\n---`
}

function writeRecoveredPage(file) {
  const html = readFileSync(join(root, file), 'utf8')
  const title = extractTitle(html, file)
  const content = extractContent(html, file)
  const target = targetMarkdownPath(file)
  ensureDir(dirname(target))
  writeFileSync(target, `${frontmatterFor(file, title)}\n\n${content}\n`)
  console.log(`recovered ${relative(root, target)}`)
}

function copyPublicAssets() {
  copyDir(join(root, 'assets'), join(publicDir, 'assets'))
  for (const file of ['circuitnet.png', 'circuitnet-dark.png', 'circuitnet-favicon.png']) {
    if (existsSync(join(root, file))) {
      copyFileSync(join(root, file), join(publicDir, file))
    }
  }
}

for (const file of htmlFiles) {
  writeRecoveredPage(file)
}
copyPublicAssets()
