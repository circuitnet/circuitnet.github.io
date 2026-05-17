import { defaultTheme } from '@vuepress/theme-default'
import { viteBundler } from '@vuepress/bundler-vite'

export default {
  base: '/',
  lang: 'en-US',
  title: '',
  description: 'A Large-Scale Open-Source AI4EDA Dataset.',
  bundler: viteBundler(),
  head: [
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css' }],
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js' }],
    ['link', { rel: 'icon', href: '/circuitnet-favicon.png' }],
  ],
  theme: defaultTheme({
    logo: '/circuitnet.png',
    logoDark: '/circuitnet-dark.png',
    colorMode: 'auto',
    colorModeSwitch: true,
    repo: null,
    locales: {
      '/': {
        selectLanguageName: 'English',
      },
    },
    navbar: [
      {
        text: 'Docs',
        link: '/get-started.md',
        children: [
          { text: 'Get Started', link: '/get-started.md' },
          { text: 'Dataset', link: '/intro/intro.md' },
          { text: 'Features', link: '/feature/properties.md' },
          { text: 'Tutorial', link: '/tutorial/experiment_tutorial.md' },
        ],
      },
      { text: 'Download', link: '/intro/download.md' },
      { text: 'GitHub', link: 'https://github.com/circuitnet/CircuitNet' },
    ],
    sidebar: [
      { text: 'Get Started', link: '/get-started.md' },
      {
        text: 'Dataset',
        collapsible: true,
        children: [
          { text: 'Introduction', link: '/intro/intro.md', collapsible: true },
          { text: 'Download', link: '/intro/download.md', collapsible: true },
          { text: 'Overview', link: '/intro/overview.md', collapsible: true },
        ],
      },
      {
        text: 'Features',
        collapsible: true,
        children: [
          { text: 'Basic Properties', link: '/feature/properties.md', collapsible: true },
          { text: 'Routability', link: '/feature/routability features.md', collapsible: true },
          { text: 'IR drop', link: '/feature/ir drop features.md', collapsible: true },
          { text: 'Graph', link: '/feature/graph.md', collapsible: true },
          { text: 'Timing', link: '/feature/timing features.md', collapsible: true },
        ],
      },
      { text: 'Tutorial', link: '/tutorial/experiment_tutorial.md' },
      { text: 'Change Log', link: '/change-log.md' },
      { text: 'FAQ', link: '/intro/FAQ.md' },
      { text: 'License', link: '/license.md' },
    ],
  }),
}
