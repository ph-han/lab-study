// Assemble the .dc.html artboards from parts/ + the shared stylesheet.
// Run from design/bellman:  node build.mjs
import fs from 'node:fs';

const shell = fs.readFileSync('_shell.css', 'utf8');

const BOARDS = [
  { name: 'Main', w: 1440, h: 940 },
  { name: 'CorridorA', w: 1440, h: 660 },
  { name: 'Improve', w: 1440, h: 940 },
  { name: 'ValueIteration', w: 1440, h: 940 },
  { name: 'Rollout', w: 1440, h: 940 },
  { name: 'MyRun', w: 1440, h: 940 },
];

const FONTS =
  'https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800' +
  '&family=Gothic+A1:wght@400;500;700;800&family=JetBrains+Mono:wght@500;700&display=swap';

for (const b of BOARDS) {
  if (!fs.existsSync(`parts/${b.name}.body.html`)) { console.log(`skip ${b.name} (no parts yet)`); continue; }
  const body = fs.readFileSync(`parts/${b.name}.body.html`, 'utf8').trim();
  const logic = fs.readFileSync(`parts/${b.name}.logic.js`, 'utf8').trim();
  const extra = fs.existsSync(`parts/${b.name}.css`)
    ? '\n' + fs.readFileSync(`parts/${b.name}.css`, 'utf8').trim() + '\n'
    : '';
  const out =
`<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="./support.js"><\/script>
</head>
<body>
<x-dc>
<helmet>
  <link rel="stylesheet" href="${FONTS}">
  <style>
${shell}${extra}
  </style>
</helmet>
${body}
</x-dc>
<script data-dc-script data-props='{"$preview":{"width":${b.w},"height":${b.h}}}'>
${logic}
<\/script>
</body>
</html>
`;
  fs.writeFileSync(`${b.name}.dc.html`, out);
  console.log(`built ${b.name}.dc.html  ${(out.length / 1024).toFixed(1)} KB`);
}
