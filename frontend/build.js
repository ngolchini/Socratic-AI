const esbuild = require('esbuild');

esbuild.build({
  entryPoints: ['src/DifferentialDiagnosis.tsx'],
  bundle: true,
  minify: true,
  sourcemap: true,
  target: ['es2015'],
  outfile: 'dist/differential.js',
  format: 'iife',
  globalName: 'DifferentialDiagnosis',
  external: ['react', 'react-dom'],
  define: {
    'process.env.NODE_ENV': '"production"'
  }
}).catch(() => process.exit(1));